# Detailed Code Explanation — Phase 1 & Phase 2

CS265 Systems Project — Zilin Dai

This document explains every piece of code line-by-line: what it does, why I chose that design, and what would break if it were done differently. Written to prepare for oral examination.

---

## Table of Contents

1. [graph_prof.py — Phase 1 Graph Profiler](#1-graph_profpy--phase-1-graph-profiler)
2. [ac_algorithm.py — Phase 2 μ-TWO Algorithm](#2-ac_algorithmpy--phase-2-μ-two-algorithm)
3. [ac_visualize.py — Visualization Module](#3-ac_visualizepy--visualization-module)
4. [run_experiments.py — Experiment Orchestration](#4-run_experimentspy--experiment-orchestration)
5. [How the Pieces Connect End-to-End](#5-how-the-pieces-connect-end-to-end)

---

## 1. graph_prof.py — Phase 1 Graph Profiler

### 1.1 Why inherit from `fx.Interpreter`?

```python
class GraphProfiler(fx.Interpreter):
```

`fx.Interpreter` already knows how to walk an FX graph node-by-node in topological order and execute each node. By subclassing it, we can override `run()` (called once per full graph execution) and `run_node()` (called once per node) to inject our measurement logic. The alternative — writing our own graph walker — would duplicate hundreds of lines of value-passing and garbage-collection logic that `fx.Interpreter` already handles correctly.

**Design choice**: `garbage_collect_values=True` (the default) means the interpreter frees intermediate values once they have no more consumers. This is important because without it, all tensors would stay alive for the whole run, and our memory simulation would be wrong — it wouldn't see tensors being freed.

### 1.2 Static Analysis (lines 46–199) — done once in `__init__`

Everything in `__init__` runs once when the profiler is constructed, before any actual execution. This is called "static analysis" because it examines the graph structure, not runtime values.

#### 1.2.1 Node list and index lookup

```python
self.nodes_list = list(self.module.graph.nodes)
self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}
```

`self.module.graph.nodes` gives us all nodes in topological order. We convert to a list for index access, and build a reverse dict for O(1) index lookup. We need indices to compute idle spans (e.g., "activation created at node 2337, first used in backward at node 4931, idle span = 4931 - 2337 = 2594 steps").

#### 1.2.2 Boundary detection (lines 59–69)

```python
for i, node in enumerate(self.nodes_list):
    if node.target == torch.ops.separator.sep.default:
        self.sep_idx = i
    elif node.target == torch.ops.separator.sep_backward.default:
        self.sep_backward_idx = i
    elif node.target == torch.ops.aten._fused_adam.default:
        self.fused_adam_idx = i
```

The traced graph is one flat list: `[placeholders..., forward ops..., sep, loss ops..., sep_backward, backward ops..., _fused_adam, optimizer ops..., output]`. We find three sentinel nodes:

- `sep` (inserted by `SEPFunction.apply(loss)` in the train step) marks the end of forward.
- `sep_backward` marks the start of backward. The loss computation lives between these two.
- `_fused_adam` marks the start of the optimizer. We use fused Adam specifically so the optimizer appears as a single graph node, making this detection reliable.

**Why not just count nodes?** Because the number of forward/backward nodes varies by model. Sentinel-based detection works regardless of model architecture.

#### 1.2.3 Region classification (lines 83–92)

```python
for i, node in enumerate(self.nodes_list):
    if self.sep_idx is not None and i <= self.sep_idx:
        self.node_region[node] = "forward"
    elif self.sep_backward_idx is not None and i < self.sep_backward_idx:
        self.node_region[node] = "loss"
    elif self.optimizer_start_idx is not None and i < self.optimizer_start_idx:
        self.node_region[node] = "backward"
    else:
        self.node_region[node] = "optimizer"
```

Each node is labeled with its region. The boundary logic: nodes up to and including `sep` are "forward" (since `sep` is just identity, its output is still a forward value). Nodes between `sep` (exclusive) and `sep_backward` (exclusive) are "loss". Nodes from `sep_backward` (inclusive) to `_fused_adam` (exclusive) are "backward". Everything else is "optimizer".

**Why do we need regions?** Two reasons: (1) An activation is defined as a forward-region node consumed by a backward-region node. Without region labels, we can't identify activations. (2) We compute peak memory for forward+backward only, excluding the optimizer, because the optimizer spike (from `_fused_adam` creating copies of all parameters) is not something AC can help with.

#### 1.2.4 Tensor type extraction from `_fused_adam` (lines 113–158)

```python
self.param_nodes: set = set()
self.grad_nodes: set = set()
self.opt_state_nodes: set = set()
if self.fused_adam_idx is not None:
    adam = self.nodes_list[self.fused_adam_idx]
    if isinstance(adam.args[0], (list, tuple)):
        self.param_nodes = set(adam.args[0])
    if isinstance(adam.args[1], (list, tuple)):
        self.grad_nodes = set(adam.args[1])
    for idx in range(2, min(len(adam.args), 6)):
        if isinstance(adam.args[idx], (list, tuple)):
            self.opt_state_nodes.update(adam.args[idx])
```

`_fused_adam`'s signature is `_fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, ...)`. We read the first 6 positional arguments, which are all lists of graph nodes. `args[0]` = parameters, `args[1]` = gradients, `args[2:6]` = optimizer state tensors (Adam's momentum buffers and step counters).

**Why extract from `_fused_adam` rather than from the model?** Because at this point we're working with FX graph nodes, not the original `nn.Module`. The FX graph has already flattened everything into a single sequence. `_fused_adam` is the only place where parameters, gradients, and optimizer states are grouped together in a way we can read.

**`_fused_adam` output classification** (lines 142–158): The node returns a tuple of 3 lists: `(updated_params, updated_exp_avgs, updated_exp_avg_sqs)`. The `getitem` nodes that extract from this tuple need type labels too, otherwise they'd all be classified as OTHER and the peak breakdown would be wrong. We walk two levels of `getitem`: level 1 picks which of the 3 output lists, level 2 picks an individual tensor from that list.

#### 1.2.5 Activation identification (lines 160–168)

```python
for node in self.nodes_list:
    if self.node_region.get(node) == "forward" \
       and node.op in (OP.CALL_FUNCTION, OP.CALL_METHOD) \
       and node not in self.param_nodes \
       and any(self.node_region.get(u) == "backward" for u in node.users):
        self.node_type_map[node] = NodeType.ACT
        self.activation_nodes.append(node)
```

An activation is defined by four conditions:
1. **In forward region** — it's produced during the forward pass.
2. **Is a computation** (`CALL_FUNCTION` or `CALL_METHOD`) — not a placeholder (input) or get_attr (constant).
3. **Not a parameter** — parameters are always available, they aren't activations.
4. **Has at least one backward user** — if nothing in the backward pass reads it, there's no reason to checkpoint it. This is the key condition from the project spec: "activations are stored when generated in the forward pass and freed after their last use in the backward pass."

**Why `any(...)` for backward users?** Some forward nodes are only used by other forward nodes and never by backward. For example, intermediate reshape operations whose output is consumed by the next forward op but not directly by any backward op. These aren't activations and checkpointing them saves nothing.

#### 1.2.6 Activation lifetime analysis (lines 170–188)

```python
for act in self.activation_nodes:
    info = {"name": act.name,
            "created_at": self.node_to_idx[act],
            "last_fwd_use": self.node_to_idx[act],
            "first_bwd_use": None, "last_bwd_use": None}
    for user in act.users:
        uidx = self.node_to_idx[user]
        if self.node_region.get(user) in ("forward", "loss"):
            info["last_fwd_use"] = max(info["last_fwd_use"], uidx)
        elif self.node_region.get(user) == "backward":
            if info["first_bwd_use"] is None or uidx < info["first_bwd_use"]:
                info["first_bwd_use"] = uidx
            if info["last_bwd_use"] is None or uidx > info["last_bwd_use"]:
                info["last_bwd_use"] = uidx
```

For each activation, we record four timestamps:
- `created_at`: when the op that produces it runs (= its own index).
- `last_fwd_use`: the last forward (or loss) node that reads it.
- `first_bwd_use`: the first backward node that reads it.
- `last_bwd_use`: the last backward node that reads it.

The **idle span** = `first_bwd_use - last_fwd_use`. This is the number of execution steps where the activation sits in memory doing nothing — waiting for the backward pass to reach it. Earlier-layer activations have longer idle spans because backward processes layers in reverse order.

**Why `last_fwd_use` starts at `created_at`?** Because the activation is "used" at the step it's created (the node itself is a consumer of its own output, in the sense that the output must exist). If no forward node reads it after creation, `last_fwd_use = created_at`.

**Why track both `first_bwd_use` and `last_bwd_use`?** Phase 2 needs `first_bwd_use` to know when to insert recomputation (just before the backward needs it). Phase 3 needs `last_bwd_use` to know when the recomputed value can be freed.

#### 1.2.7 Last-user precomputation (lines 190–199)

```python
self.last_user: Dict[fx.Node, fx.Node] = {}
for node in reversed(self.nodes_list):
    for inp in node.all_input_nodes:
        if inp not in self.last_user:
            self.last_user[inp] = node
```

For each node, `last_user[node]` is the last node in the graph that reads `node`'s output. When we reach `last_user[node]` during execution, we know `node`'s output will never be needed again and can be freed.

**Why iterate in reverse?** We walk backwards through the graph. The first time we see a node as someone's input, that's its last use (since we're going backwards). The `if inp not in self.last_user` check ensures we only record the first (= latest) occurrence.

**This is the same garbage collection logic that `fx.Interpreter` uses internally.** By replicating it, our memory simulation in `run_node` matches the interpreter's actual behavior.

### 1.3 Runtime Instrumentation

#### 1.3.1 `run_node()` — per-node measurement (lines 257–292)

```python
def run_node(self, n: fx.Node) -> Any:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = super().run_node(n)
    end.record()
    torch.cuda.synchronize()
    self._node_runtimes[n.name].append(start.elapsed_time(end))
```

**Why CUDA events instead of `time.time()`?** GPU operations are asynchronous — `time.time()` would only measure the time to *enqueue* the kernel, not to *execute* it. CUDA events are recorded in the GPU's command stream and `elapsed_time` measures actual GPU wall clock time between them.

**Why `torch.cuda.synchronize()` after every node?** Without synchronization, the GPU would pipeline operations and we couldn't attribute time to individual nodes. The sync forces the CPU to wait until the GPU finishes, giving us accurate per-node timing. This slows down profiling (~3x slower than normal execution) but is necessary for correctness.

```python
mem = self._tensor_mem(result)
self._node_output_mem[n.name] = mem
if mem > 0 and n.op != OP.OUTPUT:
    self._alive[n] = mem
for inp in n.all_input_nodes:
    if self.last_user.get(inp) == n:
        self._alive.pop(inp, None)
```

After execution, we track memory: add the new tensor to `_alive`, and remove any inputs whose last user is this node. The `n.op != OP.OUTPUT` check prevents the graph's final output node from being added to the alive set (it's a bookkeeping node, not a real tensor).

**The alive-set evolution** gives us a memory timeline: at each step, `sum(self._alive.values())` is the total memory in use. The breakdown by `NodeType` tells us how much is params, activations, gradients, etc.

#### 1.3.2 `run()` — per-iteration bookkeeping (lines 232–255)

```python
def run(self, *args, ...) -> Any:
    self._alive = {}
    self._timeline = []
    result = super().run(...)
    peak = max(self._timeline, key=lambda e: e["total_memory"])
    self._peak_per_iter.append(peak["total_memory"])
    fwdbwd = [e for e in self._timeline
              if self.node_region.get(...) in ("forward", "loss", "backward")]
    fb_peak = max(fwdbwd, key=lambda e: e["total_memory"])
    self._fwdbwd_peak_per_iter.append(fb_peak["total_memory"])
```

Each `run()` call executes the full graph once. We reset the alive set and timeline, execute, then find two peaks: the overall peak (which often lands on `_fused_adam` in the optimizer) and the forward+backward peak (what activation checkpointing targets).

**Why track two peaks?** The overall peak is dominated by the optimizer's `_fused_adam` node, which temporarily holds both old and new copies of all parameters and optimizer states. Activation checkpointing can't reduce this — it only affects the forward+backward region. So we report the fwd+bwd peak separately for meaningful comparisons.

#### 1.3.3 `aggregate_stats()` (lines 294–317)

Averages the per-iteration measurements (3 measurement iterations after 2 warmup). The warmup iterations are needed because the first GPU kernel launches are slower due to CUDA context initialization, JIT compilation, and memory allocator warm-up.

---

## 2. ac_algorithm.py — Phase 2 μ-TWO Algorithm

### 2.1 Data Structures

#### 2.1.1 `ActivationInfo` (lines 32–50)

```python
@dataclass
class ActivationInfo:
    node: fx.Node           # FX graph node reference
    name: str               # node name string (for dict keys and set ops)
    memory_size: int        # bytes — output tensor size
    op_runtime: float       # ms — time of the op that creates this activation

    recomp_time: float      # ms — starts at op_runtime, grows with side-effects
    recomp_srcs: Set[str]   # activation inputs needed to recompute this
    recompute_ratio: float  # memory_size / recomp_time (the greedy ranking key)
    recompute: bool         # final decision
```

**Why `@dataclass`?** We need mutable state (`recomp_time`, `recomp_srcs`, `recompute_ratio` change during the algorithm) with many fields. A dataclass gives us `__init__`, `__repr__`, and type hints for free.

**Why store both `node` and `name`?** We need `node` for accessing FX graph edges (`node.all_input_nodes`). We need `name` (a string) for set operations, dict keys, and serialization — `fx.Node` objects can't be meaningfully compared across different graph instances.

**Why `recomp_srcs` is `Set[str]` not `Set[fx.Node]`?** Set operations (`discard`, `update`, `in`) are the core of the side-effect propagation loop. String comparison is simpler and more reliable than node object identity.

**`recomp_time` vs `op_runtime`**: `op_runtime` is the raw profiler measurement — the time for the single op that produces this activation. `recomp_time` is the *total* time needed to recompute it from currently-available sources. Initially `recomp_time = op_runtime`, but when a dependency is evicted, `recomp_time` grows because we must first recompute the dependency.

#### 2.1.2 `ACDecision` (lines 53–61)

```python
@dataclass
class ACDecision:
    retain: List[str]
    recompute: List[str]
    baseline_peak: float      # bytes
    projected_peak: float     # bytes
    memory_saved: float       # bytes
    iteration_log: List[Dict] # one dict per greedy step
```

**Why `iteration_log`?** The greedy convergence plot needs to know the peak memory after each step. Without the log, we'd have to re-run the algorithm to generate the plot. Each log entry records: step number, which activation was evicted, its memory/time/ratio, and the projected peak after eviction.

### 2.2 `_build_activation_infos()` (lines 68–107)

```python
def _build_activation_infos(profiler: GraphProfiler) -> Dict[str, ActivationInfo]:
    act_set = set(n.name for n in profiler.activation_nodes)
    ...
    for act_node in profiler.activation_nodes:
        srcs = set()
        for inp in act_node.all_input_nodes:
            if inp.name in act_set:
                srcs.add(inp.name)
```

This bridges Phase 1 (profiler) and Phase 2 (algorithm). For each activation node in the FX graph, it:

1. Reads `memory_size` from `profiler._node_output_mem` (measured in Phase 1's `run_node()`).
2. Reads `op_runtime` from `profiler.avg_runtimes` (measured via CUDA events).
3. Reads lifetime info from `profiler.act_info` (computed in static analysis).
4. Computes `recomp_srcs`: the set of *other activations* that are direct inputs to this op.

**Why only include activation inputs in `recomp_srcs`?** Parameters and placeholders are always in GPU memory throughout the iteration — they never get evicted, so they're always available for recomputation. Including them would be wrong because it would imply they might need to be recomputed too.

```python
ai.recompute_ratio = mem / max(ai.recomp_time, 1e-6)
```

**Why `max(..., 1e-6)` instead of just dividing?** Some operations like `view`, `transpose`, and `t` have near-zero runtime (~0.001 ms) because they only change tensor metadata, not data. Without the guard, `mem / 0.0 = inf`, which technically works for comparison but breaks floating-point arithmetic in edge cases. The `1e-6` floor is small enough to not affect the ranking (even 1e-6 ms gives a ratio of ~10^15 for a 6 MB tensor, which correctly dominates).

### 2.3 `_precompute_peak_contributors()` (lines 114–155)

This is the key performance optimization. Before explaining the code, let me explain the problem it solves.

**The problem**: The naive algorithm calls `simulate_peak_memory()` after every greedy step to check if the budget is met. For ResNet-152: 777 activations × 8679 nodes per simulation = 6.7 million node visits. In Python, this takes 20+ minutes.

**The insight**: When we evict an activation, peak memory only decreases if that activation was alive at the peak node. If it was already freed before the peak, evicting it changes nothing.

```python
# Find the fwd+bwd peak entry in the timeline
fwdbwd_entries = [e for e in tl
    if profiler.node_region.get(name_to_node.get(e["node_name"]))
       in ("forward", "loss", "backward")]
peak_entry = max(fwdbwd_entries, key=lambda e: e["total_memory"])
peak_idx = next(i for i, e in enumerate(tl) if e is peak_entry)
```

We find which timeline entry has the highest memory during forward+backward. We use `is` (identity) instead of `==` (equality) in the `next()` call because we want the exact same dict object, not a dict with matching values.

```python
# Replay alive-set to find what's alive at that specific node
alive: Dict[fx.Node, int] = {}
for i, node in enumerate(profiler.nodes_list):
    mem = profiler._node_output_mem.get(node.name, 0)
    if mem > 0 and node.op != "output":
        alive[node] = mem
    for inp in node.all_input_nodes:
        if profiler.last_user.get(inp) == node:
            alive.pop(inp, None)
    if i == peak_idx:
        break
alive_acts = {n.name for n in alive if n.name in act_names}
```

We replay the exact same alive-set logic from `run_node()`, but stop at the peak node. The remaining `alive` dict tells us exactly which tensors are in memory at that moment. We filter to only activation names.

**Now the greedy loop can do**: `if best_name in peak_alive_acts: estimated_peak -= best.memory_size`. This is O(1) — a set lookup and a subtraction.

**Trade-off**: This is an approximation. After evicting many activations, the true peak might shift to a different node. But in practice, the peak stays in the same region (early backward, where all forward activations are still alive). We fix any inaccuracy with one full simulation at the end (line 312).

### 2.4 `simulate_peak_memory()` (lines 162–211)

This is the full-accuracy simulation, used once at the end to get the real projected peak and a full timeline for visualization.

```python
adjusted_last_user: Dict[fx.Node, fx.Node] = dict(profiler.last_user)
for act_node in profiler.activation_nodes:
    if act_node.name in recomputed_names:
        info = act_info[act_node]
        adjusted_last_user[act_node] = nodes[info["last_fwd_use"]]
```

**The core idea**: For retained activations, `last_user` stays unchanged — they're freed after their last backward consumer, same as baseline. For recomputed activations, we override `last_user` to be the `last_fwd_use` node. This means the activation gets freed right after the forward pass is done with it, because during backward it will be recomputed just-in-time by Phase 3's graph rewriter.

**Why `dict(profiler.last_user)` (copy) instead of modifying in place?** The profiler's `last_user` is shared state. If we modified it, subsequent calls to `simulate_peak_memory()` (e.g., in the budget sweep) would see corrupted data. The copy ensures each simulation is independent.

The rest of the function is the same alive-set walk as `run_node()`, but using `adjusted_last_user` for the free-after-last-use check.

### 2.5 `mu_two_algorithm()` (lines 218–325) — The Main Algorithm

#### 2.5.1 Budget default (lines 248–251)

```python
if memory_budget is None:
    act_mem = sum(profiler._node_output_mem.get(n.name, 0)
                  for n in profiler.activation_nodes)
    memory_budget = baseline_peak - act_mem * 0.5
```

If no budget is given, default to "remove 50% of activation memory from the peak." This is aggressive but reasonable — it only targets activations, leaving params/grads/opt_states untouched. The 50% target means the algorithm will stop once it's freed half the activation memory at the peak.

#### 2.5.2 Initialization (lines 253–264)

```python
infos = _build_activation_infos(profiler)
peak_mem, peak_alive_acts = _precompute_peak_contributors(profiler)
estimated_peak = peak_mem
recomputed: Set[str] = set()
candidates: Set[str] = set(infos.keys())
```

Build per-activation data, precompute the peak contributors, initialize the estimated peak at baseline, and start with all activations as candidates.

#### 2.5.3 Greedy loop (lines 266–302)

```python
while candidates and estimated_peak > memory_budget:
    best_name = max(candidates, key=lambda n: infos[n].recompute_ratio)
    best = infos[best_name]
```

**Selection**: Pick the activation with the highest `recompute_ratio = memory_size / recomp_time`. This heuristic maximizes memory freed per unit of added compute — the "best bang for the buck" eviction.

**Why `max()` instead of a heap?** With <800 candidates, linear scan (`max`) is fast enough (~0.1ms per step) and simpler than maintaining a heap that needs updates when side-effects change ratios.

```python
    if best.memory_size == 0:
        candidates.discard(best_name)
        continue
```

Some activations (like `getitem` extracting metadata) have 0 bytes. They would have infinite ratio but evicting them saves nothing. Skip them.

```python
    recomputed.add(best_name)
    candidates.discard(best_name)

    if best_name in peak_alive_acts:
        estimated_peak -= best.memory_size
```

Mark as recomputed, remove from candidates, and do the fast peak update. The `in` check is O(1) because `peak_alive_acts` is a set.

```python
    for cand_name in list(candidates):
        cand = infos[cand_name]
        if best_name in cand.recomp_srcs:
            cand.recomp_srcs.discard(best_name)
            cand.recomp_srcs.update(best.recomp_srcs)
            cand.recomp_time += best.recomp_time
            cand.recompute_ratio = cand.memory_size / max(cand.recomp_time, 1e-6)
```

**Side-effect propagation** — the most important part of the algorithm. When we evict activation A, any remaining candidate B that depends on A must now recompute A first. Concretely:

1. `cand.recomp_srcs.discard(best_name)`: B no longer has A as a direct source (A is gone).
2. `cand.recomp_srcs.update(best.recomp_srcs)`: B inherits A's sources — whatever A needed, B now needs too.
3. `cand.recomp_time += best.recomp_time`: B's recomputation now includes A's cost.
4. Ratio decreases: B becomes a less attractive candidate for future eviction.

**Example**: Suppose `relu` depends on `convolution`, and `convolution` depends on nothing (its inputs are parameters). Initially, `relu.recomp_srcs = {"convolution"}` and `relu.recomp_time = 0.02ms`. If we evict `convolution` (recomp_time = 0.05ms), then `relu.recomp_srcs` becomes `{}` (convolution's sources were empty) and `relu.recomp_time` becomes `0.02 + 0.05 = 0.07ms`. The ratio drops from `12.25 / 0.02 = 612M` to `12.25 / 0.07 = 175M`.

**Why `list(candidates)` in the for loop?** We're iterating over `candidates` while potentially discarding from it (via `cand.recomp_srcs.discard`). Actually, we don't modify `candidates` inside this loop, so `list()` is a defensive copy that prevents issues if the code were to change. It also ensures deterministic iteration order.

#### 2.5.4 Final simulation (lines 311–316)

```python
projected_peak, _ = simulate_peak_memory(profiler, recomputed)
if iteration_log:
    iteration_log[-1]["projected_peak_mb"] = projected_peak / (1024**2)
```

One full simulation gives us the accurate projected peak. We update the last log entry so the convergence plot ends on the true value, not the estimated one.

### 2.6 `budget_sweep()` (lines 332–349)

Runs the algorithm at 10 budget levels (95% down to 50% of baseline). Each run is independent — fresh `_build_activation_infos()` call, clean state. Used by the Pareto frontier visualization to plot the compute-memory tradeoff curve.

### 2.7 `format_ac_decision()` (lines 356–396)

Pretty-printer that generates the `ac_decision_*.txt` reports. Shows the greedy iteration log (every step with evicted name, memory, time, ratio, projected peak) and the final retain/recompute lists. Truncates at 20 entries per list for readability.

---

## 3. ac_visualize.py — Visualization Module

### 3.1 Design Principle: Line Charts Over Bar Charts

The TA feedback was: "use line graphs instead of bar graphs for memory snapshots." The original Phase 1 code used bar charts at 5 discrete snapshot points (end-of-forward, start-of-backward, etc.). This hides everything between those points — you can't see when the peak actually occurs or how memory evolves continuously.

All memory-over-time plots now use `fill_between` (stacked area) or `plot` (line), with x-axis = execution step (node index from 0 to ~8000) and y-axis = memory in MB.

### 3.2 `_add_region_spans()` (lines 47–67)

Helper that draws colored background bands for forward/loss/backward/optimizer regions. Uses `axvspan` with low alpha (0.3) so it doesn't obscure the data. Walks the timeline to find region transition points.

### 3.3 `plot_memory_timeline()` (lines 74–134)

Stacked area chart. For each tensor type, we build a numpy array of memory values at each step, then `fill_between(x, bottom, bottom + vals)` with cumulative bottoms. The stacking order (params first, other last) puts the most stable components at the bottom and the most variable (activations) in the middle where changes are visible.

```python
stacks = {nt: np.array([e["breakdown"].get(nt, 0) / MB for e in tl]) for nt in TYPE_ORDER}
bottoms = np.zeros(len(tl))
for nt in TYPE_ORDER:
    ax.fill_between(x, bottoms, bottoms + stacks[nt], ...)
    bottoms += stacks[nt]
```

### 3.4 `plot_memory_comparison()` (lines 141–208)

The most important Phase 2 visualization. Shows baseline (dashed gray) overlaid with the AC timeline (solid fill). The green hatched area between the two curves shows exactly where and how much memory is saved.

```python
_, tl_ac = simulate_peak_memory(profiler, recomputed_names)
```

We call `simulate_peak_memory()` to get the AC timeline. This runs the full graph simulation with adjusted `last_user` for recomputed activations.

**Two-color fill for the AC timeline**: Blue (`#90CAF9`) for non-activation memory (params, grads, opt states), orange (`#FF9800`) for remaining activations. This makes it visually clear that AC shrinks the orange region while leaving everything else unchanged.

### 3.5 `plot_pareto_frontier()` (lines 255–295)

Calls `budget_sweep()` to run the algorithm at 10 budget levels. For each, it computes:
- y = projected peak memory (from `decision.projected_peak`)
- x = recomputation overhead = sum of `op_runtime` for all evicted activations

**Why `op_runtime` not `recomp_time`?** `recomp_time` includes cascading costs from dependencies, which would double-count shared recomputations. `op_runtime` is the direct additional compute each evicted activation adds.

### 3.6 `plot_recompute_ratio_distribution()` (lines 349–402)

Scatter plot on log-log axes. Each activation is a dot positioned at `(recomp_time, memory_size)`. Red dots = recomputed, blue dots = retained. Diagonal dashed lines show iso-ratio contours (lines where `memory/time = constant`).

**Why log-log scale?** Activations span 5+ orders of magnitude in both dimensions: from 8 bytes (`getitem` extracting a scalar) to 25 MB (`view` of a large batch), and from 0.001 ms (`t_*` transpose) to 0.5 ms (large `convolution`). Linear axes would compress 99% of points into one pixel.

---

## 4. run_experiments.py — Experiment Orchestration

### 4.1 `make_experiment()` (lines 75–129)

Builds model, synthetic inputs, optimizer, and a `train_step` closure. Two models:
- **ResNet-152**: `resnet152()` from torchvision. Input: `(bs, 3, 224, 224)` random images. Loss: cross-entropy.
- **BERT-base**: `BertModel(config)` from transformers. Input: random token IDs and attention mask. Loss: MSE on [CLS] embedding.

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4, fused=True, capturable=True)
```

**Why `fused=True`?** Fused Adam performs the parameter update as a single CUDA kernel (`_fused_adam`), which appears as one node in the FX graph. Without `fused`, the optimizer uses separate `_foreach_*` operations that are harder to identify as optimizer ops. **Why `capturable=True`?** Required for fused Adam with graph capture — it pre-allocates step counters on the GPU so they can be captured in the graph trace.

```python
for p in model.parameters():
    if p.requires_grad:
        p.grad = torch.rand_like(p)
optimizer.step()
optimizer.zero_grad()
```

**Why pre-initialize optimizer state?** Adam has per-parameter state (exp_avg, exp_avg_sq) that's lazily initialized on the first `.step()`. If we didn't pre-initialize, the first profiling iteration would include one-time allocation costs and the optimizer state tensors wouldn't exist in the graph.

### 4.2 `profile()` (lines 133–161)

```python
def graph_transformation(gm, args):
    gp = GraphProfiler(gm)
    with torch.no_grad():
        for _ in range(2): gp.run(*args)   # warmup
        gp.reset_stats()
        for _ in range(3): gp.run(*args)   # measurement
        gp.aggregate_stats()
    profiler_box["p"] = gp
    return gm
```

The `graph_transformation` callback is called by the `compile()` function from `graph_tracer.py` after the graph is traced. It receives the traced `GraphModule` and its flattened inputs. We create a profiler, run warmup (2 iters), reset, measure (3 iters), and aggregate.

**Why `profiler_box` dict instead of return?** The callback must return the `GraphModule` (possibly transformed). We can't return the profiler from it, so we use a mutable container (a dict) to smuggle the profiler out.

**Why `torch.no_grad()`?** The graph already contains explicit backward operations (they were traced). We don't want PyTorch to build a second autograd graph on top of our already-traced one.

### 4.3 `run_phase1()` and `run_phase2()` (lines 168–321)

Phase 1 profiles all 8 (model, batch_size) combinations and stores profilers in a dict keyed by `(model, bs)`. Phase 2 reuses these profilers — no re-profiling needed.

**Why store profilers, not just results?** Phase 2 needs the full profiler object (activation nodes, timing data, memory timeline, act_info dict, last_user map) to run the algorithm. Just storing peak memory numbers wouldn't be enough.

Phase 2 generates detailed visualizations only for `DEFAULT_BS` (bs=4), not all 8 configurations. Generating 4 plots x 8 configs = 32 plots would be excessive and most would be redundant.

---

## 5. How the Pieces Connect End-to-End

```
run_experiments.py main()
├── run_phase1()
│   ├── for each (model, bs):
│   │   ├── make_experiment()          → model, optimizer, inputs, train_step
│   │   ├── profile()
│   │   │   ├── compile(train_step)    → traces FX graph via graph_tracer.py
│   │   │   ├── GraphProfiler(gm)      → static analysis in __init__
│   │   │   ├── gp.run() x5            → runtime measurement via run_node()
│   │   │   └── gp.aggregate_stats()   → averages over 3 measured iterations
│   │   ├── save profiling_stats.txt
│   │   └── plot_memory_timeline()     → stacked area PNG
│   └── plot_peak_memory_line()        → batch-size comparison PNG
│
├── run_phase2(results_no_ac, profilers)
│   ├── for each (model, bs):
│   │   ├── mu_two_algorithm(profiler, budget)
│   │   │   ├── _build_activation_infos()       → extract from profiler
│   │   │   ├── _precompute_peak_contributors() → who's alive at peak?
│   │   │   ├── greedy loop                     → pick max ratio, propagate
│   │   │   └── simulate_peak_memory()          → one final accurate simulation
│   │   ├── save ac_decision.txt
│   │   └── for default_bs only:
│   │       ├── plot_memory_comparison()
│   │       ├── plot_greedy_convergence()
│   │       ├── plot_pareto_frontier()           → calls budget_sweep() internally
│   │       └── plot_recompute_ratio_distribution()
│   └── plot_peak_memory_with_ac()
│
└── print_summary()
```

**Data flow**: Phase 1's `GraphProfiler` produces five things that Phase 2 needs:
1. `activation_nodes` — list of activation FX nodes
2. `act_info` — per-activation lifetime data (created_at, last_fwd_use, first_bwd_use)
3. `avg_runtimes` — per-node average execution time in ms
4. `_node_output_mem` — per-node output tensor size in bytes
5. `memory_timeline` — full list of memory snapshots at each step (for peak precomputation)

Phase 2's `mu_two_algorithm()` produces an `ACDecision` containing:
1. `retain` / `recompute` — the partition (input to Phase 3's graph rewriter)
2. `projected_peak` / `memory_saved` — for reporting
3. `iteration_log` — for the convergence plot

Phase 3's `apply_ac_to_graph()` uses `ACDecision.recompute` to:
1. For each recomputed activation, call `find_recomp_inputs()` to walk backward to available nodes
2. Call `_extract_graph_with_inputs_outputs()` to get the minimal recomputation subgraph
3. Insert that subgraph into the backward pass just before `first_bwd_use`
4. Call `replace_subsequent_uses_of()` to redirect backward uses to the recomputed node

---

## 6. graph_rewriter.py — Phase 3 Graph Rewriter

### 6.1 `replace_subsequent_uses_of()` (lines 29–44)

```python
def replace_subsequent_uses_of(graph, old_node, new_node):
    old_node_users = dict(old_node.users)  # snapshot
    for node in reversed(list(graph.nodes)):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)
```

This replaces uses of `old_node` with `new_node`, but ONLY for nodes that appear after `new_node` in the graph. This is critical for correctness:

- **Forward uses must stay on `old_node`**: The forward pass executes first and needs the original activation. The recomputed node doesn't exist yet during forward.
- **Backward uses switch to `new_node`**: By the time backward reaches the insertion point, the original activation has been freed (Phase 2's decision). The recomputed version is now available.

**Why `dict(old_node.users)` snapshot?** `node.users` is a live view that changes when `replace_input_with` is called. Iterating over a mutating dict raises `RuntimeError`. The snapshot freezes the user set.

**Why reversed iteration?** We walk backwards from the graph end. When we hit `new_node`, we stop — everything before it is "forward" and stays untouched.

### 6.2 `find_recomp_inputs()` (lines 51–100)

```python
def find_recomp_inputs(act_node, retained_names, act_names):
    inputs = []
    seen = set()
    stack = list(act_node.all_input_nodes)
    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        is_recomputed_act = (node.name in act_names) and (node.name not in retained_names)
        if is_recomputed_act:
            stack.extend(node.all_input_nodes)
        else:
            inputs.append(node)
    return deduplicated(inputs)
```

This is a backward BFS/DFS through the FX graph. Starting from the activation's direct inputs, it walks backwards:

- **If a node is a recomputed activation**: It won't be in memory during backward. We trace through it by pushing its inputs onto the stack. The subgraph extractor will automatically include the ops needed to recompute it.
- **If a node is anything else** (placeholder, param, retained activation, non-activation op): It will be in memory. We stop here and add it to the inputs list.

**Example**: Forward graph: `param → conv → relu → pool`. If `relu` is recomputed and `conv` is retained:
- `find_recomp_inputs(relu, retained={conv}, acts={conv, relu, pool})` starts with `relu`'s input = `[conv]`
- `conv` is in `retained_names` → it's available → add to inputs
- Result: `[conv]`
- The subgraph from `conv` to `relu` is just the `relu` op itself.

If both `relu` AND `conv` are recomputed:
- Start with `relu`'s input = `[conv]`
- `conv` is in `act_names` but NOT in `retained_names` → recomputed → trace through it
- Push `conv`'s input = `[param]`
- `param` is not in `act_names` → available → add to inputs
- Result: `[param]`
- The subgraph from `param` to `relu` includes both the `conv` and `relu` ops.

**Why `seen` set for deduplication?** In graphs with skip connections (like ResNet), multiple paths can lead to the same input node. Without deduplication, we'd pass duplicate inputs to `_extract_graph_with_inputs_outputs`, which would create duplicate placeholders.

### 6.3 `apply_ac_to_graph()` (lines 107–200) — Main Rewriting Loop

#### Processing order (lines 141–148)

```python
recomp_acts.sort(key=lambda x: x[1]["first_bwd_use"])
```

We process activations in order of their first backward use (earliest first). This matters because `inserting_before(node)` inserts relative to a specific node — earlier insertions don't shift later insertion points since FX tracks node ordering by linked-list pointers, not indices.

#### In-place to out-of-place conversion (`convert_inplace_to_outofplace`)

```python
INPLACE_TO_OUTOFPLACE = {
    torch.ops.aten.relu_.default: torch.ops.aten.relu.default,
}

def convert_inplace_to_outofplace(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in INPLACE_TO_OUTOFPLACE:
            node.target = INPLACE_TO_OUTOFPLACE[node.target]
```

**The problem**: In-place ops like `relu_` mutate their input tensor rather than allocating a new output. `_extract_graph_with_inputs_outputs` needs a distinct output tensor to define the subgraph boundary — it rejects in-place ops with "Node was invalid, but is output."

**The fix**: Before extraction, convert every in-place op to its out-of-place equivalent by swapping the node's `target` attribute. `aten.relu_` becomes `aten.relu` — identical math (`max(0, x)`) but the out-of-place version allocates a new tensor.

**Why this works**: Changing `target` on an FX node is like changing which function a call instruction invokes. The node's arguments, users, and position in the graph stay the same. The only behavioral difference is memory allocation: `relu` creates a new tensor (extractable), while `relu_` reuses the input (not extractable).

**Trade-off**: Out-of-place ops use more memory during forward (each ReLU allocates a new tensor instead of reusing the convolution's output buffer). But this extra memory is exactly what AC frees — these activations are freed after their last forward use. Net effect is strictly positive for checkpointed activations.

**Result**: ResNet-152 has 151 `relu_` nodes in its graph. After conversion, all 74 recomputed activations are successfully extracted and rewritten (previously only 14/78).

#### Subgraph extraction

```python
recomp_subgraph = _extract_graph_with_inputs_outputs(
    joint_graph=gm.graph,
    inputs=current_inputs,
    outputs=[current_act],
    outputs_descs=[AOTOutput()],
)
```

`_extract_graph_with_inputs_outputs` (from `torch._functorch.partitioners`) finds the minimal subgraph: all nodes on any path from `inputs` to `outputs`. It returns a new `fx.Graph` where inputs become placeholders and outputs become the return.

**`outputs_descs=[AOTOutput()]`**: Required by PyTorch 2.5. `AOTOutput` is a descriptor with no fields — just a type marker. One descriptor per output.

The `try/except` around this call handles edge cases (e.g., outputs that reference nodes outside the subgraph).

#### Node insertion (lines 195–216)

```python
with gm.graph.inserting_before(current_first_bwd):
    for n in recomp_subgraph.nodes:
        if n.op == "placeholder" or n.op == "output":
            continue
        new_node = gm.graph.node_copy(
            n, arg_transform=lambda arg: name_to_node[arg.name]
        )
```

`gm.graph.inserting_before(node)` is a context manager that makes all new nodes appear just before `node` in the graph. Inside:

1. **Skip placeholders/outputs**: Subgraph placeholders correspond to our `available_inputs` (already in the main graph). The output node is just `return (value,)`.

2. **`node_copy(n, arg_transform)`**: Creates a copy of subgraph node `n` in the main graph. The `arg_transform` callback maps each argument (an `fx.Node` reference in the subgraph) to the corresponding node in the main graph. We use `name_to_node[arg.name]` for this mapping.

3. **`name_to_node[n.name] = new_node`**: After copying, we update the mapping. This is essential for two reasons:
   - Within the same subgraph, later nodes may reference earlier copied nodes
   - Future activation recomputations may need nodes inserted by this iteration

#### Use replacement (lines 208–212)

```python
if n.name == act_name:
    old_node = name_to_node[act_name]
    replace_subsequent_uses_of(gm.graph, old_node=old_node, new_node=new_node)
```

When we've copied the final node of the subgraph (the activation we're recomputing), we call `replace_subsequent_uses_of` to redirect all backward uses from the original activation to this new recomputed version.

#### Finalization (lines 218–220)

```python
gm.graph.lint()
gm.recompile()
```

`lint()` validates graph integrity — checks that all node references are valid, no dangling edges, topological order is consistent. `recompile()` regenerates the Python `forward()` function from the modified graph, making the `GraphModule` executable.

### 6.4 `verify_correctness()` (lines 230–267)

Runs both graphs with the same inputs, flattens all output tensors (the training graph returns nested tuples of parameters, optimizer states, etc.), and compares element-by-element with `torch.allclose(atol=1e-5, rtol=1e-4)`.

**Why `_flatten_tensors`?** The graph output is `(None, [param_tensors...], [opt_state_tensors...])` — a nested structure. Simple `zip` on the top level would compare `None` with `None` and `list` with `list`, missing the actual tensor comparisons. Recursive flattening extracts all tensors regardless of nesting depth.

### 6.5 Phase 3 in `run_experiments.py`

`profile()` now returns `(profiler, gm, flat_args)` — the GraphModule and its flattened inputs are needed for Phase 3 rewriting. `run_phase3()`:

1. **Deep-copies the GraphModule** (`copy.deepcopy(gm)`) to avoid corrupting the profiler's graph.
2. Calls `apply_ac_to_graph()` on the copy.
3. Runs `verify_correctness()` against the original.
4. Measures latency: 2 warmup + 5 measured iterations with CUDA events, both original and modified.

### 6.6 Updated End-to-End Flow

```
run_experiments.py main()
├── run_phase1()
│   └── profile() → (profiler, gm, flat_args)   [NEW: returns gm + args]
├── run_phase2(results_no_ac, profilers)
│   └── mu_two_algorithm() → ACDecision
├── run_phase3(profilers, graph_data, decisions)  [NEW]
│   ├── copy.deepcopy(gm)
│   ├── apply_ac_to_graph(modified_gm, profiler, decision)
│   │   ├── find_recomp_inputs() per activation
│   │   ├── _extract_graph_with_inputs_outputs()
│   │   ├── gm.graph.node_copy() + inserting_before()
│   │   └── replace_subsequent_uses_of()
│   ├── verify_correctness(gm, modified_gm, args)
│   └── latency measurement (CUDA events)
└── print_summary()
```
