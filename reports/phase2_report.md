# Phase 2: μ-TWO Activation Checkpointing Algorithm
CS265, Spring 2025
Zilin Dai

## 1. Introduction

In Phase 1 we built a computational graph profiler that instruments a full training iteration and collects per-operator timing, memory footprint, tensor type classification, and activation lifetime data. The profiler showed that activations account for 46% of the forward+backward peak memory in ResNet-152 and 36% in BERT, and that earlier-layer activations sit idle in memory for thousands of execution steps before being consumed by backward.

In Phase 2 we implement the activation checkpointing algorithm from the μ-TWO paper (Purandare et al., MLSys 2023). The algorithm takes the profiler's output and decides which activations to *retain* (checkpoint) and which to *discard and recompute* during the backward pass. The goal is to reduce peak memory while minimizing the extra recomputation cost. We run the algorithm on ResNet-152 and BERT at batch sizes 2, 4, 8, and 16, and analyze the compute-memory tradeoff.

## 2. Problems Tackled

- **[Recomputation Source Tracking]** For each activation, we need to know which other activations are required as inputs to recompute it. When an activation is evicted, any dependent activation's recomputation cost increases because it must now also recompute the evicted activation first. This cascading effect must be tracked correctly.

- **[Greedy Selection with the Recompute Ratio]** The μ-TWO algorithm uses a greedy heuristic: at each step, evict the activation with the highest `recompute_ratio = memory_size / recomp_time`. We need to correctly maintain this ratio as side-effects propagate through the dependency graph.

- **[Memory Simulation with Checkpointing]** To verify that the algorithm's decisions actually reduce peak memory, we need a memory simulator that models the altered lifetime of recomputed activations — they are freed after their last forward use instead of persisting until their backward consumer.

- **[Performance Optimization]** A naive implementation that re-simulates the entire graph at every greedy step is O(num_activations × num_nodes), which is prohibitively slow for large models (777 activations × 8679 nodes for ResNet-152). We need a fast incremental approach.

## 3. Technical Description

### 3.1 Algorithm Overview

The μ-TWO paper proposes a scheduler that decides, for each intermediate tensor, whether to swap it to CPU, recompute it, or keep it in GPU memory. In our single-model, single-GPU setting (no CPU swapping), the algorithm simplifies to a pure recomputation strategy:

**Input:**
- Per-activation data from the profiler: memory size (bytes), operator runtime (ms), creation index, last forward use index, first backward use index
- A memory budget (target peak memory after checkpointing)

**Output:**
- A partition of all activations into "retain" and "recompute" sets

**Key metric — Recompute Ratio:**
```
recompute_ratio = memory_size / total_recomp_time
```
A high ratio means: large memory savings, cheap to recompute — an ideal eviction candidate. The algorithm greedily picks the activation with the highest ratio at each step.

### 3.2 Algorithm Pseudocode

```
Algorithm: μ-TWO Greedy Recomputation (single-model)
────────────────────────────────────────────────────
Input: activations (from profiler), memory_budget
Output: retain_set, recompute_set

1. For each activation a:
     a.recomp_time   = a.op_runtime        # time of the single op
     a.recomp_srcs   = {other activations that are direct inputs to a}
     a.recompute_ratio = a.memory_size / a.recomp_time

2. Precompute peak_alive_acts = set of activations alive at the
   fwd+bwd peak node (from profiler's memory timeline)

3. estimated_peak = baseline_fwd_bwd_peak

4. candidates = all activations
   recomputed = {}

5. While estimated_peak > memory_budget AND candidates is not empty:
     a. best = argmax(candidates, key=recompute_ratio)

     b. recomputed.add(best)
        candidates.remove(best)

     c. If best in peak_alive_acts:    # O(1) fast check
            estimated_peak -= best.memory_size

     d. For each remaining candidate c:
            If best in c.recomp_srcs:
                c.recomp_srcs.remove(best)
                c.recomp_srcs.add(best.recomp_srcs)  # inherit sources
                c.recomp_time += best.recomp_time      # cost increases
                c.recompute_ratio = c.memory_size / c.recomp_time

6. Run full memory simulation with recomputed set to get accurate
   projected peak.

7. Return retain = candidates, recompute = recomputed
```

### 3.3 Recomputation Source Tracking

Each activation node in the FX graph has input edges to other nodes. We classify these inputs:
- **Parameters/Placeholders**: Always available in memory — free to use, never in `recomp_srcs`.
- **Other activations**: If they are retained, they are available. If they are evicted, they must be recomputed first.

When activation A is evicted:
- Any activation B that had A in its `recomp_srcs` must now first recompute A before it can recompute itself.
- B's `recomp_time` increases by A's `recomp_time`.
- B's `recomp_srcs` replaces A with A's own sources (transitive closure).
- B's `recompute_ratio` decreases, making it a less attractive eviction candidate.

This cascading effect is critical for correctness: without it, the algorithm would underestimate recomputation costs and make overly aggressive eviction decisions.

### 3.4 Fast Peak Estimation

The naive approach re-simulates the entire execution graph (8679 nodes for ResNet-152) at every greedy step. With up to 777 activation candidates, this creates 6.7 million node visits — which took over 20 minutes in practice.

Our optimization: **precompute once** which activations are alive at the fwd+bwd peak node, then do an O(1) subtraction when an activation from that set is evicted. This reduces the inner loop to simple set lookups and arithmetic.

We only run the full simulation once at the end to get the accurate projected peak and memory timeline for visualization. This brought runtime from 20+ minutes down to under 1 second.

### 3.5 Memory Simulation with Checkpointing

The memory simulator from Phase 1 tracks an "alive" set of tensors. Each tensor is freed when its last consumer executes. For checkpointed (retained) activations, this is unchanged.

For recomputed activations, we modify the `last_user` mapping: instead of being freed after their last backward consumer, they are freed after their **last forward use**. During the backward pass, they will be recomputed just-in-time from their sources (which is the job of Phase 3's graph rewriter).

```python
for act_node in activation_nodes:
    if act_node.name in recomputed_names:
        # Free after last forward use, not last backward use
        adjusted_last_user[act_node] = nodes[act_info[act_node]["last_fwd_use"]]
```

## 4. Experimental Results

### 4.1 Memory Savings Summary

We run the algorithm with a 75% budget (target: reduce peak memory to 75% of baseline) across all model and batch-size configurations:

| Model | BS | Baseline Peak | AC Peak | Saved | Saved % | # Recomputed | # Retained |
|-------|---:|-------------:|--------:|------:|--------:|-------------:|-----------:|
| ResNet-152 | 2 | 1,089 MB | 943 MB | 146 MB | 13.4% | 179 | 598 |
| ResNet-152 | 4 | 1,423 MB | 1,071 MB | 352 MB | 24.7% | 74 | 703 |
| ResNet-152 | 8 | 2,092 MB | 1,577 MB | 515 MB | 24.6% | 48 | 729 |
| ResNet-152 | 16 | 3,429 MB | 2,596 MB | 833 MB | 24.3% | 55 | 722 |
| BERT | 2 | 1,770 MB | 1,665 MB | 106 MB | 6.0% | 161 | 194 |
| BERT | 4 | 1,967 MB | 1,664 MB | 304 MB | 15.4% | 117 | 238 |
| BERT | 8 | 2,361 MB | 1,793 MB | 568 MB | 24.0% | 81 | 274 |
| BERT | 16 | 3,156 MB | 2,394 MB | 762 MB | 24.1% | 58 | 297 |

### 4.2 Peak Memory vs Batch Size (with and without AC)

![Peak Memory vs Batch Size](../outputs/plots/peak_memory_vs_batch_size_ac.png)

Both models show consistent memory savings from activation checkpointing, with the gap widening at larger batch sizes. At bs=16, ResNet-152 saves 833 MB and BERT saves 762 MB. This is because larger batch sizes produce larger activations (feature maps scale linearly with batch size), giving the algorithm more memory to reclaim.

At small batch sizes (especially BERT bs=2), the savings are modest (6%) because the peak is dominated by fixed costs — parameters (414 MB) and optimizer states (829 MB) — which activation checkpointing cannot reduce.

### 4.3 Memory Timeline: Baseline vs Activation Checkpointing

**ResNet-152 (bs=4):**

![ResNet-152 Memory Comparison](../outputs/plots/memory_comparison_Resnet152_bs4.png)

**BERT (bs=4):**

![BERT Memory Comparison](../outputs/plots/memory_comparison_BERT_bs4.png)

These plots overlay the baseline memory curve (dashed gray) with the AC memory curve (solid orange). The green hatched region represents memory saved by activation checkpointing. The savings are concentrated in the forward-to-backward transition zone — exactly where activations sit idle, waiting for the backward pass to consume them.

In ResNet-152, the savings are visible as a clear gap starting mid-forward and persisting through backward. In BERT, the savings are spread more evenly across the forward pass because BERT's transformer layers produce uniformly-sized activations, whereas ResNet's activations decrease in size through the network.

### 4.4 Memory Timeline (Stacked Area, by Tensor Type)

**ResNet-152 (bs=4):**

![ResNet-152 Memory Timeline](../outputs/plots/memory_timeline_Resnet152_bs4.png)

**BERT (bs=4):**

![BERT Memory Timeline](../outputs/plots/memory_timeline_BERT_bs4.png)

These stacked area charts show memory composition over the full training iteration. Key observations:
- **Forward pass**: Memory ramps up as activations (orange) accumulate. Parameters (blue) are loaded at the start and persist.
- **Backward pass**: Activations are freed as gradients (green) are computed. Memory transitions from activation-dominated to gradient-dominated.
- **Optimizer step**: A spike occurs at the `_fused_adam` node, which temporarily holds both old and updated parameters/optimizer states.

### 4.5 Greedy Algorithm Convergence

**ResNet-152 (bs=4):**

![Greedy Convergence ResNet-152](../outputs/plots/greedy_convergence_Resnet152_bs4.png)

**BERT (bs=4):**

![Greedy Convergence BERT](../outputs/plots/greedy_convergence_BERT_bs4.png)

These plots show how peak memory decreases as the greedy algorithm evicts activations one by one. The curve is steepest in the first few steps — the algorithm quickly finds the highest-ratio activations (large memory, cheap to recompute). Later steps yield diminishing returns as the remaining candidates have smaller sizes or higher recomputation costs.

For ResNet-152, 74 steps are needed to reach the 75% budget (1,067 MB). The first 10 steps alone save ~83 MB because they target large `relu_` and `convolution` outputs.

For BERT, 117 steps are needed. The first 15 steps evict `t_*` (transpose) operations, each saving 9 MB at near-zero recomputation cost (~0.01 ms each).

### 4.6 Pareto Frontier: Memory vs Recomputation Overhead

**ResNet-152 (bs=4):**

![Pareto Frontier ResNet-152](../outputs/plots/pareto_frontier_Resnet152_bs4.png)

**BERT (bs=4):**

![Pareto Frontier BERT](../outputs/plots/pareto_frontier_BERT_bs4.png)

These plots sweep the budget from 100% (no AC) to 50% (aggressive AC) and show the tradeoff. Key takeaways:
- **ResNet-152**: Reducing peak from 1,423 MB to 1,071 MB (75%) costs only ~3.5 ms of extra recomputation. Going below 960 MB (60%) requires ~9 ms, beyond which the curve flattens — further savings come at rapidly increasing cost.
- **BERT**: The curve is steeper because BERT's activations have more uniform costs. Reducing to 75% costs about 2 ms. Below 60%, the overhead grows sharply.

The "knee" of each curve represents the practical sweet spot for activation checkpointing — maximum memory savings before recomputation overhead becomes significant.

### 4.7 Activation Scatter: Retain vs Recompute

**ResNet-152 (bs=4):**

![Activation Scatter ResNet-152](../outputs/plots/activation_scatter_Resnet152_bs4.png)

**BERT (bs=4):**

![Activation Scatter BERT](../outputs/plots/activation_scatter_BERT_bs4.png)

Each dot is one activation. Red = recomputed, blue = retained. The x-axis is recomputation time (ms) and the y-axis is memory size (MB). Diagonal dashed lines show iso-ratio contours.

Recomputed activations cluster in the upper-left region (high memory, low recomputation cost) — exactly the candidates that the `recompute_ratio` heuristic favors. Retained activations are either small (not worth evicting) or expensive to recompute (lower ratio).

### 4.8 What Gets Recomputed

**ResNet-152**: Primarily `relu_` activations (in-place ReLU outputs) and early-layer `convolution` outputs. ReLU is near-instantaneous (~0.02 ms) but produces feature maps of 3–12 MB. The first few convolution outputs are large (12 MB) but fast enough to recompute. Deeper convolutions are retained because their outputs are smaller (3 MB) and the recomputation cost is higher.

**BERT**: Primarily `t_*` (transpose) and `view_*` (reshape) operations. These are essentially free to recompute (~0.01 ms) because they only change the tensor's metadata, not its data. Each saves 6–9 MB. The algorithm also evicts some `expand_*` and `getitem_*` operations from the attention computation. The attention weights (`_softmax`) and layer-norm outputs are retained because they are expensive to recompute from scratch.

## 5. Challenges

- **Performance of the greedy loop.** The initial implementation re-ran the full memory simulation (walking all ~8,000 nodes) at every greedy step. For ResNet-152 with 777 activation candidates, this created 6.7M node visits and took 20+ minutes. The fix was to precompute which activations are alive at the peak node and use O(1) incremental peak estimation, reducing runtime to under 1 second. The full simulation is only run once at the end for the accurate projected peak.

- **Fixed-cost floor limits savings at small batch sizes.** At bs=2, parameters + optimizer states already consume ~690 MB for ResNet-152 and ~1,243 MB for BERT. These are fixed costs that activation checkpointing cannot touch. So even with aggressive eviction, the achievable savings are limited to the activation fraction. BERT at bs=2 can only save 6% because activations are only a small fraction of the peak at that batch size.

- **Side-effect propagation correctness.** When activation A is evicted and activation B depends on A, B's recomputation must first recreate A. If we forget to add A's cost to B's `recomp_time`, the algorithm overestimates B's ratio and may make a bad decision. We handle this by transitively propagating both costs and sources: B inherits A's sources and adds A's recomp_time. This ensures that the ratio stays accurate as the algorithm progresses.

- **Connecting to Phase 3.** The algorithm outputs a set of activation names to recompute. Phase 3 must: (1) for each recomputed activation, extract the forward subgraph that produces it from checkpointed inputs, (2) insert that subgraph into the backward pass just before the activation's first backward consumer, and (3) redirect subsequent backward uses from the original activation node to the recomputed one. The `nodes_required_to_recompute` must all be either placeholder nodes or retained (checkpointed) activations — the algorithm guarantees this through its source propagation invariant.

## 6. Implementation Details

### File Structure

| File | Description |
|------|-------------|
| `ac_algorithm.py` | μ-TWO algorithm: `ActivationInfo` and `ACDecision` dataclasses, `_build_activation_infos()`, `_precompute_peak_contributors()`, `simulate_peak_memory()`, `mu_two_algorithm()`, `budget_sweep()`, `format_ac_decision()` |
| `ac_visualize.py` | All visualizations: `plot_memory_timeline()`, `plot_memory_comparison()`, `plot_greedy_convergence()`, `plot_pareto_frontier()`, `plot_peak_memory_with_ac()`, `plot_recompute_ratio_distribution()` |
| `run_experiments.py` | Experiment runner updated with `run_phase1()` and `run_phase2()` |

### Key Data Structures

```python
@dataclass
class ActivationInfo:
    node: fx.Node          # FX graph node reference
    name: str              # node name (for cross-referencing)
    memory_size: int       # bytes — from profiler._node_output_mem
    op_runtime: float      # ms — from profiler.avg_runtimes
    created_at: int        # node index in topological order
    last_fwd_use: int      # last forward consumer index
    first_bwd_use: int     # first backward consumer index
    recomp_time: float     # total recomp time (increases with evictions)
    recomp_srcs: Set[str]  # activation inputs (shrinks as sources are evicted)
    recompute_ratio: float # memory_size / recomp_time
    recompute: bool        # final decision: True = discard and recompute
```

### How to Reproduce

```bash
conda activate cs265
python run_experiments.py
```

This runs both Phase 1 (profiling) and Phase 2 (algorithm) and generates all output files.

### Output Files

All generated outputs are under `outputs/`, organized by type:

| Directory | File Pattern | Count | Description |
|-----------|-------------|------:|-------------|
| `outputs/profiling_stats/` | `profiling_stats_{Model}_bs{N}.txt` | 8 | Per-node profiling data |
| `outputs/ac_decisions/` | `ac_decision_{Model}_bs{N}.txt` | 8 | Algorithm decisions with greedy log |
| `outputs/comp_graphs/` | `comp_graph_{Model}_bs{N}.txt` | 8 | Full FX graph dumps |
| `outputs/plots/` | `memory_timeline_*.png` | 2 | Stacked area memory timeline |
| `outputs/plots/` | `memory_comparison_*.png` | 2 | Baseline vs AC overlay |
| `outputs/plots/` | `greedy_convergence_*.png` | 2 | Peak memory vs greedy step |
| `outputs/plots/` | `pareto_frontier_*.png` | 2 | Memory vs recompute tradeoff |
| `outputs/plots/` | `activation_scatter_*.png` | 2 | Retain vs recompute scatter |
| `outputs/plots/` | `peak_memory_vs_batch_size.png` | 1 | Baseline peak vs batch size |
| `outputs/plots/` | `peak_memory_vs_batch_size_ac.png` | 1 | With/without AC comparison |
