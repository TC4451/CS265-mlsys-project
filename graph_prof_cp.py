import operator
from enum import Enum
from typing import Dict, List, Any, Optional
import torch
import torch.fx as fx


class OP(str, Enum):
    # Node in FX graph's operation kinds in the traced graph.
    CALL_FUNCTION = "call_function"   # Call a standalone function.
    CALL_MODULE = "call_module"       # Call a child module/layer.
    CALL_METHOD = "call_method"       # Call a method on an object.
    GET_ATTR = "get_attr"             # Read a stored attribute.
    OUTPUT = "output"                 # Return value of the graph.
    PLACEHOLDER = "placeholder"       # Function input.

class NodeType(Enum):
    """
    NodeType is an enum that records the type of the tensors in the graph.
    Categories from the project spec: parameter, gradient, activation,
    optimizer state, or other.
    """
    PARAM = 0
    ACT = 1
    GRAD = 2
    OPT_STATE = 3
    OTHER = 4


class GraphProfiler(fx.Interpreter):
    """
    Execute a traced graph node by node and collect timing and memory stats.

    Design choice:
    - Inherit from fx.Interpreter to hook both full-run and per-node execution.
    - Keep execution behavior faithful while adding instrumentation.
    """
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # =====================================================================
        # STATIC ANALYSIS
        # =====================================================================

        # Keep nodes in execution order and build index lookup used by analysis.
        self.nodes_list = list(self.module.graph.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}    #use for calculate idlea span

        # --- 1. Find boundary nodes ---
        # The boundary between the forward pass and backward pass can be
        # identified by locating the node '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will
        # define the end of the forward pass. You will see the loss function
        # after this operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass.

        self.sep_idx = None
        self.sep_backward_idx = None
        self.fused_adam_idx = None
        # Scan once to locate forward/backward split and optimizer start markers.
        for i, node in enumerate(self.nodes_list):
            if node.target == torch.ops.separator.sep.default:
                self.sep_idx = i
            elif node.target == torch.ops.separator.sep_backward.default:
                self.sep_backward_idx = i
            elif node.target == torch.ops.aten._fused_adam.default:
                self.fused_adam_idx = i

        # The parameters and gradients of the model can be obtained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =
        # call_function[target=torch.ops.aten._fused_adam.default]'.
        # The argument at position 0 is the list of parameter nodes, while the
        # argument at position 1 is the list of gradient nodes.

        # When _fused_adam is absent (foreach optimizer), find first _foreach_*
        # after sep_backward as the optimizer start.
        self.optimizer_start_idx = self.fused_adam_idx
        if self.optimizer_start_idx is None and self.sep_backward_idx is not None:
            for i in range(self.sep_backward_idx + 1, len(self.nodes_list)):
                if self.nodes_list[i].op == OP.CALL_FUNCTION and \
                   "_foreach_" in str(self.nodes_list[i].target):
                    self.optimizer_start_idx = i
                    break

        # --- 2. Classify each node into a region ---
        # Region labels are assigned by index boundaries once separators are found.
        self.node_region: Dict[fx.Node, str] = {}
        for i, node in enumerate(self.nodes_list):
            if self.sep_idx is not None and i <= self.sep_idx:
                self.node_region[node] = "forward"
            elif self.sep_backward_idx is not None and i < self.sep_backward_idx:
                self.node_region[node] = "loss"
            elif self.optimizer_start_idx is not None and i < self.optimizer_start_idx:
                self.node_region[node] = "backward"
            else:
                self.node_region[node] = "optimizer"

        # --- 3. Identify params, grads, optimizer states ---
        # %_fused_adam : [num_users=3] = call_function[target=torch.ops.aten._fused_adam.default](
        #     args = (
        #         [%p0, %p1, ...],          # parameters
        #         [%g0, %g1, ...],          # gradients
        #         [%m0, %m1, ...],          # exp_avgs
        #         [%v0, %v1, ...],          # exp_avg_sqs
        #         [%vmax0, %vmax1, ...],    # max_exp_avg_sqs (optional by mode)
        #         [%step0, %step1, ...],    # step counters
        #         ...
        #     ),
        #     kwargs = {}
        # )

        # %getitem = call_function[target=operator.getitem](args = (%_fused_adam, 0), kwargs = {})
        # %getitem_1 = call_function[target=operator.getitem](args = (%_fused_adam, 1), kwargs = {})
        # %getitem_2 = call_function[target=operator.getitem](args = (%_fused_adam, 2), kwargs = {})

        self.param_nodes: set = set()   #model weight
        self.grad_nodes: set = set()    #gradient of weight
        self.opt_state_nodes: set = set()       #optimizer state tensor
        if self.fused_adam_idx is not None:
            adam = self.nodes_list[self.fused_adam_idx]
            if len(adam.args) >= 2:
                if isinstance(adam.args[0], (list, tuple)):
                    self.param_nodes = set(adam.args[0])
                if isinstance(adam.args[1], (list, tuple)):
                    self.grad_nodes = set(adam.args[1])
            # args[2]=exp_avgs, args[3]=exp_avg_sqs, args[4]=max_exp_avg_sqs, args[5]=steps
            for idx in range(2, min(len(adam.args), 6)):
                if isinstance(adam.args[idx], (list, tuple)):
                    self.opt_state_nodes.update(adam.args[idx])

        # --- 4. Categorize every node as PARAM/GRAD/ACT/OPT_STATE/OTHER ---
        self.node_type_map: Dict[fx.Node, NodeType] = {}
        for node in self.nodes_list:
            if node in self.param_nodes:
                self.node_type_map[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_type_map[node] = NodeType.GRAD
            elif node in self.opt_state_nodes:
                self.node_type_map[node] = NodeType.OPT_STATE
            else:
                self.node_type_map[node] = NodeType.OTHER

        # Classify _fused_adam outputs: it returns (params, exp_avgs, exp_avg_sqs).
        # The node itself and its getitem children are otherwise left as OTHER.
        if self.fused_adam_idx is not None:
            adam = self.nodes_list[self.fused_adam_idx]
            # _fused_adam output index -> type:
            #   0 = updated params, 1 = updated exp_avgs, 2 = updated exp_avg_sqs
            adam_output_type = {0: NodeType.PARAM, 1: NodeType.OPT_STATE, 2: NodeType.OPT_STATE}
            self.node_type_map[adam] = NodeType.OPT_STATE  # mixed, but mostly opt state
            for user in adam.users:
                if user.op == OP.CALL_FUNCTION and \
                   user.target == operator.getitem and len(user.args) >= 2:
                    idx = user.args[1]
                    ntype = adam_output_type.get(idx, NodeType.OPT_STATE)
                    self.node_type_map[user] = ntype
                    # Second-level getitems extract individual tensors from the list
                    for sub_user in user.users:
                        if sub_user.op == OP.CALL_FUNCTION and \
                           sub_user.target == operator.getitem:
                            self.node_type_map[sub_user] = ntype

        # Activations: forward-pass function calls that are used in backward
        self.activation_nodes: List[fx.Node] = []
        for node in self.nodes_list:
            if self.node_region.get(node) == "forward" \
               and node.op in (OP.CALL_FUNCTION, OP.CALL_METHOD) \
               and node not in self.param_nodes \
               and any(self.node_region.get(u) == "backward" for u in node.users):
                self.node_type_map[node] = NodeType.ACT
                self.activation_nodes.append(node)

        # --- 5. Static data analysis on activations ---
        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.
        self.act_info: Dict[fx.Node, Dict] = {}
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
            self.act_info[act] = info

        # --- Precompute last user of each node (for memory simulation) ---
        # Memory model: free a tensor right after its final consumer executes.
        # Why this is needed:
        # During runtime in run_node, when current node equals last_user[input], profiler removes that input from alive memory.
        # That is how it simulates “free after final use” behavior.
        self.last_user: Dict[fx.Node, fx.Node] = {}
        for node in reversed(self.nodes_list):
            for inp in node.all_input_nodes:
                if inp not in self.last_user:
                    self.last_user[inp] = node

        # =====================================================================
        # RUNTIME STATS STORAGE
        # =====================================================================
        self._node_runtimes: Dict[str, List[float]] = {n.name: [] for n in self.nodes_list}
        self._node_output_mem: Dict[str, int] = {}
        self._peak_per_iter: List[float] = []
        self._peak_bd_per_iter: List[Dict] = []
        self._fwdbwd_peak_per_iter: List[float] = []
        self._fwdbwd_peak_bd_per_iter: List[Dict] = []
        self._timelines: List[List[Dict]] = []
        self._alive: Dict[fx.Node, int] = {}
        self._timeline: List[Dict] = []
        self._name_to_node: Dict[str, fx.Node] = {n.name: n for n in self.nodes_list}

        # Aggregated results
        self.avg_runtimes: Dict[str, float] = {}
        self.avg_peak_memory = 0.0
        self.peak_breakdown: Dict[NodeType, float] = {}
        self.avg_fwdbwd_peak = 0.0
        self.fwdbwd_peak_breakdown: Dict[NodeType, float] = {}
        self.memory_timeline: List[Dict] = []

    @staticmethod
    def _tensor_mem(val) -> int:
        # Recursively compute output memory in bytes for tensor-valued results.
        if isinstance(val, torch.Tensor):
            return val.element_size() * val.nelement()
        if isinstance(val, (tuple, list)):
            return sum(GraphProfiler._tensor_mem(v) for v in val)
        return 0

    def run(self, *args,
            initial_env: Dict[fx.Node, Any] | None = None,
            enable_io_processing: bool = True) -> Any:
        # Start a fresh memory timeline for this iteration.
        self._alive = {}
        self._timeline = []
        # Use the parent interpreter for actual graph execution.
        result = super().run(*args, initial_env=initial_env,
                             enable_io_processing=enable_io_processing)
        if self._timeline:
            # Overall peak includes forward, backward, and optimizer regions.
            peak = max(self._timeline, key=lambda e: e["total_memory"])
            self._peak_per_iter.append(peak["total_memory"])
            self._peak_bd_per_iter.append(peak["breakdown"])
            # Peak during forward+backward only (what AC targets)
            fwdbwd = [e for e in self._timeline
                      if self.node_region.get(self._name_to_node.get(e["node_name"]))
                      in ("forward", "loss", "backward")]
            if fwdbwd:
                fb_peak = max(fwdbwd, key=lambda e: e["total_memory"])
                self._fwdbwd_peak_per_iter.append(fb_peak["total_memory"])
                self._fwdbwd_peak_bd_per_iter.append(fb_peak["breakdown"])
        self._timelines.append(list(self._timeline))
        return result

    def run_node(self, n: fx.Node) -> Any:
        # Measure runtime using CUDA events
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        result = super().run_node(n)

        if use_cuda:
            end.record()
            torch.cuda.synchronize()        # make sure the program wait until work up to the endmark since operations are queued asynchronously
            self._node_runtimes[n.name].append(start.elapsed_time(end))

        # Track memory: record output size, update alive set
        mem = self._tensor_mem(result)
        self._node_output_mem[n.name] = mem
        if mem > 0 and n.op != OP.OUTPUT:
            self._alive[n] = mem
        for inp in n.all_input_nodes:
            # Remove tensors that will never be used again after this node.
            if self.last_user.get(inp) == n:
                self._alive.pop(inp, None)

        # Snapshot current memory with breakdown by type
        total = sum(self._alive.values())
        bd = {nt: 0 for nt in NodeType}
        for alive_n, alive_mem in self._alive.items():
            bd[self.node_type_map.get(alive_n, NodeType.OTHER)] += alive_mem
        self._timeline.append({"node_name": n.name, "total_memory": total,
                               "breakdown": bd})
        return result

    def aggregate_stats(self) -> None:
        # Convert per-iteration samples into final averages used by reports.
        for name, times in self._node_runtimes.items():
            self.avg_runtimes[name] = sum(times) / len(times) if times else 0.0
        if self._peak_per_iter:
            self.avg_peak_memory = sum(self._peak_per_iter) / len(self._peak_per_iter)
        if self._peak_bd_per_iter:
            self.peak_breakdown = {nt: 0.0 for nt in NodeType}
            for bd in self._peak_bd_per_iter:
                for nt, val in bd.items():
                    self.peak_breakdown[nt] += val
            n = len(self._peak_bd_per_iter)
            self.peak_breakdown = {nt: v / n for nt, v in self.peak_breakdown.items()}
        if self._fwdbwd_peak_per_iter:
            self.avg_fwdbwd_peak = sum(self._fwdbwd_peak_per_iter) / len(self._fwdbwd_peak_per_iter)
        if self._fwdbwd_peak_bd_per_iter:
            self.fwdbwd_peak_breakdown = {nt: 0.0 for nt in NodeType}
            for bd in self._fwdbwd_peak_bd_per_iter:
                for nt, val in bd.items():
                    self.fwdbwd_peak_breakdown[nt] += val
            n = len(self._fwdbwd_peak_bd_per_iter)
            self.fwdbwd_peak_breakdown = {nt: v / n for nt, v in self.fwdbwd_peak_breakdown.items()}
        if self._timelines:
            self.memory_timeline = self._timelines[-1]

    def reset_stats(self) -> None:
        # Keep static analysis; clear only runtime/memory sample history.
        self._node_runtimes = {n.name: [] for n in self.nodes_list}
        self._peak_per_iter = []
        self._peak_bd_per_iter = []
        self._fwdbwd_peak_per_iter = []
        self._fwdbwd_peak_bd_per_iter = []
        self._timelines = []

    def print_stats(self) -> None:
        print("\n" + self._format_stats())

    def save_stats(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self._format_stats() + "\n")
        print(f"Stats saved to {path}")

    def _format_stats(self) -> str:
        MB = 1024 * 1024
        lines = []
        w = lines.append

        w("=" * 90)
        w("GRAPH PROFILER STATISTICS")
        w("=" * 90)

        # --- 1. Computation time per region ---
        region_time = {"forward": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0}
        region_count = {"forward": 0, "loss": 0, "backward": 0, "optimizer": 0}
        for node in self.nodes_list:
            r = self.node_region.get(node, "other")
            if r in region_time:
                region_time[r] += self.avg_runtimes.get(node.name, 0.0)
                region_count[r] += 1
        total_time = sum(region_time.values())

        w(f"\n--- 1. Computation Time Summary ---")
        w(f"{'Region':<12} {'Nodes':>6} {'Time (ms)':>12} {'%':>8}")
        w("-" * 42)
        for r in ["forward", "loss", "backward", "optimizer"]:
            pct = region_time[r] / total_time * 100 if total_time > 0 else 0
            w(f"{r:<12} {region_count[r]:>6} {region_time[r]:>12.3f} {pct:>7.1f}%")
        w(f"{'TOTAL':<12} {sum(region_count.values()):>6} {total_time:>12.3f}")

        # --- 2. Per-node: type categorization + runtime + memory ---
        w(f"\n--- 2. Per-Node Profiling ({len(self.nodes_list)} nodes) ---")
        w(f"{'Name':<40} {'Type':<10} {'Region':<10} {'Time(ms)':>9} {'OutMem(MB)':>11}")
        w("-" * 84)
        for node in self.nodes_list:
            nt = self.node_type_map.get(node, NodeType.OTHER).name
            rg = self.node_region.get(node, "")
            rt = self.avg_runtimes.get(node.name, 0)
            mem = self._node_output_mem.get(node.name, 0) / MB
            w(f"{node.name:<40} {nt:<10} {rg:<10} {rt:>9.4f} {mem:>11.4f}")

        # --- 3. Static data analysis on activations ---
        w(f"\n--- 3. Activation Static Analysis ({len(self.activation_nodes)} activations) ---")
        w(f"  Activations are intermediate tensors created in the forward pass")
        w(f"  and consumed in the backward pass for gradient computation.")
        w(f"  'Idle span' = gap between last forward use and first backward use,")
        w(f"  during which the activation sits unused in memory.\n")
        w(f"{'Name':<40} {'Created':>7} {'LastFwd':>7} {'1stBwd':>7} {'LastBwd':>7} {'Idle':>5} {'MB':>9}")
        w("-" * 86)
        total_act = 0
        for act in self.activation_nodes:
            info = self.act_info[act]
            m = self._node_output_mem.get(act.name, 0)
            total_act += m
            idle = (info["first_bwd_use"] - info["last_fwd_use"]) \
                if info["first_bwd_use"] is not None else 0
            fb = str(info["first_bwd_use"]) if info["first_bwd_use"] is not None else "N/A"
            lb = str(info["last_bwd_use"]) if info["last_bwd_use"] is not None else "N/A"
            w(f"{act.name:<40} {info['created_at']:>7} {info['last_fwd_use']:>7} "
              f"{fb:>7} {lb:>7} {idle:>5} {m/MB:>9.4f}")
        w(f"\n  Total activation memory: {total_act/MB:.3f} MB")

        # --- 4. Peak memory breakdown ---
        pm = sum(self._node_output_mem.get(n.name, 0) for n in self.param_nodes)
        gm = sum(self._node_output_mem.get(n.name, 0) for n in self.grad_nodes)
        om = sum(self._node_output_mem.get(n.name, 0) for n in self.opt_state_nodes)

        w(f"\n--- 4. Peak Memory Breakdown ---")
        w(f"  Overall peak memory:     {self.avg_peak_memory/MB:.3f} MB")
        w(f"  Fwd+Bwd peak memory:     {self.avg_fwdbwd_peak/MB:.3f} MB")
        w(f"")
        w(f"  Breakdown at fwd+bwd peak (what activation checkpointing targets):")
        if self.fwdbwd_peak_breakdown:
            for nt in [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPT_STATE, NodeType.OTHER]:
                v = self.fwdbwd_peak_breakdown.get(nt, 0)
                pct = v / self.avg_fwdbwd_peak * 100 if self.avg_fwdbwd_peak > 0 else 0
                w(f"    {nt.name:<10}: {v/MB:>9.3f} MB  ({pct:>5.1f}%)")

        w(f"\n  Total tensor memory by type:")
        w(f"    PARAM    : {pm/MB:>9.3f} MB  ({len(self.param_nodes)} tensors)")
        w(f"    GRAD     : {gm/MB:>9.3f} MB  ({len(self.grad_nodes)} tensors)")
        w(f"    OPT_STATE: {om/MB:>9.3f} MB  ({len(self.opt_state_nodes)} tensors)")
        w(f"    ACT      : {total_act/MB:>9.3f} MB  ({len(self.activation_nodes)} tensors)")
        w("=" * 90)
        return "\n".join(lines)

    def get_peak_memory(self) -> float:
        return self.avg_peak_memory

    def get_fwdbwd_peak_memory(self) -> float:
        """Peak memory during forward+backward (excludes optimizer)."""
        return self.avg_fwdbwd_peak

    def get_peak_breakdown(self) -> Dict[NodeType, float]:
        return dict(self.peak_breakdown)
