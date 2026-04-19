"""
Phase 2: μ-TWO Activation Checkpointing Algorithm (single-model case).

Reference: "μ-TWO: 3x Faster Multi-Model Training with Orchestration and
Memory Optimization" (Purandare et al., MLSys 2023).

In the single-model setting (no CPU swapping), the algorithm reduces to a
greedy loop that picks the activation with the highest recompute_ratio
(= memory_size / total_recomp_time) and marks it for recomputation, until
peak memory fits within the budget.

Inputs from the profiler (Phase 1):
  - activation list with sizes, runtimes, and lifetime info
  - memory timeline for peak memory simulation

Output:
  - partition of activations into "retain" (checkpoint) vs "recompute" sets
  - projected peak memory after applying the decisions
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import torch.fx as fx

from graph_prof import GraphProfiler, NodeType


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivationInfo:
    """All profiler-derived info needed by the μ-TWO algorithm for one activation."""
    node: fx.Node
    name: str
    memory_size: int            # bytes
    op_runtime: float           # ms — time of the single op that produces this activation
    created_at: int             # node index in topological order
    last_fwd_use: int
    first_bwd_use: Optional[int]
    last_bwd_use: Optional[int]

    # Recomputation bookkeeping (updated as algorithm runs)
    recomp_time: float = 0.0    # ms — total time to recompute from current sources
    recomp_srcs: Set[str] = field(default_factory=set)  # names of activation inputs
    recompute_ratio: float = 0.0  # memory_size / recomp_time  (higher = better to evict)

    # Algorithm output
    recompute: bool = False     # True => discard in forward, recompute before backward use


@dataclass
class ACDecision:
    """Result of running the μ-TWO algorithm."""
    retain: List[str]           # activation names to keep (checkpoint)
    recompute: List[str]        # activation names to discard and recompute
    baseline_peak: float        # original peak memory (bytes)
    projected_peak: float       # projected peak after AC (bytes)
    memory_saved: float         # bytes saved
    iteration_log: List[Dict]   # per-iteration log for visualization


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute recomputation sources for each activation
# ─────────────────────────────────────────────────────────────────────────────

def _build_activation_infos(profiler: GraphProfiler) -> Dict[str, ActivationInfo]:
    """
    Extract per-activation data from the profiler's static analysis and
    runtime measurements.  For each activation, 'recomp_srcs' is the set of
    *other activations* that are direct inputs to the op producing it.
    Placeholders and parameters are always available, so they don't appear
    in recomp_srcs.
    """
    act_set = set(n.name for n in profiler.activation_nodes)
    infos: Dict[str, ActivationInfo] = {}

    for act_node in profiler.activation_nodes:
        info = profiler.act_info[act_node]
        mem = profiler._node_output_mem.get(act_node.name, 0)
        runtime = profiler.avg_runtimes.get(act_node.name, 0.0)

        # Direct activation inputs: other activations consumed by this op.
        # Parameters and placeholders are always in memory, so they are free.
        srcs = set()
        for inp in act_node.all_input_nodes:
            if inp.name in act_set:
                srcs.add(inp.name)

        ai = ActivationInfo(
            node=act_node,
            name=act_node.name,
            memory_size=mem,
            op_runtime=runtime,
            created_at=info["created_at"],
            last_fwd_use=info["last_fwd_use"],
            first_bwd_use=info["first_bwd_use"],
            last_bwd_use=info["last_bwd_use"],
            recomp_time=runtime,
            recomp_srcs=srcs,
        )
        # Initial ratio: avoid division by zero for near-instant ops
        ai.recompute_ratio = mem / max(ai.recomp_time, 1e-6)
        infos[act_node.name] = ai

    return infos


# ─────────────────────────────────────────────────────────────────────────────
# Fast peak estimation: precompute which activations overlap the peak node
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_peak_contributors(profiler: GraphProfiler) -> Tuple[float, Set[str]]:
    """
    Find the fwd+bwd peak node from the baseline memory timeline. Return the
    peak memory value and the set of activation names that are alive at that
    point.  Evicting any of those activations reduces peak memory by their size.

    This avoids re-simulating the whole graph at every greedy step.
    """
    tl = profiler.memory_timeline
    if not tl:
        return profiler.avg_fwdbwd_peak, set()

    name_to_node = {n.name: n for n in profiler.nodes_list}

    # Find the fwd+bwd peak entry
    fwdbwd_entries = [
        e for e in tl
        if profiler.node_region.get(name_to_node.get(e["node_name"])) in ("forward", "loss", "backward")
    ]
    if not fwdbwd_entries:
        return profiler.avg_fwdbwd_peak, set()

    peak_entry = max(fwdbwd_entries, key=lambda e: e["total_memory"])
    peak_idx = next(i for i, e in enumerate(tl) if e is peak_entry)

    # Walk the simulation to find which activations are alive at peak_idx
    act_names = set(n.name for n in profiler.activation_nodes)
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

    # Activations that are alive at the peak
    alive_acts = {n.name for n in alive if n.name in act_names}
    return peak_entry["total_memory"], alive_acts


# ─────────────────────────────────────────────────────────────────────────────
# Memory simulator (full simulation — used once at the end for visualization)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_peak_memory(
    profiler: GraphProfiler,
    recomputed_names: Set[str],
) -> Tuple[float, List[Dict]]:
    """
    Re-run the memory simulation, but free recomputed activations immediately
    after their last forward use.  Returns (fwd+bwd peak_bytes, timeline).
    """
    nodes = profiler.nodes_list
    act_info = profiler.act_info
    node_region = profiler.node_region
    node_type_map = profiler.node_type_map

    # Build adjusted last_user: recomputed activations freed after last fwd use
    adjusted_last_user: Dict[fx.Node, fx.Node] = dict(profiler.last_user)
    for act_node in profiler.activation_nodes:
        if act_node.name in recomputed_names:
            info = act_info[act_node]
            adjusted_last_user[act_node] = nodes[info["last_fwd_use"]]

    # Walk the graph and simulate alive-set evolution
    alive: Dict[fx.Node, int] = {}
    timeline: List[Dict] = []
    peak = 0.0

    for node in nodes:
        mem = profiler._node_output_mem.get(node.name, 0)
        if mem > 0 and node.op != "output":
            alive[node] = mem

        for inp in node.all_input_nodes:
            if adjusted_last_user.get(inp) == node:
                alive.pop(inp, None)

        total = sum(alive.values())
        bd = {nt: 0 for nt in NodeType}
        for alive_n, alive_mem in alive.items():
            bd[node_type_map.get(alive_n, NodeType.OTHER)] += alive_mem

        region = node_region.get(node, "optimizer")
        timeline.append({
            "node_name": node.name,
            "total_memory": total,
            "breakdown": bd,
            "region": region,
        })
        if region in ("forward", "loss", "backward"):
            peak = max(peak, total)

    return peak, timeline


# ─────────────────────────────────────────────────────────────────────────────
# μ-TWO greedy algorithm (single-model recomputation only)
# ─────────────────────────────────────────────────────────────────────────────

def mu_two_algorithm(
    profiler: GraphProfiler,
    memory_budget: Optional[float] = None,
) -> ACDecision:
    """
    Greedy activation checkpointing using the μ-TWO recompute heuristic.

    Algorithm overview (from the paper, simplified for single-model):
    ─────────────────────────────────────────────────────────────────
    1. Build candidate set = all activations with their profiler stats.
    2. Precompute which activations are alive at the fwd+bwd peak node.
    3. While estimated peak > budget and candidates remain:
       a. Pick candidate with highest recompute_ratio = mem_size / recomp_time.
          (This maximises memory freed per unit of added compute.)
       b. Mark it for recomputation; remove from candidate set.
       c. If the evicted activation was alive at the peak, subtract its memory
          from the estimated peak. (Fast O(1) update instead of full simulation.)
       d. Propagate side-effects: any remaining candidate whose recomp_srcs
          included the evicted activation must now also recompute that
          activation first, so its recomp_time increases and ratio decreases.
    4. Run one full simulation at the end to get the actual projected peak.
    5. Return the retain / recompute partition.

    Args:
        profiler:  A fully-aggregated GraphProfiler from Phase 1.
        memory_budget:  Target peak memory in bytes.  If None, defaults to
                        baseline minus 50% of activation memory.
    """
    baseline_peak = profiler.avg_fwdbwd_peak

    if memory_budget is None:
        act_mem = sum(profiler._node_output_mem.get(n.name, 0)
                      for n in profiler.activation_nodes)
        memory_budget = baseline_peak - act_mem * 0.5

    # Build per-activation info from profiler
    infos = _build_activation_infos(profiler)

    # Precompute which activations are alive at the peak node (fast eviction check)
    peak_mem, peak_alive_acts = _precompute_peak_contributors(profiler)
    estimated_peak = peak_mem

    recomputed: Set[str] = set()
    candidates: Set[str] = set(infos.keys())

    iteration_log: List[Dict] = []
    step = 0

    while candidates and estimated_peak > memory_budget:
        # ── Pick candidate with highest recompute_ratio ──
        best_name = max(candidates, key=lambda n: infos[n].recompute_ratio)
        best = infos[best_name]

        # Skip zero-size activations (nothing to save)
        if best.memory_size == 0:
            candidates.discard(best_name)
            continue

        # ── Mark for recomputation ──
        recomputed.add(best_name)
        candidates.discard(best_name)

        # ── Fast peak update: subtract memory if this activation was at peak ──
        if best_name in peak_alive_acts:
            estimated_peak -= best.memory_size

        # ── Propagate side-effects to remaining candidates ──
        for cand_name in list(candidates):
            cand = infos[cand_name]
            if best_name in cand.recomp_srcs:
                cand.recomp_srcs.discard(best_name)
                cand.recomp_srcs.update(best.recomp_srcs)
                cand.recomp_time += best.recomp_time
                cand.recompute_ratio = cand.memory_size / max(cand.recomp_time, 1e-6)

        step += 1
        iteration_log.append({
            "step": step,
            "evicted": best_name,
            "evicted_mem_mb": best.memory_size / (1024**2),
            "evicted_recomp_ms": best.recomp_time,
            "evicted_ratio": best.recompute_ratio,
            "num_recomputed": len(recomputed),
            "projected_peak_mb": estimated_peak / (1024**2),
        })

    # ── Mark final decisions ──
    for name in recomputed:
        infos[name].recompute = True

    retain_list = [n for n in infos if n not in recomputed]
    recompute_list = list(recomputed)

    # Final accurate peak via full simulation
    projected_peak, _ = simulate_peak_memory(profiler, recomputed)

    # Update the last log entry with the accurate peak
    if iteration_log:
        iteration_log[-1]["projected_peak_mb"] = projected_peak / (1024**2)

    return ACDecision(
        retain=retain_list,
        recompute=recompute_list,
        baseline_peak=baseline_peak,
        projected_peak=projected_peak,
        memory_saved=baseline_peak - projected_peak,
        iteration_log=iteration_log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Budget sweep: run algorithm at multiple budget levels
# ─────────────────────────────────────────────────────────────────────────────

def budget_sweep(
    profiler: GraphProfiler,
    budget_fractions: List[float] = None,
) -> List[Tuple[float, ACDecision]]:
    """
    Run the μ-TWO algorithm at several memory budget levels (as fractions of
    the baseline peak).  Returns list of (budget_fraction, decision) pairs.
    """
    if budget_fractions is None:
        budget_fractions = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

    baseline = profiler.avg_fwdbwd_peak
    results = []
    for frac in budget_fractions:
        budget = baseline * frac
        decision = mu_two_algorithm(profiler, memory_budget=budget)
        results.append((frac, decision))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print results
# ─────────────────────────────────────────────────────────────────────────────

def format_ac_decision(decision: ACDecision, model_name: str = "") -> str:
    """Format the AC decision into a human-readable report."""
    MB = 1024 ** 2
    lines = []
    w = lines.append

    w("=" * 90)
    w(f"μ-TWO ACTIVATION CHECKPOINTING DECISION{f'  ({model_name})' if model_name else ''}")
    w("=" * 90)

    w(f"\n  Baseline fwd+bwd peak:    {decision.baseline_peak/MB:>10.1f} MB")
    w(f"  Projected peak (with AC): {decision.projected_peak/MB:>10.1f} MB")
    w(f"  Memory saved:             {decision.memory_saved/MB:>10.1f} MB "
      f"({decision.memory_saved/decision.baseline_peak*100:.1f}%)")
    w(f"  Activations retained:     {len(decision.retain):>10d}")
    w(f"  Activations recomputed:   {len(decision.recompute):>10d}")

    if decision.iteration_log:
        w(f"\n--- Greedy Iteration Log ({len(decision.iteration_log)} steps) ---")
        w(f"{'Step':>4} {'Evicted':<35} {'Mem(MB)':>8} {'Recomp(ms)':>10} "
          f"{'Ratio':>12} {'Peak(MB)':>10}")
        w("-" * 90)
        for entry in decision.iteration_log:
            w(f"{entry['step']:>4} {entry['evicted']:<35} "
              f"{entry['evicted_mem_mb']:>8.3f} {entry['evicted_recomp_ms']:>10.4f} "
              f"{entry['evicted_ratio']:>12.1f} {entry['projected_peak_mb']:>10.1f}")

    w(f"\n--- Retained activations ({len(decision.retain)}) ---")
    for name in sorted(decision.retain)[:20]:
        w(f"  [RETAIN]    {name}")
    if len(decision.retain) > 20:
        w(f"  ... and {len(decision.retain) - 20} more")

    w(f"\n--- Recomputed activations ({len(decision.recompute)}) ---")
    for name in sorted(decision.recompute)[:20]:
        w(f"  [RECOMPUTE] {name}")
    if len(decision.recompute) > 20:
        w(f"  ... and {len(decision.recompute) - 20} more")

    w("=" * 90)
    return "\n".join(lines)
