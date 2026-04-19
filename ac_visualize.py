"""
Visualization module for Phase 1 and Phase 2 results.

Phase 1 plots (updated per TA feedback — line graphs instead of bar graphs):
  - Memory timeline: stacked area / line chart showing memory over execution
  - Peak memory vs batch size

Phase 2 plots:
  - Memory timeline comparison: baseline vs with activation checkpointing
  - Greedy algorithm convergence: peak memory vs iteration step
  - Pareto frontier: memory saved vs recomputation overhead
  - Peak memory vs batch size with and without AC
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Dict, List, Tuple, Optional

from graph_prof import GraphProfiler, NodeType
from ac_algorithm import ACDecision, simulate_peak_memory, budget_sweep


# ─── Shared style constants ──────────────────────────────────────────────────
TYPE_COLORS = {
    NodeType.PARAM:     "#2196F3",
    NodeType.ACT:       "#FF9800",
    NodeType.GRAD:      "#4CAF50",
    NodeType.OPT_STATE: "#AB47BC",
    NodeType.OTHER:     "#9E9E9E",
}
TYPE_NAMES = {
    NodeType.PARAM:     "Parameters",
    NodeType.ACT:       "Activations",
    NodeType.GRAD:      "Gradients",
    NodeType.OPT_STATE: "Opt States",
    NodeType.OTHER:     "Other",
}
# Canonical order for stacking
TYPE_ORDER = [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPT_STATE, NodeType.OTHER]

MB = 1024 ** 2


def _add_region_spans(ax, timeline, profiler):
    """Add subtle shaded background spans for forward/loss/backward/optimizer regions."""
    region_colors = {"forward": "#e3f2fd", "loss": "#fff3e0",
                     "backward": "#fce4ec", "optimizer": "#e8f5e9"}
    name_to_node = {n.name: n for n in profiler.nodes_list}

    # Find region boundaries
    prev_region = None
    start_idx = 0
    n_points = len(timeline)
    for i, entry in enumerate(timeline):
        node = name_to_node.get(entry["node_name"])
        region = profiler.node_region.get(node, "optimizer") if node else "optimizer"
        if region != prev_region:
            if prev_region is not None and prev_region in region_colors:
                ax.axvspan(start_idx, i, color=region_colors[prev_region], alpha=0.3)
            start_idx = i
            prev_region = region
    # Last region
    if prev_region in region_colors:
        ax.axvspan(start_idx, n_points, color=region_colors[prev_region], alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Memory Timeline (stacked area line chart — replaces bar snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_memory_timeline(profiler: GraphProfiler, title: str, path: str,
                         show_regions: bool = True):
    """
    Line chart of memory over execution time, stacked by tensor type.
    Uses the profiler's memory_timeline from the last measured iteration.
    """
    tl = profiler.memory_timeline
    if not tl:
        print(f"  No timeline data for {title}, skipping plot.")
        return

    x = np.arange(len(tl))
    # Build arrays for each tensor type
    stacks = {nt: np.array([e["breakdown"].get(nt, 0) / MB for e in tl]) for nt in TYPE_ORDER}

    fig, ax = plt.subplots(figsize=(12, 5))

    # Stacked area chart
    bottoms = np.zeros(len(tl))
    for nt in TYPE_ORDER:
        vals = stacks[nt]
        ax.fill_between(x, bottoms, bottoms + vals,
                        color=TYPE_COLORS[nt], alpha=0.6, label=TYPE_NAMES[nt])
        bottoms += vals

    # Total memory line on top
    totals = np.array([e["total_memory"] / MB for e in tl])
    ax.plot(x, totals, color="black", linewidth=0.8, alpha=0.7)

    # Mark the peak
    peak_idx = np.argmax(totals)
    ax.annotate(f"Peak: {totals[peak_idx]:.0f} MB",
                xy=(peak_idx, totals[peak_idx]),
                xytext=(peak_idx + len(tl)*0.05, totals[peak_idx] * 0.95),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                fontsize=9, color="red", fontweight="bold")

    if show_regions:
        _add_region_spans(ax, tl, profiler)
        # Add region labels at top
        name_to_node = {n.name: n for n in profiler.nodes_list}
        region_starts = {}
        for i, entry in enumerate(tl):
            node = name_to_node.get(entry["node_name"])
            region = profiler.node_region.get(node, "") if node else ""
            if region and region not in region_starts:
                region_starts[region] = i
        for region, start in region_starts.items():
            ax.text(start + 20, totals.max() * 1.02, region.capitalize(),
                    fontsize=8, fontstyle="italic", alpha=0.7)

    ax.set_xlabel("Execution Step (node index)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, len(tl))
    ax.set_ylim(0, totals.max() * 1.12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Memory Timeline Comparison (baseline vs AC)
# ─────────────────────────────────────────────────────────────────────────────

def plot_memory_comparison(profiler: GraphProfiler, recomputed_names: set,
                           title: str, path: str):
    """
    Overlay line chart: baseline memory (dashed) vs AC memory (solid) over
    execution, showing how checkpointing reduces the peak.
    """
    # Baseline timeline (from profiler)
    tl_base = profiler.memory_timeline
    if not tl_base:
        return

    # AC timeline (re-simulated)
    _, tl_ac = simulate_peak_memory(profiler, recomputed_names)

    x = np.arange(len(tl_base))
    base_total = np.array([e["total_memory"] / MB for e in tl_base])
    ac_total = np.array([e["total_memory"] / MB for e in tl_ac])

    # Stacked area for AC timeline (activations shown separately)
    act_mem_ac = np.array([e["breakdown"].get(NodeType.ACT, 0) / MB for e in tl_ac])
    other_mem_ac = ac_total - act_mem_ac

    fig, ax = plt.subplots(figsize=(12, 5))

    # Baseline: dashed line
    ax.plot(x, base_total, color="gray", linewidth=1.2, linestyle="--",
            alpha=0.7, label="Baseline (no AC)")

    # AC: solid colored fill
    ax.fill_between(x, 0, other_mem_ac, color="#90CAF9", alpha=0.5,
                    label="Non-activation mem (with AC)")
    ax.fill_between(x, other_mem_ac, ac_total, color="#FF9800", alpha=0.5,
                    label="Activation mem (with AC)")
    ax.plot(x, ac_total, color="#E65100", linewidth=0.8, alpha=0.8)

    # Mark peaks
    base_peak_idx = np.argmax(base_total)
    ac_peak_idx = np.argmax(ac_total)
    ax.axhline(y=base_total[base_peak_idx], color="gray", linestyle=":", alpha=0.4)
    ax.axhline(y=ac_total[ac_peak_idx], color="#E65100", linestyle=":", alpha=0.4)

    ax.annotate(f"Baseline peak: {base_total[base_peak_idx]:.0f} MB",
                xy=(base_peak_idx, base_total[base_peak_idx]),
                xytext=(len(tl_base)*0.6, base_total[base_peak_idx] + 30),
                fontsize=9, color="gray", fontweight="bold")
    ax.annotate(f"AC peak: {ac_total[ac_peak_idx]:.0f} MB",
                xy=(ac_peak_idx, ac_total[ac_peak_idx]),
                xytext=(len(tl_base)*0.6, ac_total[ac_peak_idx] - 50),
                fontsize=9, color="#E65100", fontweight="bold")

    # Shade the memory saved region
    saved_region = np.maximum(base_total - ac_total, 0)
    ax.fill_between(x, ac_total, ac_total + saved_region,
                    color="green", alpha=0.1, hatch="//",
                    label="Memory saved")

    _add_region_spans(ax, tl_base, profiler)

    ax.set_xlabel("Execution Step (node index)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, len(tl_base))
    ax.set_ylim(0, base_total.max() * 1.12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Greedy Algorithm Convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_greedy_convergence(decision: ACDecision, budget_mb: float,
                            title: str, path: str):
    """
    Line chart showing how peak memory decreases as the greedy algorithm
    evicts activations one by one.  Horizontal line = memory budget.
    """
    if not decision.iteration_log:
        return

    steps = [0] + [e["step"] for e in decision.iteration_log]
    peaks = [decision.baseline_peak / MB] + [e["projected_peak_mb"] for e in decision.iteration_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, peaks, marker="o", markersize=3, color="#1565C0", linewidth=1.5,
            label="Projected peak memory")
    ax.axhline(y=budget_mb, color="red", linestyle="--", linewidth=1,
               label=f"Budget ({budget_mb:.0f} MB)")
    ax.axhline(y=decision.baseline_peak / MB, color="gray", linestyle=":",
               alpha=0.5, label=f"Baseline ({decision.baseline_peak/MB:.0f} MB)")

    # Shade the region above budget
    ax.fill_between(steps, budget_mb, peaks, where=[p > budget_mb for p in peaks],
                    color="red", alpha=0.08)

    ax.set_xlabel("Greedy Step (activations evicted)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, max(steps))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Pareto Frontier — Memory vs Recomputation Overhead
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_frontier(profiler: GraphProfiler, title: str, path: str):
    """
    Sweep budget fractions and plot the trade-off: x = extra recompute time,
    y = peak memory.  Shows the Pareto curve of the compute-memory tradeoff.
    """
    fracs = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    results = budget_sweep(profiler, budget_fractions=fracs[1:])  # skip 1.0

    baseline_peak = profiler.avg_fwdbwd_peak / MB
    # Estimate total recomp overhead: sum of recomp_time for evicted activations
    from ac_algorithm import _build_activation_infos
    base_infos = _build_activation_infos(profiler)

    peaks = [baseline_peak]
    recomp_overheads = [0.0]  # ms of extra recomputation

    for frac, dec in results:
        peaks.append(dec.projected_peak / MB)
        # Sum of op_runtime for recomputed activations (approximate)
        overhead = sum(base_infos[n].op_runtime for n in dec.recompute
                       if n in base_infos)
        recomp_overheads.append(overhead)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recomp_overheads, peaks, marker="s", markersize=5, color="#D32F2F",
            linewidth=1.5, label="Memory vs Recomp. Overhead")

    # Label each point with the budget fraction
    all_fracs = fracs
    for i, (ro, pk, fr) in enumerate(zip(recomp_overheads, peaks, all_fracs)):
        ax.annotate(f"{fr*100:.0f}%", (ro, pk), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, alpha=0.8)

    ax.set_xlabel("Additional Recomputation Time (ms)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Peak Memory vs Batch Size — with and without AC
# ─────────────────────────────────────────────────────────────────────────────

def plot_peak_memory_with_ac(results_no_ac: Dict, results_with_ac: Dict,
                             path: str = "peak_memory_vs_batch_size_ac.png"):
    """
    Line chart comparing peak memory across batch sizes, with and without AC.
    results_no_ac / results_with_ac: {model: {bs: peak_mb}}
    """
    models = list(results_no_ac.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        bs_list = sorted(results_no_ac[model].keys())
        no_ac = [results_no_ac[model][bs] for bs in bs_list]
        with_ac = [results_with_ac[model].get(bs, None) for bs in bs_list]

        ax.plot(bs_list, no_ac, marker="o", linewidth=2, color="#1565C0",
                label="Without AC")
        valid_ac = [(bs, m) for bs, m in zip(bs_list, with_ac) if m is not None]
        if valid_ac:
            ax.plot([b for b, _ in valid_ac], [m for _, m in valid_ac],
                    marker="s", linewidth=2, color="#D32F2F", label="With AC")

        # Annotate values
        for bs, m in zip(bs_list, no_ac):
            ax.annotate(f"{m:.0f}", (bs, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#1565C0")
        for bs, m in valid_ac:
            ax.annotate(f"{m:.0f}", (bs, m), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#D32F2F")

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title(f"{model}: Peak Memory vs Batch Size")
        ax.legend(fontsize=9)
        ax.set_xticks(bs_list)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Activation Recompute Ratio Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_recompute_ratio_distribution(profiler: GraphProfiler, decision: ACDecision,
                                      title: str, path: str):
    """
    Scatter plot of each activation: x = recomp_time (ms), y = memory_size (MB).
    Color indicates retain (blue) vs recompute (red).  Lines of constant
    recompute_ratio shown for reference.
    """
    from ac_algorithm import _build_activation_infos
    infos = _build_activation_infos(profiler)

    retain_x, retain_y = [], []
    recomp_x, recomp_y = [], []

    for name, ai in infos.items():
        x_val = max(ai.recomp_time, 1e-4)  # ms
        y_val = ai.memory_size / MB
        if name in decision.recompute:
            recomp_x.append(x_val)
            recomp_y.append(y_val)
        else:
            retain_x.append(x_val)
            retain_y.append(y_val)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(retain_x, retain_y, color="#1565C0", alpha=0.5, s=15,
               label=f"Retained ({len(retain_x)})", zorder=3)
    ax.scatter(recomp_x, recomp_y, color="#D32F2F", alpha=0.5, s=15,
               label=f"Recomputed ({len(recomp_x)})", zorder=3)

    # Draw iso-ratio lines: ratio = mem / time, so mem = ratio * time
    if retain_x or recomp_x:
        all_x = retain_x + recomp_x
        x_range = np.linspace(min(all_x) * 0.5, max(all_x) * 1.5, 100)
        for ratio_val in [0.1, 1.0, 10.0, 100.0]:
            y_line = ratio_val * x_range / MB  # ratio is bytes/ms, x is ms, y is MB
            ax.plot(x_range, y_line, color="gray", linewidth=0.5, alpha=0.3, linestyle="--")
            # Label at right edge
            label_y = ratio_val * x_range[-1] / MB
            all_y = retain_y + recomp_y
            if label_y <= max(all_y) * 1.2:
                ax.text(x_range[-1], label_y, f"ratio={ratio_val:.0f}",
                        fontsize=6, alpha=0.5, va="bottom")

    ax.set_xlabel("Recomputation Time (ms)")
    ax.set_ylabel("Memory Size (MB)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
