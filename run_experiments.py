"""
Full experiment runner for CS265 Systems Project — Phases 1 and 2.

Phase 1 deliverables:
  4(a) Computation/memory profiling stats + static analysis   (saved to .txt)
  4(b) Peak memory vs batch size (w/o AC)                     (saved to .png)

Phase 2 deliverables:
  - μ-TWO algorithm decisions for each model/batch-size       (saved to .txt)
  - Memory timeline line charts (baseline and with AC)        (saved to .png)
  - Greedy convergence plots                                  (saved to .png)
  - Pareto frontier (memory vs recompute cost)                (saved to .png)
  - Peak memory vs batch size with and without AC             (saved to .png)
  - Activation scatter (retain vs recompute)                  (saved to .png)

Models: ResNet-152 and BERT (as specified in the project description).

Usage:
    conda activate cs265
    python run_experiments.py
"""
import gc
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import resnet152
from transformers import BertModel, BertConfig
from graph_tracer import SEPFunction, compile
from graph_prof import GraphProfiler, NodeType
from ac_algorithm import mu_two_algorithm, format_ac_decision, ACDecision
from ac_visualize import (
    plot_memory_timeline,
    plot_memory_comparison,
    plot_greedy_convergence,
    plot_pareto_frontier,
    plot_peak_memory_with_ac,
    plot_recompute_ratio_distribution,
)
from graph_rewriter import apply_ac_to_graph, verify_correctness


# ------------------------------ Config ------------------------------
MODELS = ["Resnet152", "BERT"]
BATCH_SIZES = {
    "Resnet152": [2, 4, 8, 16],
    "BERT":      [2, 4, 8, 16],
}
DEFAULT_BS = {"Resnet152": 4, "BERT": 4}

# Memory budget as fraction of baseline peak (target: keep 50% of activation
# memory, which is the default in mu_two_algorithm when budget=None).
AC_BUDGET_FRACTION = 0.75  # 75% of baseline = aim to save 25%

# Output directories
PLOT_DIR = "outputs/plots"
STATS_DIR = "outputs/profiling_stats"
GRAPH_DIR = "outputs/comp_graphs"
AC_DIR = "outputs/ac_decisions"

# Create output dirs if they don't exist
for d in [PLOT_DIR, STATS_DIR, GRAPH_DIR, AC_DIR]:
    os.makedirs(d, exist_ok=True)


# -------------------------- Experiment setup ------------------------
def make_experiment(model_name: str, batch_size: int):
    """Build model, synthetic inputs, optimizer, and one train_step closure."""
    dev = torch.device("cuda")

    if model_name == "Resnet152":
        with torch.device(dev):
            model = resnet152()
        inp = torch.randn(batch_size, 3, 224, 224, device=dev)
        target = torch.randint(0, 10, (batch_size,), device=dev)
        inputs = (inp, target)

        def train_step(model, opt, inputs):
            logits = model(inputs[0])
            loss = F.cross_entropy(logits, inputs[1])
            loss = SEPFunction.apply(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    elif model_name == "BERT":
        config = BertConfig(
            vocab_size=30522, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072, max_position_embeddings=128,
        )
        with torch.device(dev):
            model = BertModel(config)
        seq_len = 128
        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=dev)
        attention_mask = torch.ones(batch_size, seq_len, device=dev)
        target = torch.randn(batch_size, 768, device=dev)
        inputs = (input_ids, attention_mask, target)

        def train_step(model, opt, inputs):
            out = model(input_ids=inputs[0], attention_mask=inputs[1])
            logits = out.last_hidden_state[:, 0, :]  # [CLS] token
            loss = F.mse_loss(logits, inputs[2])
            loss = SEPFunction.apply(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Fused Adam: optimizer updates appear as graph ops for profiling.
    optimizer = optim.Adam(model.parameters(), lr=1e-4, fused=True, capturable=True)

    # Initialize optimizer state so profiling includes optimizer tensors.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.rand_like(p)
    optimizer.step()
    optimizer.zero_grad()

    return model, optimizer, inputs, train_step


# ----------------------------- Profiling ----------------------------
def profile(model_name: str, batch_size: int):
    """Trace, warmup, measure, and return the profiler, GraphModule, and args.

    Returns:
        (profiler, gm, flat_args) — profiler has aggregated stats;
        gm and flat_args can be reused for Phase 3 graph rewriting.
    """
    torch.cuda.empty_cache()
    gc.collect()

    model, optimizer, inputs, train_step = make_experiment(model_name, batch_size)
    result_box = {}

    def graph_transformation(gm, args):
        graph_path = f"{GRAPH_DIR}/comp_graph_{model_name}_bs{batch_size}.txt"
        with open(graph_path, "w") as f:
            f.write(str(gm.graph))
        print(f"  Saved graph to {graph_path}")
        gp = GraphProfiler(gm)
        with torch.no_grad():
            # 2 warmup + 3 measurement iterations
            for _ in range(2):
                gp.run(*args)
            gp.reset_stats()
            for _ in range(3):
                gp.run(*args)
            gp.aggregate_stats()
            gp.print_stats()
        result_box["profiler"] = gp
        result_box["gm"] = gm
        result_box["args"] = args
        return gm

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optimizer, inputs)
    return result_box["profiler"], result_box["gm"], result_box["args"]


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Profiling and Memory Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_phase1():
    """Run Phase 1 experiments: profile all models/batch-sizes, generate plots."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Graph Profiling")
    print("=" * 60)

    results_no_ac = {}
    profilers = {}  # {(model, bs): profiler}  — reused by Phase 2
    graph_data = {}  # {(model, bs): (gm, args)}  — reused by Phase 3

    for model in MODELS:
        results_no_ac[model] = {}
        for bs in BATCH_SIZES[model]:
            print(f"\n{'=' * 50}\n  {model}  bs={bs}\n{'=' * 50}")
            try:
                p, gm, args = profile(model, bs)
                peak_mb = p.get_fwdbwd_peak_memory() / (1024 ** 2)
                results_no_ac[model][bs] = peak_mb
                profilers[(model, bs)] = p
                graph_data[(model, bs)] = (gm, args)
                print(f"  => Fwd+Bwd peak: {peak_mb:.1f} MB")
                p.save_stats(f"{STATS_DIR}/profiling_stats_{model}_bs{bs}.txt")

                # Phase 1 visualization: memory timeline line chart (TA feedback)
                if bs == DEFAULT_BS.get(model):
                    plot_memory_timeline(
                        p,
                        title=f"{model} (bs={bs}) — Memory Timeline",
                        path=f"{PLOT_DIR}/memory_timeline_{model}_bs{bs}.png",
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  => OOM at bs={bs}")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

    # Peak memory vs batch size (line chart, no AC)
    plot_peak_memory_line(results_no_ac, path=f"{PLOT_DIR}/peak_memory_vs_batch_size.png")

    return results_no_ac, profilers, graph_data


def plot_peak_memory_line(results, path="peak_memory_vs_batch_size.png"):
    """Peak memory vs batch size — line chart (updated from bar graph)."""
    models = list(results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data = results[model]
        bs_list = sorted(data.keys())
        mem = [data[bs] for bs in bs_list]
        ax.plot(bs_list, mem, marker="o", linewidth=2, color="steelblue")
        for b, m in zip(bs_list, mem):
            ax.annotate(f"{m:.0f}", (b, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title(f"{model}: Fwd+Bwd Peak Memory vs Batch Size (w/o AC)")
        ax.set_xticks(bs_list)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: μ-TWO Activation Checkpointing Algorithm
# ═══════════════════════════════════════════════════════════════════════

def run_phase2(results_no_ac, profilers):
    """
    Run Phase 2: apply the μ-TWO algorithm to each profiled model/batch-size,
    generate comparison plots and decision reports.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: μ-TWO Activation Checkpointing Algorithm")
    print("=" * 60)

    MB = 1024 ** 2
    results_with_ac = {}
    decisions = {}

    for model in MODELS:
        results_with_ac[model] = {}
        for bs in BATCH_SIZES[model]:
            key = (model, bs)
            if key not in profilers:
                continue

            p = profilers[key]
            baseline_peak = p.avg_fwdbwd_peak
            budget = baseline_peak * AC_BUDGET_FRACTION

            print(f"\n{'=' * 50}")
            print(f"  {model}  bs={bs}  —  μ-TWO Algorithm")
            print(f"  Baseline peak: {baseline_peak/MB:.1f} MB")
            print(f"  Budget ({AC_BUDGET_FRACTION*100:.0f}%): {budget/MB:.1f} MB")
            print(f"{'=' * 50}")

            # Run μ-TWO algorithm
            decision = mu_two_algorithm(p, memory_budget=budget)
            decisions[key] = decision
            results_with_ac[model][bs] = decision.projected_peak / MB

            # Print and save the decision report
            report = format_ac_decision(decision, model_name=f"{model} bs={bs}")
            print(report)
            report_path = f"{AC_DIR}/ac_decision_{model}_bs{bs}.txt"
            with open(report_path, "w") as f:
                f.write(report + "\n")
            print(f"  Saved {report_path}")

            # ── Phase 2 visualizations (default batch size only) ──
            if bs == DEFAULT_BS.get(model):
                recomputed_set = set(decision.recompute)

                # 1. Memory timeline comparison (baseline vs AC)
                plot_memory_comparison(
                    p, recomputed_set,
                    title=f"{model} (bs={bs}) — Baseline vs AC Memory",
                    path=f"{PLOT_DIR}/memory_comparison_{model}_bs{bs}.png",
                )

                # 2. Greedy convergence: peak memory vs step
                plot_greedy_convergence(
                    decision, budget_mb=budget / MB,
                    title=f"{model} (bs={bs}) — Greedy Convergence",
                    path=f"{PLOT_DIR}/greedy_convergence_{model}_bs{bs}.png",
                )

                # 3. Pareto frontier: memory vs recompute overhead
                plot_pareto_frontier(
                    p,
                    title=f"{model} (bs={bs}) — Memory vs Recompute Overhead",
                    path=f"{PLOT_DIR}/pareto_frontier_{model}_bs{bs}.png",
                )

                # 4. Activation scatter plot (retain vs recompute)
                plot_recompute_ratio_distribution(
                    p, decision,
                    title=f"{model} (bs={bs}) — Activation Recompute Decision",
                    path=f"{PLOT_DIR}/activation_scatter_{model}_bs{bs}.png",
                )

    # 5. Peak memory vs batch size: with and without AC
    plot_peak_memory_with_ac(
        results_no_ac, results_with_ac,
        path=f"{PLOT_DIR}/peak_memory_vs_batch_size_ac.png",
    )

    return results_with_ac, decisions


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Graph Rewriting and Verification
# ═══════════════════════════════════════════════════════════════════════

def _measure_latency(gm, flat_args, warmup=2, measure=5):
    """Measure iteration latency in ms (warmup + CUDA-timed measurement)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        with torch.no_grad():
            gm(*flat_args)
    torch.cuda.synchronize()
    times = []
    for _ in range(measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            gm(*flat_args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def run_phase3(profilers, graph_data, decisions):
    """
    Apply Phase 2's AC decisions to the actual FX graphs:
      1. Rewrite each graph (insert recomputation subgraphs)
      2. Verify gradient correctness (default BS only)
      3. Measure iteration latency with and without AC (all batch sizes)
      4. Generate deliverable 4(c): latency vs batch size plot

    Runs across ALL batch sizes for latency measurement.
    Correctness verification only on DEFAULT_BS to save time.
    """
    import copy
    print("\n" + "=" * 60)
    print("  PHASE 3: Graph Rewriting and Verification")
    print("=" * 60)

    MB = 1024 ** 2
    phase3_results = {}
    latency_baseline = {}   # {model: {bs: ms}}
    latency_ac = {}         # {model: {bs: ms}}

    for model in MODELS:
        latency_baseline[model] = {}
        latency_ac[model] = {}

        for bs in BATCH_SIZES[model]:
            key = (model, bs)
            if key not in profilers or key not in graph_data or key not in decisions:
                continue

            p = profilers[key]
            gm, flat_args = graph_data[key]
            decision = decisions[key]
            is_default = (bs == DEFAULT_BS[model])

            print(f"\n{'=' * 50}")
            print(f"  {model}  bs={bs}  —  Phase 3")
            print(f"  Activations to recompute: {len(decision.recompute)}")
            print(f"{'=' * 50}")

            # ── Apply graph rewriting ──
            print(f"  Rewriting graph...")
            modified_gm = copy.deepcopy(gm)
            modified_gm = apply_ac_to_graph(modified_gm, p, decision)

            if is_default:
                # Save modified graph for inspection
                mod_graph_path = f"{GRAPH_DIR}/comp_graph_{model}_bs{bs}_ac.txt"
                with open(mod_graph_path, "w") as f:
                    f.write(str(modified_gm.graph))
                print(f"    Saved modified graph to {mod_graph_path}")

                # Verify correctness (default BS only — expensive)
                print(f"  Verifying gradient correctness...")
                correct = verify_correctness(gm, modified_gm, flat_args)
                print(f"    Correctness: {'PASS' if correct else 'FAIL'}")
            else:
                correct = None  # skip verification for non-default BS

            # ── Measure latency ──
            print(f"  Measuring latency...")
            base_latency = _measure_latency(gm, flat_args)
            ac_latency = _measure_latency(modified_gm, flat_args)
            overhead_ms = ac_latency - base_latency
            overhead_pct = overhead_ms / base_latency * 100

            print(f"    Baseline: {base_latency:.2f} ms  |  AC: {ac_latency:.2f} ms  "
                  f"|  Overhead: {overhead_ms:+.2f} ms ({overhead_pct:+.1f}%)")

            latency_baseline[model][bs] = base_latency
            latency_ac[model][bs] = ac_latency

            phase3_results[key] = {
                "correct": correct,
                "baseline_latency_ms": base_latency,
                "ac_latency_ms": ac_latency,
                "overhead_ms": overhead_ms,
                "overhead_pct": overhead_pct,
                "baseline_peak_mb": decision.baseline_peak / MB,
                "projected_peak_mb": decision.projected_peak / MB,
            }

    # ── Deliverable 4(c): Iteration latency vs batch size (w and w/o AC) ──
    plot_latency_vs_batch_size(latency_baseline, latency_ac,
                               path=f"{PLOT_DIR}/latency_vs_batch_size.png")

    # Print Phase 3 summary
    print(f"\n{'=' * 80}")
    print(f"  PHASE 3 SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<12} {'BS':>4} {'Correct':>8} {'Base(ms)':>10} {'AC(ms)':>10} "
          f"{'Overhead':>10} {'PeakSaved':>10}")
    print(f"{'-' * 80}")
    for key, res in sorted(phase3_results.items()):
        model, bs = key
        saved_pct = (1 - res["projected_peak_mb"] / res["baseline_peak_mb"]) * 100
        corr_str = 'PASS' if res['correct'] else ('FAIL' if res['correct'] is not None else '—')
        print(f"{model:<12} {bs:>4} {corr_str:>8} "
              f"{res['baseline_latency_ms']:>9.1f}  {res['ac_latency_ms']:>9.1f}  "
              f"{res['overhead_pct']:>+9.1f}% {saved_pct:>9.1f}%")
    print(f"{'=' * 80}")

    return phase3_results


def plot_latency_vs_batch_size(latency_baseline, latency_ac,
                               path="latency_vs_batch_size.png"):
    """Deliverable 4(c): Iteration latency vs mini-batch size (w and w/o AC)."""
    models = list(latency_baseline.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        bs_list = sorted(latency_baseline[model].keys())
        base = [latency_baseline[model][bs] for bs in bs_list]
        ac = [latency_ac[model].get(bs) for bs in bs_list]

        ax.plot(bs_list, base, marker="o", linewidth=2, color="#1565C0",
                label="Without AC")
        valid_ac = [(b, l) for b, l in zip(bs_list, ac) if l is not None]
        if valid_ac:
            ax.plot([b for b, _ in valid_ac], [l for _, l in valid_ac],
                    marker="s", linewidth=2, color="#D32F2F", label="With AC")

        for b, l in zip(bs_list, base):
            ax.annotate(f"{l:.1f}", (b, l), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#1565C0")
        for b, l in valid_ac:
            ax.annotate(f"{l:.1f}", (b, l), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#D32F2F")

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Iteration Latency (ms)")
        ax.set_title(f"{model}: Latency vs Batch Size")
        ax.legend(fontsize=9)
        ax.set_xticks(bs_list)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results_no_ac, results_with_ac, decisions):
    """Print a final summary table comparing baseline vs AC across all settings."""
    MB = 1024 ** 2
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY: Baseline vs μ-TWO Activation Checkpointing")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'BS':>4} {'Baseline':>10} {'With AC':>10} {'Saved':>10} {'Saved%':>8} {'#Recomp':>8}")
    print(f"{'-' * 70}")
    for model in MODELS:
        for bs in sorted(results_no_ac.get(model, {}).keys()):
            base = results_no_ac[model].get(bs)
            ac = results_with_ac.get(model, {}).get(bs)
            key = (model, bs)
            dec = decisions.get(key)
            if base is not None and ac is not None:
                saved = base - ac
                pct = saved / base * 100
                n_recomp = len(dec.recompute) if dec else 0
                print(f"{model:<12} {bs:>4} {base:>9.1f}M {ac:>9.1f}M "
                      f"{saved:>9.1f}M {pct:>7.1f}% {n_recomp:>8}")
            elif base is not None:
                print(f"{model:<12} {bs:>4} {base:>9.1f}M {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
    print(f"{'=' * 70}")


# ------------------------------- Main -------------------------------
if __name__ == "__main__":
    # Phase 1: profiling
    results_no_ac, profilers, graph_data = run_phase1()

    # Phase 2: μ-TWO activation checkpointing algorithm
    results_with_ac, decisions = run_phase2(results_no_ac, profilers)

    # Phase 3: graph rewriting and verification
    phase3_results = run_phase3(profilers, graph_data, decisions)

    # Summary
    print_summary(results_no_ac, results_with_ac, decisions)
