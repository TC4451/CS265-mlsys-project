#!/usr/bin/env python3
"""
Visualize an FX graph dumped as text (for example: comp_graph_starter.txt).

Outputs:
1) A Graphviz .dot file with node/edge structure.
2) A PNG "lane view" (forward/loss/backward/optimizer) using matplotlib.

This script is dependency-light:
- Required: matplotlib
- Optional: graphviz command-line tool (only if you later want to render the .dot externally)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt


@dataclass
class GraphNode:
    idx: int
    name: str
    op: str
    target: str
    inputs: List[str]
    raw: str


PLACEHOLDER_RE = re.compile(
    r"^\s*%([A-Za-z0-9_]+)\s*:\s*\[num_users=\d+\]\s*=\s*placeholder\[target=([^\]]+)\]\s*$"
)
CALL_NODE_RE = re.compile(
    r"^\s*%([A-Za-z0-9_]+)\s*:\s*\[num_users=\d+\]\s*=\s*([A-Za-z_]+)"
    r"\[target=([^\]]+)\]\(args\s*=\s*(.*),\s*kwargs\s*=\s*\{.*\}\)\s*$"
)
OUTPUT_RE = re.compile(r"^\s*return\s+\[(.*)\]\s*$")
PERCENT_NAME_RE = re.compile(r"%([A-Za-z0-9_]+)")
NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


REGION_COLORS = {
    "forward": "#8ecae6",
    "loss": "#ffb703",
    "backward": "#fb8500",
    "optimizer": "#90be6d",
    "unknown": "#adb5bd",
}


def parse_fx_graph_text(path: Path) -> Tuple[List[GraphNode], List[str]]:
    """
    Parse the FX graph block from a text dump.
    Returns (nodes, output_refs).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    in_graph = False
    nodes: List[GraphNode] = []
    output_refs: List[str] = []

    for line in lines:
        if not in_graph:
            if line.strip() == "graph():":
                in_graph = True
            continue

        if line.strip().startswith("="):
            break

        out_match = OUTPUT_RE.match(line)
        if out_match:
            # Output list in this dump is name-based (no % prefix), e.g. [None, copy__240, ...]
            maybe_names = NAME_RE.findall(out_match.group(1))
            output_refs = [n for n in maybe_names if n != "None"]
            break

        ph = PLACEHOLDER_RE.match(line)
        if ph:
            name, target = ph.groups()
            nodes.append(
                GraphNode(
                    idx=len(nodes),
                    name=name,
                    op="placeholder",
                    target=target,
                    inputs=[],
                    raw=line.strip(),
                )
            )
            continue

        m = CALL_NODE_RE.match(line)
        if not m:
            continue

        name, op, target, args_blob = m.groups()
        inputs = PERCENT_NAME_RE.findall(args_blob)
        nodes.append(
            GraphNode(
                idx=len(nodes),
                name=name,
                op=op,
                target=target,
                inputs=inputs,
                raw=line.strip(),
            )
        )

    return nodes, output_refs


def classify_regions(nodes: Sequence[GraphNode]) -> Dict[str, str]:
    """
    Match project convention:
    - forward ends at separator.sep
    - loss is between sep and sep_backward
    - backward is between sep_backward and first optimizer op
    - optimizer is everything after optimizer start
    """
    sep_idx: Optional[int] = None
    sep_bwd_idx: Optional[int] = None
    opt_start_idx: Optional[int] = None

    for n in nodes:
        if n.target == "torch.ops.separator.sep.default":
            sep_idx = n.idx
        elif n.target == "torch.ops.separator.sep_backward.default":
            sep_bwd_idx = n.idx

    if sep_bwd_idx is not None:
        for n in nodes[sep_bwd_idx + 1 :]:
            if "_foreach_" in n.target or n.target == "torch.ops.aten._fused_adam.default":
                opt_start_idx = n.idx
                break

    out: Dict[str, str] = {}
    for n in nodes:
        region = "unknown"
        if sep_idx is not None and n.idx <= sep_idx:
            region = "forward"
        elif sep_bwd_idx is not None and n.idx < sep_bwd_idx:
            region = "loss"
        elif opt_start_idx is not None and n.idx < opt_start_idx:
            region = "backward"
        elif opt_start_idx is not None and n.idx >= opt_start_idx:
            region = "optimizer"
        out[n.name] = region
    return out


def short_target(target: str, max_len: int = 36) -> str:
    if len(target) <= max_len:
        return target
    return target[: max_len - 3] + "..."


def should_keep_node(
    node: GraphNode,
    include_placeholders: bool,
    include_getitem_copy: bool,
) -> bool:
    if node.op == "placeholder" and not include_placeholders:
        return False

    if not include_getitem_copy and (
        node.target == "operator.getitem" or node.target == "torch.ops.aten.copy_.default"
    ):
        return False

    return True


def select_nodes(
    nodes: Sequence[GraphNode],
    include_placeholders: bool,
    include_getitem_copy: bool,
    max_nodes: int,
) -> List[GraphNode]:
    kept = [
        n
        for n in nodes
        if should_keep_node(n, include_placeholders, include_getitem_copy)
    ]
    if len(kept) <= max_nodes:
        return kept

    # Keep boundary/semantic nodes first, then fill by order.
    must_keep = {
        "torch.ops.separator.sep.default",
        "torch.ops.separator.sep_backward.default",
    }
    anchor = [n for n in kept if n.target in must_keep]
    remainder = [n for n in kept if n.target not in must_keep]
    budget = max(0, max_nodes - len(anchor))
    return anchor + remainder[:budget]


def build_edge_list(nodes: Sequence[GraphNode], output_refs: Sequence[str]) -> List[Tuple[str, str]]:
    present: Set[str] = {n.name for n in nodes}
    edges: List[Tuple[str, str]] = []

    for n in nodes:
        for src in n.inputs:
            if src in present:
                edges.append((src, n.name))

    if output_refs:
        for src in output_refs:
            if src in present:
                edges.append((src, "output"))

    return edges


def write_dot(
    nodes: Sequence[GraphNode],
    edges: Sequence[Tuple[str, str]],
    regions: Dict[str, str],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("digraph FXGraph {")
    lines.append('  rankdir="LR";')
    lines.append('  splines="spline";')
    lines.append('  overlap="false";')
    lines.append('  node [shape=box, style="rounded,filled", fontsize=9, fontname="Helvetica"];')
    lines.append('  edge [color="#7f8c8d", arrowsize=0.4, penwidth=0.8];')

    for n in nodes:
        region = regions.get(n.name, "unknown")
        color = REGION_COLORS.get(region, REGION_COLORS["unknown"])
        label = f"{n.name}\\n{short_target(n.target)}"
        lines.append(f'  "{n.name}" [label="{label}", fillcolor="{color}"];')

    lines.append('  "output" [label="output", shape=oval, fillcolor="#e9ecef"];')

    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}";')

    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def draw_lane_png(
    nodes: Sequence[GraphNode],
    edges: Sequence[Tuple[str, str]],
    regions: Dict[str, str],
    out_path: Path,
) -> None:
    region_order = ["forward", "loss", "backward", "optimizer", "unknown"]
    lane_y = {r: float(len(region_order) - 1 - i) for i, r in enumerate(region_order)}

    # Per-lane x positions in node order.
    x_counter = {r: 0 for r in region_order}
    pos: Dict[str, Tuple[float, float]] = {}

    for n in nodes:
        r = regions.get(n.name, "unknown")
        if r not in x_counter:
            r = "unknown"
        x_counter[r] += 1
        x = float(x_counter[r])
        y = lane_y[r]
        pos[n.name] = (x, y)

    # Output node on the far right.
    max_x = max([p[0] for p in pos.values()], default=1.0)
    pos["output"] = (max_x + 2.0, lane_y["unknown"])

    fig_w = max(12, min(60, max_x * 0.22))
    fig_h = 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)

    # Lane backgrounds
    for r in region_order:
        y = lane_y[r]
        ax.axhspan(y - 0.35, y + 0.35, color=REGION_COLORS[r], alpha=0.08, zorder=0)
        ax.text(0.2, y + 0.42, r, fontsize=10, fontweight="bold")

    # Edges
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color="#6c757d", alpha=0.2, linewidth=0.6, zorder=1)

    # Nodes
    for n in nodes:
        x, y = pos[n.name]
        region = regions.get(n.name, "unknown")
        ax.scatter([x], [y], s=26, color=REGION_COLORS.get(region, REGION_COLORS["unknown"]), zorder=2)

    # Output node
    ax.scatter([pos["output"][0]], [pos["output"][1]], s=44, color="#495057", zorder=3)
    ax.text(pos["output"][0] + 0.15, pos["output"][1] + 0.05, "output", fontsize=8)

    # Label a subset to keep readability.
    label_every = 1 if len(nodes) <= 120 else 5
    for i, n in enumerate(nodes):
        if i % label_every != 0:
            continue
        x, y = pos[n.name]
        label = short_target(n.target, max_len=22).replace("torch.ops.", "")
        ax.text(x + 0.07, y + 0.04, f"{n.name}: {label}", fontsize=6, alpha=0.85)

    ax.set_title("FX Graph Lane View")
    ax.set_xlim(0, pos["output"][0] + 1.5)
    ax.set_ylim(-0.8, len(region_order) - 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    all_nodes: Sequence[GraphNode],
    kept_nodes: Sequence[GraphNode],
    edges: Sequence[Tuple[str, str]],
    regions: Dict[str, str],
    out_path: Path,
) -> None:
    op_count: Dict[str, int] = {}
    reg_count: Dict[str, int] = {"forward": 0, "loss": 0, "backward": 0, "optimizer": 0, "unknown": 0}
    for n in all_nodes:
        op_count[n.target] = op_count.get(n.target, 0) + 1
        r = regions.get(n.name, "unknown")
        reg_count[r] = reg_count.get(r, 0) + 1

    lines = []
    lines.append(f"Total parsed nodes: {len(all_nodes)}")
    lines.append(f"Visualized nodes: {len(kept_nodes)}")
    lines.append(f"Visualized edges: {len(edges)}")
    lines.append("")
    lines.append("Region counts:")
    for r in ["forward", "loss", "backward", "optimizer", "unknown"]:
        lines.append(f"  {r:10s}: {reg_count.get(r, 0)}")
    lines.append("")
    lines.append("Top 20 targets by frequency:")
    for target, cnt in sorted(op_count.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"  {cnt:4d}  {target}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize FX graph text dump.")
    p.add_argument("--input", type=Path, default=Path("comp_graph_starter.txt"), help="Path to graph text dump.")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix (without extension). Default: <input stem>_viz in same folder.",
    )
    p.add_argument(
        "--include-placeholders",
        action="store_true",
        help="Include placeholder nodes (model params, optimizer state, inputs).",
    )
    p.add_argument(
        "--include-getitem-copy",
        action="store_true",
        help="Include getitem and copy_ nodes (can make graph very dense).",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=350,
        help="Maximum number of nodes to include in visualization.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path: Path = args.input
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_prefix = args.output_prefix
    if out_prefix is None:
        out_prefix = in_path.with_name(f"{in_path.stem}_viz")

    all_nodes, output_refs = parse_fx_graph_text(in_path)
    if not all_nodes:
        raise RuntimeError(f"Could not parse graph nodes from: {in_path}")

    regions = classify_regions(all_nodes)
    kept_nodes = select_nodes(
        all_nodes,
        include_placeholders=args.include_placeholders,
        include_getitem_copy=args.include_getitem_copy,
        max_nodes=args.max_nodes,
    )
    edges = build_edge_list(kept_nodes, output_refs)

    dot_path = out_prefix.with_suffix(".dot")
    png_path = out_prefix.with_suffix(".png")
    summary_path = out_prefix.with_suffix(".summary.txt")

    write_dot(kept_nodes, edges, regions, dot_path)
    draw_lane_png(kept_nodes, edges, regions, png_path)
    write_summary(all_nodes, kept_nodes, edges, regions, summary_path)

    print(f"Wrote DOT:     {dot_path}")
    print(f"Wrote PNG:     {png_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
