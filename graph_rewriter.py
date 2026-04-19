"""
Phase 3: Graph Extractor and Rewriter for Activation Checkpointing.

For each activation that Phase 2 marked for recomputation, this module:
  1. Finds the available inputs (retained activations, params, placeholders)
     by walking backward through the forward graph.
  2. Extracts the minimal recomputation subgraph using
     _extract_graph_with_inputs_outputs().
  3. Inserts the subgraph into the backward pass just before the
     activation's first backward consumer.
  4. Redirects backward uses from the original node to the recomputed one.

The rewritten graph produces identical gradients but uses less peak memory
because recomputed activations are freed after their last forward use.
"""

import operator
from typing import Dict, List, Set, Tuple
import torch
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch._functorch._aot_autograd.descriptors import AOTOutput

from graph_prof import GraphProfiler, NodeType, OP
from ac_algorithm import ACDecision, _build_activation_infos


# ─────────────────────────────────────────────────────────────────────────────
# Helper: replace only uses that come after a given node
# ─────────────────────────────────────────────────────────────────────────────

def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    """
    Replace all uses of old_node with new_node, but ONLY for nodes that
    appear after new_node in the graph.  Forward-pass uses keep pointing
    to old_node; backward-pass uses switch to new_node.

    We walk the graph in reverse and stop when we reach new_node. Any node
    that uses old_node and is after new_node gets its input swapped.
    """
    old_node_users = dict(old_node.users)  # snapshot — users dict changes during iteration
    for node in reversed(list(graph.nodes)):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find available inputs for recomputing one activation
# ─────────────────────────────────────────────────────────────────────────────

def find_recomp_inputs(
    act_node: fx.Node,
    retained_names: Set[str],
    act_names: Set[str],
) -> List[fx.Node]:
    """
    Walk backward from act_node through the forward graph until we reach
    nodes whose values will be available in the backward pass:
      - Placeholders (graph inputs — always in memory)
      - Parameters (always in memory)
      - Retained activations (checkpointed — still in memory)
      - Non-activation forward ops (e.g., constants, shapes)

    Any recomputed activation on the path is NOT available, so we walk
    through it to find its own available inputs.

    Returns the list of available input nodes (no duplicates, order preserved).
    """
    inputs = []
    seen = set()
    stack = list(act_node.all_input_nodes)

    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen.add(node.name)

        # A node is "available" if it's not a recomputed activation.
        # Recomputed activations are in act_names but NOT in retained_names.
        is_recomputed_act = (node.name in act_names) and (node.name not in retained_names)

        if is_recomputed_act:
            # This input won't be in memory — trace through it to find
            # available nodes deeper in the graph.
            stack.extend(node.all_input_nodes)
        else:
            # This node is available (placeholder, param, retained act, or
            # non-activation forward op like a constant/shape).
            inputs.append(node)

    # Deduplicate while preserving order
    seen_inputs = set()
    unique_inputs = []
    for n in inputs:
        if n.name not in seen_inputs:
            seen_inputs.add(n.name)
            unique_inputs.append(n)

    return unique_inputs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Apply activation checkpointing to a GraphModule
# ─────────────────────────────────────────────────────────────────────────────

def apply_ac_to_graph(
    gm: fx.GraphModule,
    profiler: GraphProfiler,
    decision: ACDecision,
) -> fx.GraphModule:
    """
    Rewrite the FX graph to implement activation checkpointing decisions.

    For each recomputed activation (sorted by first backward use, earliest
    first so insertions don't shift later insertion points):
      1. Find available inputs via backward walk.
      2. Extract the minimal recomputation subgraph.
      3. Insert subgraph nodes before the first backward consumer.
      4. Redirect backward uses to the recomputed node.

    Returns the modified GraphModule (modified in-place).
    """
    recomputed_names = set(decision.recompute)
    retained_names = set(decision.retain)
    act_names = set(n.name for n in profiler.activation_nodes)

    # Build name->node map for the current graph
    name_to_node: Dict[str, fx.Node] = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node

    # Sort recomputed activations by first_bwd_use (earliest first).
    # This ensures earlier insertions don't shift later insertion points,
    # because we insert before a specific backward node and nodes added
    # before an earlier point don't affect later points.
    recomp_acts = []
    for act_node in profiler.activation_nodes:
        if act_node.name in recomputed_names:
            info = profiler.act_info[act_node]
            if info["first_bwd_use"] is not None:
                recomp_acts.append((act_node, info))

    recomp_acts.sort(key=lambda x: x[1]["first_bwd_use"])

    rewrite_count = 0
    skip_count = 0

    for act_node, act_info in recomp_acts:
        act_name = act_node.name
        first_bwd_idx = act_info["first_bwd_use"]
        first_bwd_node = profiler.nodes_list[first_bwd_idx]

        # Skip in-place ops (e.g., relu_).  They mutate their input tensor
        # rather than creating a new output, so _extract_graph_with_inputs_outputs
        # cannot extract them as subgraph outputs.  The target name usually ends
        # with an underscore (PyTorch in-place convention).
        target_name = str(getattr(act_node, 'target', ''))
        if target_name.endswith('_') and 'relu' in target_name:
            skip_count += 1
            continue

        # ── Step 1: Find available inputs ──
        available_inputs = find_recomp_inputs(act_node, retained_names, act_names)

        if not available_inputs:
            # No inputs found — this activation has no dependencies (unlikely
            # but possible for trivial ops).  Skip.
            skip_count += 1
            continue

        # ── Step 2: Extract the recomputation subgraph ──
        # Map from original graph nodes to current graph nodes (they may have
        # been renamed by earlier insertions).
        current_inputs = []
        for inp in available_inputs:
            current = name_to_node.get(inp.name)
            if current is None:
                skip_count += 1
                break
            current_inputs.append(current)
        else:
            # All inputs found — proceed with extraction
            current_act = name_to_node.get(act_name)
            if current_act is None:
                skip_count += 1
                continue

            try:
                recomp_subgraph = _extract_graph_with_inputs_outputs(
                    joint_graph=gm.graph,
                    inputs=current_inputs,
                    outputs=[current_act],
                    outputs_descs=[AOTOutput()],
                )
            except Exception as e:
                # Some subgraphs can't be extracted (e.g., in-place ops).
                # Skip these activations gracefully.
                print(f"    [SKIP] {act_name}: subgraph extraction failed ({e})")
                skip_count += 1
                continue

            # ── Step 3: Insert subgraph before first backward use ──
            current_first_bwd = name_to_node.get(first_bwd_node.name)
            if current_first_bwd is None:
                skip_count += 1
                continue

            with gm.graph.inserting_before(current_first_bwd):
                for n in recomp_subgraph.nodes:
                    # Skip placeholder and output nodes — they're structural,
                    # not computation.  Placeholders map to our available_inputs;
                    # the output node is just a return wrapper.
                    if n.op == "placeholder" or n.op == "output":
                        continue

                    # Copy the subgraph node into the main graph.
                    # arg_transform maps subgraph node references to main graph
                    # nodes using name_to_node.
                    new_node = gm.graph.node_copy(
                        n, arg_transform=lambda arg: name_to_node[arg.name]
                    )

                    # If this is the activation we're recomputing, redirect
                    # backward uses from the original to this new node.
                    if n.name == act_name:
                        old_node = name_to_node[act_name]
                        replace_subsequent_uses_of(
                            gm.graph, old_node=old_node, new_node=new_node
                        )

                    # Update name_to_node so subsequent subgraph nodes and
                    # future activations can find this newly inserted node.
                    name_to_node[n.name] = new_node

            rewrite_count += 1
            continue  # explicit continue for the else clause

    # Validate and recompile the modified graph
    gm.graph.lint()
    gm.recompile()

    print(f"  Graph rewriting complete: {rewrite_count} activations recomputed, "
          f"{skip_count} skipped")
    return gm


# ─────────────────────────────────────────────────────────────────────────────
# Correctness verification
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_tensors(obj):
    """Recursively extract all tensors from a nested structure."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        result = []
        for item in obj:
            result.extend(_flatten_tensors(item))
        return result
    return []


def verify_correctness(
    original_gm: fx.GraphModule,
    modified_gm: fx.GraphModule,
    args: list,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """
    Run both the original and modified graphs with the same inputs and
    verify that they produce identical outputs (within tolerance).

    Recursively flattens nested tuple/list outputs to compare all tensors.
    Returns True if all tensor outputs match, False otherwise.
    """
    with torch.no_grad():
        original_out = original_gm(*args)
        modified_out = modified_gm(*args)

    # Flatten nested structures to get all tensors
    orig_tensors = _flatten_tensors(original_out)
    mod_tensors = _flatten_tensors(modified_out)

    if len(orig_tensors) != len(mod_tensors):
        print(f"    Tensor count mismatch: {len(orig_tensors)} vs {len(mod_tensors)}")
        return False

    if len(orig_tensors) == 0:
        print(f"    No tensor outputs to compare")
        return True

    all_match = True
    for i, (orig, mod) in enumerate(zip(orig_tensors, mod_tensors)):
        if orig.shape != mod.shape:
            print(f"    Tensor {i}: shape mismatch ({orig.shape} vs {mod.shape})")
            all_match = False
            continue
        match = torch.allclose(orig, mod, atol=atol, rtol=rtol)
        if not match:
            max_diff = (orig - mod).abs().max().item()
            print(f"    Tensor {i}: MISMATCH (max diff = {max_diff:.6e})")
            all_match = False
        else:
            print(f"    Tensor {i}/{len(orig_tensors)}: OK")

    return all_match
