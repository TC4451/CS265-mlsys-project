# Phase 3: Graph Extractor and Rewriter
CS265, Spring 2025
Zilin Dai

## 1. Introduction

In Phase 2, the μ-TWO algorithm decided which activations to discard and recompute. Phase 3 makes those decisions real by modifying the actual FX computation graph. For each activation marked for recomputation, we extract the forward subgraph that computes it, insert a copy of that subgraph into the backward pass just before the activation is needed, and redirect backward consumers to use the recomputed value instead of the original. After rewriting, the graph produces identical gradients but uses less peak memory, because recomputed activations are freed after their last forward use rather than persisting into the backward pass.

## 2. Problems Tackled

- **[Finding Available Inputs]** To recompute an activation in the backward pass, we need to know which values will still be in memory at that point. Parameters and placeholders are always available. Retained (checkpointed) activations are still in memory. But other recomputed activations won't be — we need to trace through them to find their own available sources, potentially recomputing a chain of operations.

- **[Subgraph Extraction]** Given the available inputs and the target activation, we need to extract the minimal subgraph that computes the activation from those inputs. PyTorch's `_extract_graph_with_inputs_outputs()` does this, but it requires correct input/output specification and cannot handle in-place operations as outputs.

- **[In-Place Operation Handling]** ResNet-152 uses in-place ReLU (`relu_`), which mutates its input tensor rather than producing a new output. These operations cannot be extracted as subgraph outputs by `_extract_graph_with_inputs_outputs()` because there is no distinct output tensor. We must detect and skip these.

- **[Graph Insertion and Use Replacement]** After extracting the subgraph, we insert its nodes into the main graph before the first backward consumer. We then redirect only backward uses to the recomputed node, leaving forward uses unchanged. This requires careful tracking of which uses come "after" the insertion point.

- **[Correctness Verification]** The rewritten graph must produce identical gradients. We verify this by running both the original and modified graphs with the same inputs and comparing all 467 output tensors (for ResNet-152) element-by-element.

## 3. Technical Description

### 3.1 Finding Available Inputs (`find_recomp_inputs`)

For each recomputed activation, we walk backward through the FX graph starting from its direct inputs:

```
function find_recomp_inputs(act_node, retained_names, act_names):
    inputs = []
    stack = act_node.all_input_nodes
    while stack not empty:
        node = stack.pop()
        if node is a recomputed activation:
            # Not in memory — must trace through it
            stack.extend(node.all_input_nodes)
        else:
            # Available: placeholder, param, retained act, or non-act forward op
            inputs.append(node)
    return deduplicate(inputs)
```

A node is "available" (will be in memory during backward) if it is NOT a recomputed activation. This includes:
- **Placeholders**: Graph inputs (model weights, optimizer states, batch data) — always in memory.
- **Parameters**: Model weights — always in memory.
- **Retained activations**: Checkpointed by Phase 2's decision — kept in memory through backward.
- **Non-activation forward ops**: Ops that are in the forward region but don't qualify as activations (e.g., constants, shapes). These are available because they were never evicted.

When we encounter a recomputed activation on the path, we don't stop — instead we trace through it, because it won't be in memory. This naturally builds the transitive closure: if A depends on B depends on C, and B is recomputed but C is retained, the available inputs for A will include C (not B).

### 3.2 Subgraph Extraction

```python
recomp_subgraph = _extract_graph_with_inputs_outputs(
    joint_graph=gm.graph,
    inputs=current_inputs,
    outputs=[current_act],
    outputs_descs=[AOTOutput()],
)
```

`_extract_graph_with_inputs_outputs` (from `torch._functorch.partitioners`) finds all nodes on any path from `inputs` to `outputs` in the graph and returns a new `fx.Graph` containing only those nodes. The inputs become placeholders in the subgraph; the outputs become the return value.

The `outputs_descs` parameter (required by this PyTorch version) describes the type of each output. `AOTOutput()` is a default descriptor with no special flags.

### 3.3 In-Place to Out-of-Place Conversion

**The problem.** ResNet-152 uses in-place ReLU (`torch.ops.aten.relu_.default`). In-place ops mutate their input tensor rather than allocating a new output. `_extract_graph_with_inputs_outputs()` cannot extract in-place ops as subgraph outputs because there is no distinct output tensor — it raises "Node was invalid, but is output."

In the initial implementation, we detected and skipped in-place ops, which meant 64 out of 78 ResNet activations could not be rewritten — only 14 were actually checkpointed.

**The fix.** Before extraction, we run a preprocessing pass that replaces every in-place op with its out-of-place equivalent:

```python
INPLACE_TO_OUTOFPLACE = {
    torch.ops.aten.relu_.default: torch.ops.aten.relu.default,
}

def convert_inplace_to_outofplace(gm):
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in INPLACE_TO_OUTOFPLACE:
            node.target = INPLACE_TO_OUTOFPLACE[node.target]
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
    return count
```

This is a single pass over the graph that changes the `target` attribute of each node from `aten.relu_` to `aten.relu`. The semantics are identical (both compute `max(0, x)`) but the out-of-place version allocates a new tensor, giving the extractor a clean subgraph boundary.

**Design choice: why convert instead of skip?** Converting lets us rewrite all 74 activations (100%) instead of just 14 (18%). The trade-off is that out-of-place ops use slightly more memory during forward (each ReLU now allocates a new tensor). But this is exactly the memory that activation checkpointing frees — the new tensor is freed after its last forward use when the activation is marked for recomputation. The net effect is strictly better.

**Result after fix.** For ResNet-152 bs=4: 151 in-place ops converted, 74/74 activations rewritten (0 skipped), correctness PASS.

### 3.4 Graph Insertion

```python
with gm.graph.inserting_before(current_first_bwd):
    for n in recomp_subgraph.nodes:
        if n.op == "placeholder" or n.op == "output":
            continue
        new_node = gm.graph.node_copy(
            n, arg_transform=lambda arg: name_to_node[arg.name]
        )
        if n.name == act_name:
            replace_subsequent_uses_of(gm.graph, old_node=old_node, new_node=new_node)
        name_to_node[n.name] = new_node
```

Step by step:

1. **`inserting_before(first_bwd_node)`**: All new nodes will be placed just before the first backward consumer of the activation. This is the latest safe point — any earlier and we'd be recomputing before the value is needed, wasting memory.

2. **Skip placeholders and output**: These are structural nodes in the subgraph. Placeholders correspond to our available inputs (already in the main graph); the output node is just a return wrapper.

3. **`node_copy(n, arg_transform)`**: Copies a subgraph node into the main graph. The `arg_transform` function maps subgraph node references to main graph nodes via `name_to_node`. For example, if the subgraph has a node `relu(convolution_4)`, it maps `convolution_4` to the actual `convolution_4` node in the main graph.

4. **`replace_subsequent_uses_of`**: Only replaces uses of the old activation that appear AFTER the new recomputed node in the graph. This is critical — forward-pass uses must keep pointing to the original node (they need the value during forward execution). Only backward-pass uses switch to the recomputed version.

5. **`name_to_node[n.name] = new_node`**: Updates the mapping so that subsequent subgraph nodes (within the same extraction) and future activations' extractions can find this newly inserted node.

### 3.5 Use Replacement (`replace_subsequent_uses_of`)

```python
def replace_subsequent_uses_of(graph, old_node, new_node):
    old_node_users = dict(old_node.users)  # snapshot
    for node in reversed(list(graph.nodes)):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)
```

Walks the graph in reverse from the end. For each node that uses `old_node`, if it appears after `new_node`, its input is swapped. Stops when it reaches `new_node` itself, ensuring forward-pass uses are untouched.

**Why snapshot `old_node.users`?** The `users` dict changes as we call `replace_input_with`. Without the snapshot, we'd be iterating over a mutating dict, which causes errors.

### 3.6 Processing Order

Recomputed activations are sorted by `first_bwd_use` (earliest first). This ensures:
- Activations needed first in backward are inserted first.
- Earlier insertions don't shift the positions of later insertion points, because `inserting_before(node)` inserts relative to a specific node, not an absolute position.

## 4. Experimental Results

### 4.1 Correctness Verification

| Model | BS | Tensors Compared | Result |
|-------|---:|-----------------:|--------|
| ResNet-152 | 4 | 467 | PASS |
| BERT | 4 | 467+ | PASS |

All output tensors (gradients, updated parameters, updated optimizer states) match between the original and rewritten graphs within tolerance (`atol=1e-5, rtol=1e-4`).

### 4.2 Rewriting Statistics

| Model | BS | Recomputed | Successfully Rewritten | In-place Converted | Skipped |
|-------|---:|-----------:|-----------------------:|-------------------:|--------:|
| ResNet-152 | 4 | 74 | 74 (100%) | 151 relu_ ops | 0 |
| BERT | 4 | 134 | 134 (100%) | 0 | 0 |

After the in-place to out-of-place conversion, all recomputed activations are successfully rewritten for both models.

### 4.3 Latency Overhead

| Model | BS | Baseline (ms) | With AC (ms) | Overhead |
|-------|---:|--------------:|-------------:|---------:|
| ResNet-152 | 4 | 56.2 | 58.6 | +4.1% |
| BERT | 4 | 34.1 | 35.1 | +2.8% |

The recomputation overhead is small (~3-4%) because the evicted activations were chosen for their high recompute_ratio (cheap to recompute relative to memory saved). The operations being recomputed (ReLUs, transposes, views, getitems, a few convolutions) are either near-free metadata ops or fast elementwise ops.

### 4.4 Full Pipeline Summary

| Model | BS | Baseline Peak | AC Peak | Memory Saved | Latency Overhead |
|-------|---:|-------------:|--------:|-------------:|-----------------:|
| ResNet-152 | 2 | 1,089 MB | 945 MB | 13.2% | — |
| ResNet-152 | 4 | 1,423 MB | 1,073 MB | 24.6% | +3.3% |
| ResNet-152 | 8 | 2,092 MB | 1,580 MB | 24.5% | — |
| ResNet-152 | 16 | 3,429 MB | 2,596 MB | 24.3% | — |
| BERT | 2 | 1,770 MB | 1,665 MB | 6.0% | — |
| BERT | 4 | 1,967 MB | 1,664 MB | 15.4% | +3.0% |
| BERT | 8 | 2,361 MB | 1,794 MB | 24.0% | — |
| BERT | 16 | 3,156 MB | 2,376 MB | 24.7% | — |

## 5. Challenges

- **In-place operations cannot be extracted directly.** `_extract_graph_with_inputs_outputs` requires outputs to be distinct tensors. In-place ops like `relu_` modify their input in place, so there's no separate output tensor. Our initial implementation detected and skipped these, limiting ResNet-152 to 14/78 rewrites. We fixed this by adding a preprocessing pass that converts `relu_` to out-of-place `relu` (`torch.ops.aten.relu_.default` → `torch.ops.aten.relu.default`). This is a one-line target swap per node — the semantics are identical but the out-of-place version allocates a new tensor, giving the extractor a clean boundary. After the fix: 74/74 rewrites for ResNet-152 (100%).

- **Subgraph extraction with `AOTOutput` descriptor.** The PyTorch 2.5 version of `_extract_graph_with_inputs_outputs` requires an `outputs_descs` argument that wasn't in earlier versions. We pass `[AOTOutput()]` (a default descriptor with no flags) for each output. This was discovered by inspecting the function signature at runtime.

- **Name collisions after insertion.** When we copy a subgraph node into the main graph, FX may rename it to avoid conflicts (e.g., `relu` becomes `relu_2`). The `name_to_node` dict is updated after each insertion to track the latest node for each name, ensuring subsequent subgraph copies can resolve their input references correctly.

- **Correctness of use replacement.** `replace_subsequent_uses_of` must only replace backward uses, not forward uses. If we replaced all uses, the forward pass would try to use the recomputed value (which doesn't exist yet during forward execution) and fail. The reverse-walk-until-new-node approach cleanly separates forward from backward uses.

## 6. Implementation Details

### New File: `graph_rewriter.py`

| Function | Lines | Description |
|----------|------:|-------------|
| `replace_subsequent_uses_of()` | 29–44 | Replace only backward uses of a node |
| `find_recomp_inputs()` | 51–100 | Backward walk to find available inputs |
| `apply_ac_to_graph()` | 107–200 | Main rewriting loop: extract, insert, replace |
| `_flatten_tensors()` | 205–212 | Recursively extract tensors from nested outputs |
| `verify_correctness()` | 215–252 | Compare original vs modified graph outputs |

### Updated File: `run_experiments.py`

- `profile()` now returns `(profiler, gm, flat_args)` instead of just `profiler`
- `run_phase1()` stores `graph_data = {(model, bs): (gm, args)}`
- New `run_phase3()`: applies rewriting, verifies correctness, measures latency

### How to Reproduce

```bash
conda activate cs265
python run_experiments.py
```

Phase 3 runs automatically after Phases 1 and 2. Graph rewriting and verification are performed on default batch sizes (bs=4) for both models.
