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

### 3.3 In-Place Operation Detection

```python
target_name = str(getattr(act_node, 'target', ''))
if target_name.endswith('_') and 'relu' in target_name:
    skip_count += 1
    continue
```

PyTorch convention: in-place operations end with underscore (e.g., `relu_`, `add_`). In-place ReLU modifies the input tensor's data in place rather than allocating a new output tensor. The subgraph extractor expects outputs to be distinct tensors, so it raises "Node was invalid, but is output" for in-place ops.

ResNet-152 has 64 `relu_` activations out of 78 total recomputed — these are skipped, resulting in 14 actual rewrites. BERT doesn't use in-place ops, so all 115 activations are successfully rewritten.

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

| Model | BS | Recomputed | Successfully Rewritten | Skipped (in-place) |
|-------|---:|-----------:|-----------------------:|-------------------:|
| ResNet-152 | 4 | 78 | 14 | 64 |
| BERT | 4 | 115 | 115 | 0 |

ResNet's high skip rate is due to in-place `relu_` ops. BERT uses out-of-place operations exclusively, allowing full rewriting.

### 4.3 Latency Overhead

| Model | BS | Baseline (ms) | With AC (ms) | Overhead |
|-------|---:|--------------:|-------------:|---------:|
| ResNet-152 | 4 | 57.1 | 59.0 | +3.3% |
| BERT | 4 | 34.5 | 35.5 | +3.0% |

The recomputation overhead is minimal (~3%) because the evicted activations were chosen for their high recompute_ratio (cheap to recompute relative to memory saved). The operations being recomputed (transposes, views, getitems, a few convolutions) are either near-free metadata ops or small convolutions.

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

- **In-place operations cannot be extracted.** `_extract_graph_with_inputs_outputs` requires outputs to be distinct tensors. In-place ops like `relu_` modify their input in place, so there's no separate output tensor to extract. We detect these via PyTorch's naming convention (trailing underscore) and skip them. This limits ResNet-152's rewriting to 14 out of 78 activations. A production implementation could replace `relu_` with out-of-place `relu` in the graph before extraction, but this adds complexity.

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
