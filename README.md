# CS265 Systems Project — Activation Checkpointing

Implementing activation checkpointing in PyTorch using the [μ-TWO algorithm](https://proceedings.mlsys.org/paper_files/paper/2023/file/a72071d84c001596e97a2c7e1e880559-Paper-mlsys2023.pdf) (Purandare et al., MLSys 2023). Models: ResNet-152 and BERT-base.

## Project Structure

```
.
├── graph_tracer.py          # FX graph tracing infrastructure (starter code)
├── graph_prof.py            # Phase 1: computation graph profiler
├── graph_prof_cp.py         # Profiler copy (backup)
├── ac_algorithm.py          # Phase 2: μ-TWO activation checkpointing algorithm
├── ac_visualize.py          # Phase 2: visualization functions
├── graph_rewriter.py        # Phase 3: subgraph extractor and graph rewriter
├── activation_checkpoint.py # Phase 3: starter/example code (simple 2-layer net)
├── run_experiments.py       # Experiment runner (Phases 1 + 2 + 3)
├── utils.py                 # Decomposition tables for tracing
├── starter_code.py          # Original starter code
├── benchmarks.py            # Benchmarking utilities
├── visualize_graph.py       # Graph visualization helper
├── reports/                 # Written reports
│   ├── midway_checkin_report.md / .pdf / .tex
│   ├── phase2_report.md
│   ├── phase3_report.md
│   └── code_explanation.md
├── docs/                    # Reference documents
│   ├── CS_265_Systems_Project_Description.pdf
│   └── report_template.pdf
└── outputs/                 # Generated experiment results (gitignored)
    ├── profiling_stats/     # Per-node profiling data (.txt)
    ├── comp_graphs/         # Full FX graph dumps (.txt)
    ├── ac_decisions/        # Algorithm decisions (.txt)
    └── plots/               # All visualization PNGs
```

## Setup

```bash
conda create -n cs265
conda activate cs265
conda install conda-forge::python=3.12 conda-forge::numpy=2.2.2 \
    pytorch::pytorch=2.5.1 pytorch::pytorch-cuda=12.4 -n cs265
pip install transformers torchvision matplotlib
```

## Running Experiments

```bash
conda activate cs265
python run_experiments.py
```

This runs all three phases (profiling, μ-TWO algorithm, graph rewriting) for ResNet-152 and BERT at batch sizes 2, 4, 8, 16. Results are saved under `outputs/`.

## Phases

| Phase | Weight | Status | Description |
|-------|-------:|--------|-------------|
| 1. Graph Profiler | 35% | Done | Per-node timing, memory, tensor classification, activation lifetimes |
| 2. AC Algorithm | 20% | Done | μ-TWO greedy recomputation: `recompute_ratio = mem / time` |
| 3. Graph Rewriter | 45% | Done | Subgraph extraction, backward-pass insertion, correctness verification, latency measurement |

## References

- [μ-TWO: 3x Faster Multi-Model Training (MLSys 2023)](https://proceedings.mlsys.org/paper_files/paper/2023/file/a72071d84c001596e97a2c7e1e880559-Paper-mlsys2023.pdf)
- [torch.fx documentation](https://pytorch.org/docs/2.5/fx.html)
- [CS265 course page](http://daslab.seas.harvard.edu/classes/cs265)
