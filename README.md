# Between Interface and Truth

Multi-task selection drives ecologically veridical perception.

This repository contains the manuscript, simulation code, and data-analysis pipelines for the paper "Between Interface and Truth: Multi-Task Selection Drives Ecologically Veridical Perception" by Giulio V. Dalla Riva.

Preprint: [EcoEvoRxiv](https://ecoevorxiv.org/) (forthcoming).

## Summary

We develop a mathematical theory of agents with a single fixed encoding shared across tasks. The governing object is a separation condition on the task distribution: if a pair of world states is distinguished on tasks with positive measure, optimal encodings must separate it. We prove static optimality, deterministic evolutionary convergence (Price equation + quasispecies recursion), and a graded separation cascade. The framework recovers Hoffman's Fitness-Beats-Truth theorem as the single-task special case.

## Repository Structure

```
evo/
  src/           Julia package (EvoPaper) with core types and functions
  scripts/       Simulation and analysis scripts
  data/          Simulation outputs (gitignored; see data/README.md)
  figures/       Generated figures (gitignored; see figures/README.md)
  latex/         Manuscript source (main.tex, refs.bib, compiled figures)
  docs/          Prose drafts and experiment designs
```

## Requirements

Julia >= 1.10. Dependencies are declared in `Project.toml`:

- Distributions.jl, StatsBase.jl, DataFrames.jl
- CairoMakie.jl, JLD2.jl, ProgressMeter.jl
- LinearAlgebra, Statistics, Random (stdlib)

Install with:

```bash
cd evo
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Simulations

All scripts are run from `evo/`:

```bash
# Main simulations (threaded)
julia -t 16 --project=. scripts/sim1_berke.jl
julia -t 16 --project=. scripts/sim3_nonuniform.jl
julia -t 16 --project=. scripts/sim2_condition.jl

# Analysis and figure generation
julia --project=. scripts/analyze_and_plot_results.jl
julia --project=. scripts/analyze_sim2_summary.jl

# Theorem illustrations and quick checks
julia --project=. scripts/plot_theorem_dynamics.jl
julia --project=. scripts/quick_claim_check.jl
julia --project=. scripts/deterministic_claim_check.jl
julia --project=. scripts/quick_claim_bridge.jl
```

See `data/README.md` for the full provenance matrix and `figures/README.md` for the figure index.

## Compiling the Manuscript

```bash
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

All rights reserved.
