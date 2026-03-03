module EvoPaper

using Random: AbstractRNG, MersenneTwister
using Statistics: mean
using LinearAlgebra: I, Symmetric, eigvals
using Distributions: Beta, MvNormal, pdf
using StatsBase: Weights, sample
using DataFrames: DataFrame
using ProgressMeter: Progress, next!

# Source files in dependency order
include("types.jl")
include("encoding.jl")
include("risk.jl")
include("tasks.jl")
include("theory.jl")
include("price.jl")
include("evolution.jl")

# Types
export EvolutionParams, TaskSampler, BetaTaskSampler, GaussianTaskSampler,
       WeightedTaskSampler, TheoreticalPredictions

# Encoding
export random_encoding, random_encoding!, mutate, mutate_into!,
       is_injective, used_complexity,
       is_ecologically_veridical, is_ecologically_veridical_cells

# Risk
export bayes_risk, multi_task_risk, multi_task_risk_cells,
       precompute_cells, compute_fitness

# Tasks
export sample_task, build_task_matrix

# Theory
export task_distance_matrix, separation_margin, equivalence_classes,
       condition_number, theoretical_predictions, optimal_partition_swap

# Price
export price_equation_check, fishers_theorem_check

# Evolution
export run_evolution

end # module
