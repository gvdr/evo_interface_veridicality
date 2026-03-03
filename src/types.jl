"""
Core types for the evolutionary perception simulations.
"""

struct EvolutionParams
    N::Int              # number of world states
    M::Int              # number of percepts
    K::Int              # population size
    T::Int              # tasks per generation
    n_generations::Int  # number of generations
    epsilon::Float64    # per-position mutation rate
    C::Float64          # payoff constant (ensures positive fitness)
    B::Float64          # task bound
end

abstract type TaskSampler end

struct BetaTaskSampler <: TaskSampler
    N::Int
    B::Float64
    a_lo::Float64
    a_hi::Float64
    b_lo::Float64
    b_hi::Float64
end

function BetaTaskSampler(N::Int, B::Float64; a_range=(0.5, 10.0), b_range=(0.5, 10.0))
    return BetaTaskSampler(N, B, a_range[1], a_range[2], b_range[1], b_range[2])
end

struct GaussianTaskSampler{D} <: TaskSampler
    N::Int
    B::Float64
    phi::Matrix{Float64}       # D x N feature matrix (columns = world states)
    Sigma_c::Matrix{Float64}   # D x D covariance of task coefficients
    mvn::D                     # cached MvNormal distribution
end

function GaussianTaskSampler(N::Int, B::Float64; Sigma_c::Matrix{Float64}=Matrix{Float64}(I, N, N))
    phi = Matrix{Float64}(I, N, N)
    D = size(Sigma_c, 1)
    mvn = MvNormal(zeros(D), Sigma_c)
    return GaussianTaskSampler(N, B, phi, Sigma_c, mvn)
end

function GaussianTaskSampler(N::Int, B::Float64, phi::Matrix{Float64}, Sigma_c::Matrix{Float64})
    D = size(Sigma_c, 1)
    mvn = MvNormal(zeros(D), Sigma_c)
    return GaussianTaskSampler(N, B, phi, Sigma_c, mvn)
end

struct WeightedTaskSampler <: TaskSampler
    N::Int
    B::Float64
    category_samplers::Vector{BetaTaskSampler}
    weights::Vector{Float64}
    cumweights::Vector{Float64}
end

function WeightedTaskSampler(N::Int, B::Float64,
                              category_samplers::Vector{BetaTaskSampler},
                              weights::Vector{Float64})
    return WeightedTaskSampler(N, B, category_samplers, weights, cumsum(weights))
end

struct TheoreticalPredictions
    delta_mu::Float64
    fitness_gap_lower::Float64
    convergence_rate_lower::Float64
    kappa::Float64
    effective_rank::Int
    k_T::Int
end
