using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, DataFrames, Statistics, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "EvoPaper.jl"))
using .EvoPaper

N = 11
B = 1.0
rng = MersenneTwister(42)

# Build families same as sim2
function make_gaussian_sigma(N::Int, kappa_target::Float64)
    if kappa_target <= 1.0
        return Matrix{Float64}(I, N, N)
    end
    r = kappa_target^(1.0 / (N - 1))
    diag_vals = [r^(-(i - 1)) for i in 1:N]
    return diagm(diag_vals)
end

families = Dict{String, TaskSampler}()
families["gauss_iso"] = GaussianTaskSampler(N, B; Sigma_c=Matrix{Float64}(I, N, N))
for kappa in [10.0, 100.0, 1000.0]
    label = "gauss_k" * string(Int(kappa))
    Sigma_c = make_gaussian_sigma(N, kappa)
    families[label] = GaussianTaskSampler(N, B; Sigma_c=Sigma_c)
end
families["beta_narrow"] = BetaTaskSampler(N, B; a_range=(1.0, 3.0), b_range=(1.0, 3.0))
families["beta_wide"] = BetaTaskSampler(N, B; a_range=(0.1, 20.0), b_range=(0.1, 20.0))

# For each family, sample a large number of tasks and compute the population sigma^2 matrix
println("===== Population sigma^2 matrix (T=10000 sample) =====")
println()

for fam in ["gauss_iso", "gauss_k10", "gauss_k100", "gauss_k1000", "beta_narrow", "beta_wide"]
    sampler = families[fam]
    F = build_task_matrix(rng, sampler, 10000)

    # Compute sigma^2 matrix
    sigma2 = zeros(N, N)
    for i in 1:N, j in 1:N
        diffs = F[:, i] .- F[:, j]
        sigma2[i, j] = mean(diffs .^ 2)
    end

    # Extract upper triangle
    dists = Float64[]
    for i in 1:N
        for j in (i+1):N
            push!(dists, sigma2[i, j])
        end
    end
    sort!(dists)

    println("Family: " * fam)
    println("  n_pairs = " * string(length(dists)))
    println("  min (= population delta_mu) = " * string(round(minimum(dists), digits=6)))
    println("  5th percentile = " * string(round(dists[max(1, Int(ceil(0.05 * length(dists))))], digits=6)))
    println("  median = " * string(round(median(dists), digits=6)))
    println("  mean   = " * string(round(mean(dists), digits=6)))
    println("  max    = " * string(round(maximum(dists), digits=6)))
    println("  ratio max/min = " * string(round(maximum(dists) / max(minimum(dists), 1e-12), digits=2)))
    println("  std    = " * string(round(std(dists), digits=6)))
    println("  coeff of variation = " * string(round(std(dists) / mean(dists), digits=3)))
    println("  all 55 pairwise: " * join([string(round(d, digits=4)) for d in dists], ", "))
    println()
end

# Also: what does the theory predict for Gaussian families?
# For GaussianTaskSampler with phi=I, Sigma_c diagonal, clamping to [-B, B]:
# f(w) ~ clamp(N(0, Sigma_c[w,w]), -B, B)
# sigma^2(i,j) = Var(f(i)) + Var(f(j)) since f(i), f(j) are independent
# (Cov(f(i), f(j)) = 0 since phi=I and Sigma_c is diagonal)
println("===== Analytical check: diagonal variances of clamped Gaussian =====")
println("(Sigma_c[w,w] for each world state w)")
println()

for fam in ["gauss_iso", "gauss_k10", "gauss_k1000"]
    sampler = families[fam]
    Sc = sampler.Sigma_c

    # Sample variances per world state
    n_samp = 100000
    vars = zeros(N)
    for w in 1:N
        samples = clamp.(sqrt(Sc[w, w]) .* randn(rng, n_samp), -B, B)
        vars[w] = var(samples)
    end

    println("Family: " * fam)
    println("  Sigma_c diagonal: " * join([string(round(Sc[w, w], digits=6)) for w in 1:N], ", "))
    println("  Clamped variances: " * join([string(round(vars[w], digits=6)) for w in 1:N], ", "))
    println("  Min var = " * string(round(minimum(vars), digits=6)) *
            "  Max var = " * string(round(maximum(vars), digits=6)))
    println("  Predicted delta_mu = min_{i<j}(var_i + var_j) = " *
            string(round(minimum([vars[i] + vars[j] for i in 1:N for j in (i+1):N]), digits=6)))
    println()
end
