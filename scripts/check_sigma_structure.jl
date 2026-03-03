using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, DataFrames, Statistics

# We need the task generation code to inspect sigma^2 matrices
include(joinpath(@__DIR__, "..", "src", "EvoPaper.jl"))
using .EvoPaper

chunk_dir = joinpath(@__DIR__, "..", "data", "sim2_condition_chunks")

println("===== Task distance matrix structure by family (T=50, M=11) =====")
println()

for fam in ["gauss_iso", "gauss_k10", "gauss_k100", "gauss_k1000", "beta_narrow", "beta_wide"]
    # Load one replicate to get the per-generation delta_mu trajectory
    fname = fam * "__M11__T50__rep1.jld2"
    fpath = joinpath(chunk_dir, fname)
    isfile(fpath) || continue
    df = load(fpath, "result")

    # Get mean delta_mu and pred_kappa
    mean_dm = mean(df.delta_mu)
    kappa = df.pred_kappa[1]  # constant across generations

    println("Family: " * fam)
    println("  pred_kappa = " * string(round(kappa, digits=2)))
    println("  mean delta_mu (min sigma^2) = " * string(round(mean_dm, digits=6)))
    println("  delta_mu range over gens: " * string(round(minimum(df.delta_mu), digits=6)) *
            " to " * string(round(maximum(df.delta_mu), digits=6)))
    println("  final eco_verid = " * string(round(df.frac_eco_veridical[end], digits=3)))
    println("  final full_verid = " * string(round(df.frac_full_veridical[end], digits=3)))
    println("  final risk = " * string(round(df.mean_risk[end], digits=5)))
    println()
end

# Now let's look at the FULL sigma^2 matrix for each family
# We need to reconstruct it from the task sampler
println("===== Full sigma^2 matrix statistics (single draw, T=50) =====")
println()

using Random
rng = Random.MersenneTwister(42)
N = 11
T_val = 50

for fam in ["gauss_iso", "gauss_k10", "gauss_k100", "gauss_k1000", "beta_narrow", "beta_wide"]
    sampler = EvoPaper.make_task_sampler(fam, N)
    F = EvoPaper.build_task_matrix(rng, sampler, T_val)

    # Compute full sigma^2 matrix
    sigma2 = zeros(N, N)
    for i in 1:N, j in 1:N
        sigma2[i, j] = mean((F[:, i] .- F[:, j]).^2)
    end

    # Extract upper triangle (pairwise distances)
    dists = Float64[]
    for i in 1:N
        for j in (i+1):N
            push!(dists, sigma2[i, j])
        end
    end
    sort!(dists)

    println("Family: " * fam)
    println("  N*(N-1)/2 = " * string(length(dists)) * " pairwise distances")
    println("  min    = " * string(round(minimum(dists), digits=6)))
    println("  median = " * string(round(median(dists), digits=6)))
    println("  mean   = " * string(round(mean(dists), digits=6)))
    println("  max    = " * string(round(maximum(dists), digits=6)))
    println("  ratio max/min = " * string(round(maximum(dists) / maximum(minimum(dists), 1e-12), digits=2)))
    println("  5 smallest: " * join([string(round(d, digits=6)) for d in dists[1:min(5, end)]], ", "))
    println("  5 largest:  " * join([string(round(d, digits=6)) for d in dists[max(1, end-4):end]], ", "))

    # What fraction of pairs have sigma^2 < various thresholds?
    for thresh in [0.001, 0.01, 0.1]
        frac = count(d -> d < thresh, dists) / length(dists)
        println("  frac with sigma^2 < " * string(thresh) * ": " * string(round(frac, digits=3)))
    end
    println()
end
