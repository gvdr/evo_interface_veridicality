"""
Generate three key figures showing how the task distance structure
(σ² matrix) drives veridicality outcomes.

  (A) Distribution of all 55 pairwise σ²(i,j) values per family
  (B) Final veridicality vs population δ_μ
  (C) Veridicality-fraction trajectories grouped by family/δ_μ

Run:
  julia --project=. scripts/plot_distance_structure.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using DataFrames
using JLD2
using Statistics
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "EvoPaper.jl"))
using .EvoPaper

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const FIG_DIR  = joinpath(@__DIR__, "..", "figures", "sim2")
const CHUNK_DIR = joinpath(DATA_DIR, "sim2_condition_chunks")

mkpath(FIG_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function make_gaussian_sigma(N::Int, kappa_target::Float64)
    kappa_target <= 1.0 && return Matrix{Float64}(I, N, N)
    r = kappa_target^(1.0 / (N - 1))
    diagm([r^(-(i - 1)) for i in 1:N])
end

function build_families(N::Int, B::Float64)
    fams = Dict{String, TaskSampler}()
    fams["gauss_iso"]   = GaussianTaskSampler(N, B; Sigma_c=Matrix{Float64}(I, N, N))
    for kappa in [10.0, 100.0, 1000.0]
        fams["gauss_k" * string(Int(kappa))] =
            GaussianTaskSampler(N, B; Sigma_c=make_gaussian_sigma(N, kappa))
    end
    fams["beta_narrow"] = BetaTaskSampler(N, B; a_range=(1.0, 3.0), b_range=(1.0, 3.0))
    fams["beta_wide"]   = BetaTaskSampler(N, B; a_range=(0.1, 20.0), b_range=(0.1, 20.0))
    fams
end

"""Compute population σ² matrix via Monte Carlo with T_mc tasks."""
function population_sigma2(rng, sampler::TaskSampler, N::Int, T_mc::Int)
    F = build_task_matrix(rng, sampler, T_mc)
    sigma2 = zeros(N, N)
    for i in 1:N, j in 1:N
        sigma2[i, j] = mean((F[:, i] .- F[:, j]) .^ 2)
    end
    sigma2
end

"""Extract upper-triangle pairwise distances from σ² matrix."""
function pairwise_dists(sigma2)
    N = size(sigma2, 1)
    dists = Float64[]
    for i in 1:N, j in (i+1):N
        push!(dists, sigma2[i, j])
    end
    sort!(dists)
    dists
end

# ---------------------------------------------------------------------------
# Figure A – Distribution of pairwise σ² by family
# ---------------------------------------------------------------------------

function figure_a(families, rng, N)
    # Order families by decreasing population δ_μ
    fam_order = ["gauss_iso", "gauss_k10", "beta_wide",
                 "gauss_k100", "beta_narrow", "gauss_k1000"]
    labels    = [rich("gauss iso\n", rich("(isotropic)", fontsize=10)),
                 rich("gauss ", rich("k", fontsize=10), "=10"),
                 "beta wide",
                 rich("gauss ", rich("k", fontsize=10), "=100"),
                 "beta narrow",
                 rich("gauss ", rich("k", fontsize=10), "=1000")]

    fig = Figure(size=(820, 440), fontsize=14)
    ax  = Axis(fig[1, 1],
        ylabel=L"Pairwise task distance $\sigma^2(w_i, w_j)$",
        yscale=log10,
        xticks=(1:6, ["gauss iso", "gauss k=10", "beta wide",
                       "gauss k=100", "beta narrow", "gauss k=1000"]),
        xticklabelrotation=0.3)

    palette = [:dodgerblue, :mediumseagreen, :goldenrod,
               :darkorange, :firebrick, :mediumpurple]

    delta_mus = Float64[]

    for (k, fam) in enumerate(fam_order)
        sigma2 = population_sigma2(rng, families[fam], N, 10_000)
        dists  = pairwise_dists(sigma2)
        dm     = minimum(dists)
        push!(delta_mus, dm)

        # Jitter x positions centred on k
        jitter = 0.25 .* (rand(rng, length(dists)) .- 0.5)
        xs = fill(Float64(k), length(dists)) .+ jitter
        scatter!(ax, xs, dists, color=(palette[k], 0.6), markersize=7)

        # Mark δ_μ as a short horizontal segment centred on k
        hw = 0.18  # half-width matches jitter spread
        linesegments!(ax, [Point2f(k - hw, dm), Point2f(k + hw, dm)],
                       color=palette[k], linewidth=2.5, linestyle=:dash)
    end

    # Mutation-selection threshold: s = π²_min δ_μ / (2C), need s > ε
    # => δ_μ > 2Cε / π²_min = 2*2*0.001 / (1/121) = 0.484
    hlines!(ax, [0.484], color=:black, linewidth=1.5, linestyle=:dot,
            label=L"mutation-selection threshold $\delta_\mu^*$")

    ylims!(ax, 1e-3, 2.0)

    # Add a dummy entry for the dashed δ_μ lines in the legend
    linesegments!(ax, [Point2f(NaN, NaN), Point2f(NaN, NaN)],
                  color=:gray50, linewidth=2.5, linestyle=:dash,
                  label=L"$\delta_\mu$ (bottleneck pair)")
    axislegend(ax, position=:lb, framevisible=false, labelsize=14)

    save(joinpath(FIG_DIR, "sigma2_distributions.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sigma2_distributions.png"), fig; px_per_unit=3)
    println("Saved sigma2_distributions")
    delta_mus, fam_order
end

# ---------------------------------------------------------------------------
# Figure B – Final veridicality vs population δ_μ
# ---------------------------------------------------------------------------

function figure_b(delta_mus_map::Dict{String, Float64})
    summary_path = joinpath(DATA_DIR, "sim2_condition_summary.jld2")
    s = load(summary_path, "summary")

    # Restrict to M=11
    s11 = s[Int.(s.M) .== 11, :]

    fig = Figure(size=(750, 480), fontsize=15)

    # Panel (a): full-veridical fraction
    ax1 = Axis(fig[1, 1],
        xlabel=L"Population $\delta_\mu$ (log scale)",
        ylabel="Full-veridical fraction",
        xscale=log10)

    # Panel (b): mean Bayes risk
    ax2 = Axis(fig[1, 2],
        xlabel=L"Population $\delta_\mu$ (log scale)",
        ylabel="Mean Bayes risk",
        xscale=log10)

    palette = Dict(
        "gauss_iso"   => (:dodgerblue,    :circle),
        "gauss_k10"   => (:mediumseagreen, :utriangle),
        "gauss_k100"  => (:darkorange,    :diamond),
        "gauss_k1000" => (:mediumpurple,  :rect),
        "beta_narrow" => (:firebrick,     :pentagon),
        "beta_wide"   => (:goldenrod,     :cross),
    )

    for fam in sort(collect(keys(delta_mus_map)))
        dm = delta_mus_map[fam]
        sub = s11[String.(s11.family) .== fam, :]
        col, marker = palette[fam]

        # One point per replicate (jitter slightly in x for visibility)
        n = nrow(sub)
        jx = dm .* exp.(0.03 .* randn(n))
        scatter!(ax1, jx, Float64.(sub.final_frac_full_veridical),
                 color=(col, 0.35), marker=marker, markersize=6)
        scatter!(ax2, jx, Float64.(sub.final_mean_risk),
                 color=(col, 0.35), marker=marker, markersize=6)

        # Family mean
        mv = mean(Float64.(sub.final_frac_full_veridical))
        mr = mean(Float64.(sub.final_mean_risk))
        scatter!(ax1, [dm], [mv], color=col, marker=marker, markersize=14,
                 strokewidth=1.5, strokecolor=:black,
                 label=replace(fam, "_" => " "))
        scatter!(ax2, [dm], [mr], color=col, marker=marker, markersize=14,
                 strokewidth=1.5, strokecolor=:black)
    end

    axislegend(ax1, position=:lt, framevisible=false, labelsize=11)

    save(joinpath(FIG_DIR, "veridicality_vs_delta_mu.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "veridicality_vs_delta_mu.png"), fig; px_per_unit=3)
    println("Saved veridicality_vs_delta_mu")
end

# ---------------------------------------------------------------------------
# Figure C – Veridicality-fraction trajectories by family
# ---------------------------------------------------------------------------

function figure_c(delta_mus_map::Dict{String, Float64})
    isdir(CHUNK_DIR) || (println("No chunk dir, skipping figure C"); return)

    # Order by decreasing δ_μ
    fam_order = sort(collect(keys(delta_mus_map)), by=k -> -delta_mus_map[k])

    palette = Dict(
        "gauss_iso"   => :dodgerblue,
        "gauss_k10"   => :mediumseagreen,
        "gauss_k100"  => :darkorange,
        "gauss_k1000" => :mediumpurple,
        "beta_narrow" => :firebrick,
        "beta_wide"   => :goldenrod,
    )

    fig = Figure(size=(1100, 520), fontsize=15)

    # Panel (a): full-veridical fraction trajectory
    ax1 = Axis(fig[1, 1],
        xlabel="Generation",
        ylabel="Full-veridical fraction")

    # Panel (b): mean Bayes risk trajectory
    ax2 = Axis(fig[1, 2],
        xlabel="Generation",
        ylabel="Mean Bayes risk")

    n_reps = 30
    # Use T=50 as representative (middle of range)
    T_repr = 50

    for fam in fam_order
        all_verid = Vector{Vector{Float64}}()
        all_risk  = Vector{Vector{Float64}}()
        for rep in 1:n_reps
            fname = fam * "__M11__T" * string(T_repr) * "__rep" * string(rep) * ".jld2"
            fpath = joinpath(CHUNK_DIR, fname)
            isfile(fpath) || continue
            df = load(fpath, "result")
            push!(all_verid, Float64.(df.frac_full_veridical))
            push!(all_risk,  Float64.(df.mean_risk))
        end
        length(all_verid) == 0 && continue

        n_gen = length(all_verid[1])
        mean_v = zeros(n_gen)
        mean_r = zeros(n_gen)
        for g in 1:n_gen
            mean_v[g] = mean(v[g] for v in all_verid)
            mean_r[g] = mean(r[g] for r in all_risk)
        end

        idx = 1:10:n_gen
        dm_str = string(round(delta_mus_map[fam], digits=3))
        lab_simple = replace(fam, "_" => " ") * " (dm=" * dm_str * ")"

        col = palette[fam]
        lines!(ax1, collect(1:n_gen)[idx], mean_v[idx],
               color=col, linewidth=2.0, label=lab_simple)
        lines!(ax2, collect(1:n_gen)[idx], mean_r[idx],
               color=col, linewidth=2.0)
    end

    Legend(fig[2, 1:2], ax1, orientation=:horizontal, framevisible=false,
           labelsize=12, nbanks=1)

    save(joinpath(FIG_DIR, "veridicality_trajectories_by_family.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "veridicality_trajectories_by_family.png"), fig; px_per_unit=3)
    println("Saved veridicality_trajectories_by_family")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    N = 11
    B = 1.0
    rng = MersenneTwister(2024)

    families = build_families(N, B)

    # Figure A: distance distributions
    delta_mus, fam_order = figure_a(families, rng, N)
    delta_mus_map = Dict(fam_order[k] => delta_mus[k] for k in eachindex(fam_order))

    # Figure B: veridicality vs δ_μ
    figure_b(delta_mus_map)

    # Figure C: trajectories
    figure_c(delta_mus_map)

    # Print summary table
    println()
    println("===== Population δ_μ by family =====")
    for fam in fam_order
        println("  " * rpad(fam, 15) * "δ_μ = " * string(round(delta_mus_map[fam], digits=6)))
    end
end

main()
