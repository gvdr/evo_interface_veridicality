"""
Plot theorem-focused toy dynamics not covered by sim1/sim2/sim3:
1) Accessible-vs-trapped optimum under constrained/reducible mutation.
2) Quotient communicating-class mass dynamics.
3) Periodic dominant block with Cesaro stabilization.

Run with:
  julia --project=. scripts/plot_theorem_dynamics.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra
using Statistics

function normalize_prob(x::Vector{Float64})
    s = sum(x)
    s > 0 || error("Cannot normalize zero vector.")
    return x ./ s
end

function iterate_quasispecies(A::Matrix{Float64}, x0::Vector{Float64}, nsteps::Int)
    x = normalize_prob(copy(x0))
    history = Vector{Vector{Float64}}(undef, nsteps + 1)
    history[1] = copy(x)
    for g in 1:nsteps
        y = A * x
        x = normalize_prob(y)
        history[g + 1] = copy(x)
    end
    return history
end

function principal_right_eigenvector(A::Matrix{Float64})
    vals, vecs = eigen(A)
    idx = argmax(real.(vals))
    v = abs.(real.(vecs[:, idx]))
    return normalize_prob(v)
end

function block_stationary_risk(A_block::Matrix{Float64}, R_block::Vector{Float64})
    v = principal_right_eigenvector(A_block)
    return dot(v, R_block)
end

function main()
    outdir = joinpath(@__DIR__, "..", "figures", "theorems")
    mkpath(outdir)

    # Toy constrained-mutation system with two closed classes and one transient state.
    # K1 = {1,2} is dominant (higher spectral radius), K2 = {3,4} is subdominant, 5 transient.
    W = [1.4, 1.3, 1.05, 1.0, 1.1]
    C = 2.0
    T = 1.0
    R = C .- W ./ T
    Q = [
        0.90 0.10 0.00 0.00 0.00
        0.20 0.80 0.00 0.00 0.00
        0.00 0.00 0.85 0.15 0.00
        0.00 0.00 0.25 0.75 0.00
        0.70 0.00 0.30 0.00 0.00
    ]
    A = Q' * Diagonal(W)

    nsteps = 500
    gens = 0:nsteps

    x0_access = [0.0, 0.0, 0.0, 0.0, 1.0]   # can flow to both classes
    x0_trapped = [0.0, 0.0, 0.6, 0.4, 0.0]  # trapped in suboptimal class K2

    hist_access = iterate_quasispecies(A, x0_access, nsteps)
    hist_trapped = iterate_quasispecies(A, x0_trapped, nsteps)

    risk_access = [dot(x, R) for x in hist_access]
    risk_trapped = [dot(x, R) for x in hist_trapped]

    R_star_K1 = block_stationary_risk(A[1:2, 1:2], R[1:2])
    R_star_K2 = block_stationary_risk(A[3:4, 3:4], R[3:4])

    mK1 = [sum(x[1:2]) for x in hist_access]
    mK2 = [sum(x[3:4]) for x in hist_access]
    mTr = [x[5] for x in hist_access]

    # Periodic dominant block example
    Wp = [1.4, 1.0]
    Qp = [0.0 1.0; 1.0 0.0]  # irreducible, period 2
    Ap = Qp' * Diagonal(Wp)
    hist_p = iterate_quasispecies(Ap, [0.61, 0.39], nsteps)
    x1 = [x[1] for x in hist_p]
    x1_cesaro = [mean(x1[1:g]) for g in 1:length(x1)]
    xpf1 = principal_right_eigenvector(Ap)[1]

    # Use generation+1 for log scale (avoid log(0))
    gens_log = collect(gens) .+ 1

    # Individual figures (with legends)

    fig1 = Figure(size=(900, 520), fontsize=16)
    ax1 = Axis(fig1[1, 1],
        title="Constrained Mutation: Accessible vs Trapped Asymptote",
        xlabel="Generation", ylabel="Mean Bayes risk",
        xscale=log10, xticks=[1, 10, 100, 500])
    lines!(ax1, gens_log, risk_access, color=:steelblue, linewidth=2.5,
        label="Reachable to dominant class")
    lines!(ax1, gens_log, risk_trapped, color=:firebrick, linewidth=2.5,
        label="Trapped in suboptimal class")
    hlines!(ax1, [R_star_K1], color=:steelblue, linestyle=:dash, linewidth=2,
        label="Asymptotic risk (dominant)")
    hlines!(ax1, [R_star_K2], color=:firebrick, linestyle=:dash, linewidth=2,
        label="Asymptotic risk (trapped)")
    axislegend(ax1, position=:rt)
    save(joinpath(outdir, "fig_accessible_vs_trapped.pdf"), fig1; pt_per_unit=1)
    save(joinpath(outdir, "fig_accessible_vs_trapped.png"), fig1; px_per_unit=3)

    fig2 = Figure(size=(900, 520), fontsize=16)
    ax2 = Axis(fig2[1, 1],
        title="Quotient Dynamics: Mass by Communicating Class",
        xlabel="Generation", ylabel="Population mass",
        xscale=log10, xticks=[1, 10, 100, 500])
    lines!(ax2, gens_log, mK1, color=:dodgerblue, linewidth=2.5,
        label="K1 (dominant closed class)")
    lines!(ax2, gens_log, mK2, color=:darkorange, linewidth=2.5,
        label="K2 (subdominant closed class)")
    lines!(ax2, gens_log, mTr, color=:gray45, linewidth=2.5,
        label="Transient set")
    ylims!(ax2, -0.02, 1.02)
    axislegend(ax2, position=:rc)
    save(joinpath(outdir, "fig_class_mass_dynamics.pdf"), fig2; pt_per_unit=1)
    save(joinpath(outdir, "fig_class_mass_dynamics.png"), fig2; px_per_unit=3)

    fig3 = Figure(size=(900, 520), fontsize=16)
    ax3 = Axis(fig3[1, 1],
        title="Periodic Dominant Block: Oscillation and Cesaro Stabilization",
        xlabel="Generation", ylabel="x_1 mass",
        xscale=log10, xticks=[1, 10, 100, 500])
    lines!(ax3, gens_log, x1, color=:purple4, linewidth=1.5, alpha=0.6,
        label="x_1 (raw trajectory)")
    lines!(ax3, gens_log, x1_cesaro, color=:seagreen4, linewidth=2.5,
        label="Cesaro mean")
    hlines!(ax3, [xpf1], color=:black, linestyle=:dash, linewidth=2,
        label="Perron profile")
    axislegend(ax3, position=:rt)
    save(joinpath(outdir, "fig_periodic_block_cesaro.pdf"), fig3; pt_per_unit=1)
    save(joinpath(outdir, "fig_periodic_block_cesaro.png"), fig3; px_per_unit=3)

    # Composite 3-panel figure for manuscript.
    # Designed at target print proportions (~180mm wide) so fonts stay
    # readable when the figure is scaled to fit a page.
    fig = Figure(size=(780, 320), fontsize=12)

    c1 = Axis(fig[1, 1],
        title="(A) Accessible vs Trapped Risk",
        xlabel="Generation", ylabel="Mean Bayes risk",
        xscale=log10, xticks=[1, 10, 100, 500],
        titlesize=13, xlabelsize=11, ylabelsize=11,
        xticklabelsize=10, yticklabelsize=10)
    l1a = lines!(c1, gens_log, risk_access, color=:steelblue, linewidth=2.5)
    l1b = lines!(c1, gens_log, risk_trapped, color=:firebrick, linewidth=2.5)
    l1c = hlines!(c1, [R_star_K1], color=:steelblue, linestyle=:dash, linewidth=1.5)
    l1d = hlines!(c1, [R_star_K2], color=:firebrick, linestyle=:dash, linewidth=1.5)
    Legend(fig[2, 1],
        [l1a, l1b, l1c, l1d],
        ["Reachable", "Trapped",
         L"$R^*_{K_1}$", L"$R^*_{K_2}$"],
        orientation=:horizontal, framevisible=false, labelsize=11,
        nbanks=1, tellwidth=false, tellheight=true, patchsize=(18, 10))

    c2 = Axis(fig[1, 2],
        title="(B) Class Mass on Quotient",
        xlabel="Generation", ylabel="Population mass",
        xscale=log10, xticks=[1, 10, 100, 500],
        titlesize=13, xlabelsize=11, ylabelsize=11,
        xticklabelsize=10, yticklabelsize=10)
    l2a = lines!(c2, gens_log, mK1, color=:dodgerblue, linewidth=2.5)
    l2b = lines!(c2, gens_log, mK2, color=:darkorange, linewidth=2.5)
    l2c = lines!(c2, gens_log, mTr, color=:gray45, linewidth=2.5)
    ylims!(c2, -0.02, 1.02)
    Legend(fig[2, 2],
        [l2a, l2b, l2c],
        [L"$K_1$", L"$K_2$", "Transient"],
        orientation=:horizontal, framevisible=false, labelsize=11,
        nbanks=1, tellwidth=false, tellheight=true, patchsize=(18, 10))

    c3 = Axis(fig[1, 3],
        title="(C) Periodic Block + Cesaro",
        xlabel="Generation", ylabel=L"$x_1$ mass",
        xscale=log10, xticks=[1, 10, 100, 500],
        titlesize=13, xlabelsize=11, ylabelsize=11,
        xticklabelsize=10, yticklabelsize=10)
    l3a = lines!(c3, gens_log, x1, color=:purple4, linewidth=1.2, alpha=0.6)
    l3b = lines!(c3, gens_log, x1_cesaro, color=:seagreen4, linewidth=2.5)
    l3c = hlines!(c3, [xpf1], color=:black, linestyle=:dash, linewidth=1.5)
    Legend(fig[2, 3],
        [l3a, l3b, l3c],
        [L"$x_1$", "Cesaro", "Perron"],
        orientation=:horizontal, framevisible=false, labelsize=11,
        nbanks=1, tellwidth=false, tellheight=true, patchsize=(18, 10))

    rowgap!(fig.layout, 1, 5)
    colgap!(fig.layout, 15)

    save(joinpath(outdir, "fig_theorem_dynamics_panels.pdf"), fig; pt_per_unit=1)
    save(joinpath(outdir, "fig_theorem_dynamics_panels.png"), fig; px_per_unit=4)

    println("Saved theorem-focused figures in: " * outdir)
end

main()
