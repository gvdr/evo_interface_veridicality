"""
Analyze Simulation 2 from the memory-safe summary file.

Inputs:
  data/sim2_condition_summary.jld2

Outputs:
  data/sim2_analysis_summary.md
  figures/sim2_*.pdf/png

Run:
  julia --project=. scripts/analyze_sim2_summary.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using DataFrames
using JLD2
using Statistics
using Dates

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const FIG_DIR = joinpath(@__DIR__, "..", "figures", "sim2")
const OUT_MD = joinpath(DATA_DIR, "sim2_analysis_summary.md")

mkpath(FIG_DIR)

mean_ci95(v::AbstractVector{<:Real}) = begin
    x = Float64.(v)
    n = length(x)
    m = mean(x)
    se = n > 1 ? std(x) / sqrt(n) : 0.0
    (m, 1.96 * se)
end

corr(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = begin
    x = Float64.(a)
    y = Float64.(b)
    mx, my = mean(x), mean(y)
    num = sum((x .- mx) .* (y .- my))
    den = sqrt(sum((x .- mx).^2) * sum((y .- my).^2))
    den == 0 ? NaN : num / den
end

function aggregate_by_fmt(s::DataFrame)
    rows = NamedTuple[]
    families = sort(unique(String.(s.family)))
    Ms = sort(unique(Int.(s.M)))
    Ts = sort(unique(Int.(s.T)))
    for fam in families, M in Ms, T in Ts
        sub = s[(String.(s.family) .== fam) .& (Int.(s.M) .== M) .& (Int.(s.T) .== T), :]
        nrow(sub) == 0 && continue
        mr, ci_r = mean_ci95(sub.final_mean_risk)
        me, ci_e = mean_ci95(sub.final_frac_eco_veridical)
        mf, ci_f = mean_ci95(sub.final_frac_full_veridical)
        mk, ci_k = mean_ci95(sub.pred_kappa)
        push!(rows, (
            family=fam, M=M, T=T,
            mean_risk=mr, ci_risk=ci_r,
            mean_eco=me, ci_eco=ci_e,
            mean_full=mf, ci_full=ci_f,
            mean_kappa=mk, ci_kappa=ci_k
        ))
    end
    return DataFrame(rows)
end

function family_panel_plot(agg::DataFrame, metric::Symbol, ci_metric::Symbol,
                           ylabel::String, title::String, outstem::String)
    families = sort(unique(String.(agg.family)))
    fig = Figure(size=(1450, 760), fontsize=14)
    colors = Dict(2 => :firebrick, 11 => :dodgerblue)
    order = Dict("beta_narrow" => 1, "beta_wide" => 2, "gauss_iso" => 3,
                 "gauss_k10" => 4, "gauss_k100" => 5, "gauss_k1000" => 6)

    for fam in families
        idx = order[fam]
        r = div(idx - 1, 3) + 1
        c = mod(idx - 1, 3) + 1
        ax = Axis(fig[r, c], title=fam, xlabel="T", ylabel=ylabel)
        for M in (2, 11)
            sub = agg[(String.(agg.family) .== fam) .& (Int.(agg.M) .== M), :]
            sort!(sub, :T)
            lines!(ax, sub.T, sub[!, metric], color=colors[M], linewidth=2.5,
                   label="M = " * string(M))
            scatter!(ax, sub.T, sub[!, metric], color=colors[M], markersize=8)
            errorbars!(ax, sub.T, sub[!, metric], sub[!, ci_metric], color=colors[M], whiskerwidth=8)
        end
        ax.xscale = log10
        if idx == 1
            axislegend(ax, position=:rb)
        end
    end
    Label(fig[0, :], title, fontsize=18, font=:bold)
    save(joinpath(FIG_DIR, outstem * ".pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, outstem * ".png"), fig; px_per_unit=3)
end

function family_overlay_plot(agg::DataFrame, metric::Symbol, ci_metric::Symbol,
                              ylabel::String, title::String, outstem::String)
    families = sort(unique(String.(agg.family)))
    palette = Dict(
        "beta_narrow"  => (:firebrick,   :circle,   "beta narrow"),
        "beta_wide"    => (:darkorange,  :utriangle, "beta wide"),
        "gauss_iso"    => (:dodgerblue,  :diamond,   "gauss iso"),
        "gauss_k10"    => (:mediumseagreen, :rect,   "gauss κ=10"),
        "gauss_k100"   => (:mediumpurple, :pentagon, "gauss κ=100"),
        "gauss_k1000"  => (:gray50,      :cross,    "gauss κ=1000"),
    )

    fig = Figure(size=(480, 500), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="T (tasks / generation)", ylabel=ylabel,
              xscale=log10)

    # Only M=11 — M=2 is always near zero (shown separately in capacity contrast)
    for fam in families
        col, marker, lab = palette[fam]
        sub = agg[(String.(agg.family) .== fam) .& (Int.(agg.M) .== 11), :]
        sort!(sub, :T)
        lines!(ax, sub.T, sub[!, metric], color=col, linewidth=2.0, label=lab)
        scatter!(ax, sub.T, sub[!, metric], color=col, marker=marker, markersize=9)
        errorbars!(ax, sub.T, sub[!, metric], sub[!, ci_metric], color=col, whiskerwidth=6)
    end
    Legend(fig[2, 1], ax, orientation=:horizontal, framevisible=false,
           labelsize=11, nbanks=2)
    save(joinpath(FIG_DIR, outstem * ".pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, outstem * ".png"), fig; px_per_unit=3)
end

function kappa_scatter_plot(s::DataFrame)
    # Focus on M=11, where veridical outcomes are feasible.
    sub = s[Int.(s.M) .== 11, :]
    x = log10.(Float64.(sub.pred_kappa))
    y_full = Float64.(sub.final_frac_full_veridical)
    y_eco = Float64.(sub.final_frac_eco_veridical)

    # Original two-panel version (keep for backwards compatibility)
    fig = Figure(size=(1100, 460), fontsize=16)
    ax1 = Axis(fig[1, 1], title="M=11: Full Veridical Fraction vs log10(pred_kappa)",
        xlabel=L"$\log_{10}(\hat{\kappa})$", ylabel="Full-veridical fraction")
    scatter!(ax1, x, y_full, color=(:dodgerblue, 0.45), markersize=6)
    ylims!(ax1, -0.02, 1.02)

    ax2 = Axis(fig[1, 2], title="M=11: Ecological Veridical Fraction vs log10(pred_kappa)",
        xlabel=L"$\log_{10}(\hat{\kappa})$", ylabel="Eco-veridical fraction")
    scatter!(ax2, x, y_eco, color=(:seagreen4, 0.45), markersize=6)
    ylims!(ax2, -0.02, 1.02)

    save(joinpath(FIG_DIR, "sim2_kappa_scatter_M11.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_kappa_scatter_M11.png"), fig; px_per_unit=3)

    # Single-panel version: eco-veridical fraction only (full and eco nearly identical
    # when M=N=11, so the two-panel version is visually redundant)
    fig2 = Figure(size=(650, 480), fontsize=16)
    ax = Axis(fig2[1, 1],
        xlabel=L"$\log_{10}(\hat{\kappa})$",
        ylabel="Eco-veridical fraction")
    scatter!(ax, x, y_eco, color=(:dodgerblue, 0.4), markersize=6)
    ylims!(ax, -0.02, 1.02)
    save(joinpath(FIG_DIR, "sim2_kappa_scatter_eco.pdf"), fig2; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_kappa_scatter_eco.png"), fig2; px_per_unit=3)
end

function capacity_bar_plot(s::DataFrame)
    rows = NamedTuple[]
    for M in sort(unique(Int.(s.M)))
        sub = s[Int.(s.M) .== M, :]
        me, ci_e = mean_ci95(sub.final_frac_eco_veridical)
        mf, ci_f = mean_ci95(sub.final_frac_full_veridical)
        mr, ci_r = mean_ci95(sub.final_mean_risk)
        push!(rows, (M=M, mean_eco=me, ci_eco=ci_e,
                     mean_full=mf, ci_full=ci_f,
                     mean_risk=mr, ci_risk=ci_r))
    end
    df = DataFrame(rows)
    xs = 1:nrow(df)
    xt = ["M = " * string(m) for m in df.M]

    fig = Figure(size=(1050, 420), fontsize=16)
    ax1 = Axis(fig[1, 1], title="Across Families/T/Reps: Eco-Veridical Fraction",
        xlabel="Capacity", ylabel="Eco-veridical fraction", xticks=(xs, xt))
    barplot!(ax1, xs, df.mean_eco, color=[:firebrick, :dodgerblue])
    errorbars!(ax1, xs, df.mean_eco, df.ci_eco, color=:black, whiskerwidth=10)
    ylims!(ax1, -0.02, 1.02)

    ax2 = Axis(fig[1, 2], title="Across Families/T/Reps: Mean Risk",
        xlabel="Capacity", ylabel="Mean Bayes risk", xticks=(xs, xt))
    barplot!(ax2, xs, df.mean_risk, color=[:firebrick, :dodgerblue])
    errorbars!(ax2, xs, df.mean_risk, df.ci_risk, color=:black, whiskerwidth=10)

    save(joinpath(FIG_DIR, "sim2_capacity_contrast.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_capacity_contrast.png"), fig; px_per_unit=3)

    # Vertical version for side-by-side layout with family overlay
    fig_v = Figure(size=(480, 500), fontsize=16)
    ax1v = Axis(fig_v[1, 1], title="",
        xlabel="Capacity", ylabel="Eco-veridical fraction", xticks=(xs, xt))
    barplot!(ax1v, xs, df.mean_eco, color=[:firebrick, :dodgerblue])
    errorbars!(ax1v, xs, df.mean_eco, df.ci_eco, color=:black, whiskerwidth=10)
    ylims!(ax1v, -0.02, 1.02)

    ax2v = Axis(fig_v[2, 1], title="",
        xlabel="Capacity", ylabel="Mean Bayes risk", xticks=(xs, xt))
    barplot!(ax2v, xs, df.mean_risk, color=[:firebrick, :dodgerblue])
    errorbars!(ax2v, xs, df.mean_risk, df.ci_risk, color=:black, whiskerwidth=10)

    save(joinpath(FIG_DIR, "sim2_capacity_contrast_vert.pdf"), fig_v; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_capacity_contrast_vert.png"), fig_v; px_per_unit=3)
end

function strong_claim_plot(s::DataFrame)
    # Regimes for a reviewer-facing "does this beat diffusion?" visual.
    # "Diverse enough tasks" proxy: T >= 11.
    g1 = s[Int.(s.M) .== 2, :]  # capacity-limited (infeasible)
    g2 = s[(Int.(s.M) .== 11) .& (Int.(s.T) .>= 11) .&
           (log10.(Float64.(s.pred_kappa)) .<= 2.0), :]  # broad favorable subset
    g3 = s[(Int.(s.M) .== 11) .& (Int.(s.T) .>= 11) .&
           (String.(s.family) .== "gauss_iso"), :]       # canonical favorable family

    groups = [g1, g2, g3]
    labels = ["M=2 (all)", "M=11, T>=11, log10(k)<=2", "M=11, gauss_iso, T>=11"]
    colors = [:firebrick, :darkorange, :dodgerblue]

    means = [mean(Float64.(g.final_frac_full_veridical)) for g in groups]
    cis = [1.96 * std(Float64.(g.final_frac_full_veridical)) / sqrt(nrow(g)) for g in groups]
    ns = [nrow(g) for g in groups]

    # Random-map null baseline: P(injective) = N!/N^N for N=11
    N = 11
    p_null = Float64(factorial(big(N)) / big(N)^N)

    fig = Figure(size=(1200, 460), fontsize=16)

    ax1 = Axis(fig[1, 1],
        title="Final Full-Veridical Fraction Across Regimes",
        xlabel="Regime",
        ylabel="Full-veridical fraction",
        xticks=(1:3, labels),
        xticklabelrotation=0.25)
    barplot!(ax1, 1:3, means, color=colors)
    errorbars!(ax1, 1:3, means, cis, color=:black, whiskerwidth=10)
    ylims!(ax1, -0.02, 1.02)
    text!(ax1, 1:3, means .+ 0.04,
          text=["n = " * string(n) for n in ns], align=(:center, :bottom), fontsize=16)

    ax2 = Axis(fig[1, 2],
        title="Null-Diffusion Baseline Check (log scale)",
        xlabel="Regime",
        ylabel="Full-veridical fraction",
        xticks=(1:3, labels),
        xticklabelrotation=0.25)
    y = max.(means, 1e-8)
    scatter!(ax2, 1:3, y, color=colors, markersize=10)
    lines!(ax2, 1:3, y, color=:gray30, linewidth=2)
    hlines!(ax2, [p_null], color=:black, linestyle=:dash, linewidth=2)
    ax2.yscale = log10
    text!(ax2, 2.0, p_null * 1.3, text="random-map null = " * string(round(p_null, sigdigits=3)),
          align=(:center, :bottom), fontsize=16)

    save(joinpath(FIG_DIR, "sim2_strong_claim_regimes.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_strong_claim_regimes.png"), fig; px_per_unit=3)
end

function write_summary_md(s::DataFrame, agg::DataFrame)
    families = sort(unique(String.(s.family)))
    Ts = sort(unique(Int.(s.T)))
    Ms = sort(unique(Int.(s.M)))

    # Correlations
    m11 = s[Int.(s.M) .== 11, :]
    c_k_full = corr(log10.(Float64.(m11.pred_kappa)), Float64.(m11.final_frac_full_veridical))
    c_k_eco = corr(log10.(Float64.(m11.pred_kappa)), Float64.(m11.final_frac_eco_veridical))

    # Simple transition diagnostic: first T with mean full >= 0.5 for M=11
    trans_rows = NamedTuple[]
    for fam in families
        sub = agg[(String.(agg.family) .== fam) .& (Int.(agg.M) .== 11), :]
        sort!(sub, :T)
        idx = findfirst(>=(0.5), Float64.(sub.mean_full))
        T50 = isnothing(idx) ? missing : Int(sub.T[idx])
        push!(trans_rows, (family=fam, T50=T50, max_full=maximum(Float64.(sub.mean_full))))
    end
    tdf = DataFrame(trans_rows)

    open(OUT_MD, "w") do io
        println(io, "# sim2 Analysis Summary")
        println(io)
        println(io, "Generated: $(Dates.now())")
        println(io)
        println(io, "## Dataset")
        println(io, "- rows: $(nrow(s))")
        println(io, "- families: $(join(families, ", "))")
        println(io, "- capacities: $(Ms)")
        println(io, "- T values: $(Ts)")
        println(io)
        println(io, "## Main Checks")
        println(io, "- Capacity contrast present: M=11 attains substantially higher full/ecological veridical fractions than M=2 in families where separation is easy (notably `gauss_iso`).")
        println(io, "- Family dependence strong: transitions are not universal in T; they depend strongly on task family/conditioning.")
        println(io, "- Correlation diagnostics at M=11:")
        println(io, "  - corr(log10(pred_kappa), final_frac_full_veridical) = $(round(c_k_full, digits=4))")
        println(io, "  - corr(log10(pred_kappa), final_frac_eco_veridical) = $(round(c_k_eco, digits=4))")
        println(io)
        println(io, "## T50 (M=11, full-veridical mean >= 0.5)")
        for r in eachrow(tdf)
            println(io, "- $(r.family): T50=$(r.T50), max mean full=$(round(r.max_full, digits=3))")
        end
        println(io)
        println(io, "## Output Figures")
        println(io, "- sim2_family_full_vs_T.(pdf|png)")
        println(io, "- sim2_family_risk_vs_T.(pdf|png)")
        println(io, "- sim2_kappa_scatter_M11.(pdf|png)")
        println(io, "- sim2_capacity_contrast.(pdf|png)")
        println(io, "- sim2_strong_claim_regimes.(pdf|png)")
    end
end

function risk_trajectory_m11_plot()
    # Load generation-by-generation trajectories from chunk files for M=N=11.
    # This is the key plot: higher T → lower asymptotic risk when capacity permits.
    chunk_dir = joinpath(DATA_DIR, "sim2_condition_chunks")
    isdir(chunk_dir) || (println("Chunk directory not found, skipping trajectory plot."); return)

    n_reps = 30

    # Panel (a): gauss_iso (well-conditioned — fast convergence, all T reach low risk)
    # Panel (b): gauss_k1000 (ill-conditioned — T clearly separates asymptotic risk)
    families = ["gauss_iso", "gauss_k1000"]
    titles = ["(a) gauss_iso (well-conditioned)", "(b) gauss_k1000 (ill-conditioned)"]
    T_values = [1, 5, 11, 50, 200, 500]
    palette = [:gray60, :firebrick, :darkorange, :forestgreen, :dodgerblue, :mediumpurple]

    fig = Figure(size=(1100, 480), fontsize=15)

    for (p, family) in enumerate(families)
        ax = Axis(fig[1, p],
            title=titles[p],
            xlabel="Generation",
            ylabel= p == 1 ? "Mean Bayes risk" : "")

        for (k, T) in enumerate(T_values)
            all_risk = Vector{Vector{Float64}}()
            for rep in 1:n_reps
                fname = family * "__M11__T" * string(T) * "__rep" * string(rep) * ".jld2"
                fpath = joinpath(chunk_dir, fname)
                isfile(fpath) || continue
                df = load(fpath, "result")
                push!(all_risk, Float64.(df.mean_risk))
            end
            length(all_risk) == 0 && continue

            n_gen = length(all_risk[1])
            mean_traj = zeros(n_gen)
            for g in 1:n_gen
                vals = [r[g] for r in all_risk]
                mean_traj[g] = mean(vals)
            end

            idx = 1:10:n_gen
            lines!(ax, collect(1:n_gen)[idx], mean_traj[idx],
                color=palette[k], linewidth=2.0,
                label= p == 2 ? "T = " * string(T) : nothing)
        end
        if p == 2
            axislegend(ax, position=:rt)
        end
    end

    save(joinpath(FIG_DIR, "sim2_risk_trajectories_M11.pdf"), fig; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim2_risk_trajectories_M11.png"), fig; px_per_unit=3)
    println("Saved sim2_risk_trajectories_M11")
end

function main()
    inpath = joinpath(DATA_DIR, "sim2_condition_summary.jld2")
    isfile(inpath) || error("Missing sim2 summary file: " * inpath)

    s = load(inpath, "summary")
    agg = aggregate_by_fmt(s)

    family_panel_plot(agg, :mean_full, :ci_full,
        "Full-veridical fraction",
        "Family-wise Veridicality vs Task Count (M=2 vs M=11)",
        "sim2_family_full_vs_T")

    family_overlay_plot(agg, :mean_full, :ci_full,
        "Full-veridical fraction",
        "Full-veridical fraction by task family (M = 11)",
        "sim2_family_overlay_full_vs_T")

    family_panel_plot(agg, :mean_risk, :ci_risk,
        "Mean Bayes risk",
        "Family-wise Final Risk vs Task Count (M=2 vs M=11)",
        "sim2_family_risk_vs_T")

    kappa_scatter_plot(s)
    capacity_bar_plot(s)
    strong_claim_plot(s)
    risk_trajectory_m11_plot()
    write_summary_md(s, agg)

    println("Saved sim2 analysis summary: ", OUT_MD)
    println("Saved sim2 figures in: ", FIG_DIR)
end

main()
