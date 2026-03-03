"""
Analyze available simulation outputs and generate manuscript-friendly plots.

Runs on currently available files in ../data:
  - sim1_berke.jld2
  - sim3_nonuniform.jld2
  - sim2_condition.jld2 (optional; skipped if absent)

Run with:
  julia --project=. scripts/analyze_and_plot_results.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using DataFrames
using JLD2
using Statistics
using Dates
using Printf

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const FIG_DIR = joinpath(@__DIR__, "..", "figures", "sim1_sim3")
const REPORT_PATH = joinpath(DATA_DIR, "analysis_summary.md")

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

function summarise_sim1(io::IO)
    sim1_path = joinpath(DATA_DIR, "sim1_berke.jld2")
    if !isfile(sim1_path)
        println(io, "- sim1: missing `sim1_berke.jld2` (skipped).")
        return
    end

    results = load(sim1_path, "results")
    max_gen = maximum(results.generation)
    final = results[results.generation .== max_gen, :]
    Ts = sort(unique(Int.(final.T)))

    # Aggregate final stats by T
    rows = NamedTuple[]
    for T in Ts
        sub = final[Int.(final.T) .== T, :]
        m_r, ci_r = mean_ci95(sub.mean_risk)
        m_f, ci_f = mean_ci95(sub.mean_fitness)
        m_d, ci_d = mean_ci95(sub.pred_delta_mu)
        push!(rows, (
            T=T,
            mean_risk=m_r, ci_risk=ci_r,
            mean_fitness=m_f, ci_fitness=ci_f,
            mean_pred_delta=m_d, ci_pred_delta=ci_d
        ))
    end
    agg = DataFrame(rows)

    # Figure 1: final risk vs T
    fig1 = Figure(size=(900, 520), fontsize=16)
    ax1 = Axis(fig1[1, 1],
        title="Final Mean Bayes Risk vs Number of Tasks",
        xlabel=L"$T$ (tasks/generation)",
        ylabel="Final mean risk")
    lines!(ax1, agg.T, agg.mean_risk, color=:navy, linewidth=2.5)
    scatter!(ax1, agg.T, agg.mean_risk, color=:navy, markersize=8)
    errorbars!(ax1, agg.T, agg.mean_risk, agg.ci_risk, color=:navy, whiskerwidth=10)
    ax1.xscale = log10
    save(joinpath(FIG_DIR, "sim1_final_risk_vs_T.pdf"), fig1; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim1_final_risk_vs_T.png"), fig1; px_per_unit=3)

    # Figure 2: risk trajectories for representative T values
    chosen_T = [1, 10, 100, 1000]
    chosen_T = [T for T in chosen_T if T in Ts]
    fig2 = Figure(size=(980, 560), fontsize=16)
    ax2 = Axis(fig2[1, 1],
        title="Risk Trajectories (Representative T Values)",
        xlabel="Generation",
        ylabel="Mean risk")
    palette = [:dodgerblue, :darkorange, :forestgreen, :firebrick]
    for (k, T) in enumerate(chosen_T)
        sub = results[Int.(results.T) .== T, :]
        gdf = combine(groupby(sub, :generation), :mean_risk => mean => :mean_risk)
        lines!(ax2, gdf.generation, gdf.mean_risk, color=palette[k], linewidth=2.5,
            label="T = " * string(T))
    end
    axislegend(ax2, position=:rt)
    save(joinpath(FIG_DIR, "sim1_risk_trajectories_selected_T.pdf"), fig2; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim1_risk_trajectories_selected_T.png"), fig2; px_per_unit=3)

    # Figure 3: predicted separation margin vs T
    fig3 = Figure(size=(900, 520), fontsize=16)
    ax3 = Axis(fig3[1, 1],
        title="Predicted Separation Margin vs T",
        xlabel=L"$T$ (tasks/generation)",
        ylabel=L"Predicted $\delta_\mu$")
    lines!(ax3, agg.T, agg.mean_pred_delta, color=:purple4, linewidth=2.5)
    scatter!(ax3, agg.T, agg.mean_pred_delta, color=:purple4, markersize=8)
    errorbars!(ax3, agg.T, agg.mean_pred_delta, agg.ci_pred_delta, color=:purple4, whiskerwidth=10)
    ax3.xscale = log10
    save(joinpath(FIG_DIR, "sim1_pred_delta_vs_T.pdf"), fig3; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim1_pred_delta_vs_T.png"), fig3; px_per_unit=3)

    println(io, "- sim1 loaded: $(nrow(results)) rows, $(length(Ts)) T-values, final generation = $(Int(max_gen)).")
    println(io, "- sim1 final mean-risk range across T: $(round(minimum(agg.mean_risk), digits=4)) to $(round(maximum(agg.mean_risk), digits=4)).")
    println(io, "- sim1 note: `frac_eco_veridical` and `frac_full_veridical` are zero throughout for this setup (M=2 with almost surely separating task draws), so risk/fitness trajectories are the informative diagnostics.")
end

function summarise_sim3(io::IO)
    sim3_path = joinpath(DATA_DIR, "sim3_nonuniform.jld2")
    if !isfile(sim3_path)
        println(io, "- sim3: missing `sim3_nonuniform.jld2` (skipped).")
        return
    end

    results = load(sim3_path, "results")
    summary = load(sim3_path, "summary")
    wrs = sort(unique(Float64.(summary.weight_ratio)))

    rows = NamedTuple[]
    for wr in wrs
        sub = summary[Float64.(summary.weight_ratio) .== wr, :]
        m_r, ci_r = mean_ci95(sub.final_mean_risk)
        m_o, ci_o = mean_ci95(sub.optimal_risk)
        m_gap, ci_gap = mean_ci95(sub.final_mean_risk .- sub.optimal_risk)
        m_pa, ci_pa = mean_ci95(Float64.(sub.percepts_region_a))
        m_pb, ci_pb = mean_ci95(Float64.(sub.percepts_region_b))
        m_diff, ci_diff = mean_ci95(Float64.(sub.percepts_region_a .- sub.percepts_region_b))
        push!(rows, (
            weight_ratio=wr,
            mean_final_risk=m_r, ci_final_risk=ci_r,
            mean_opt_risk=m_o, ci_opt_risk=ci_o,
            mean_gap=m_gap, ci_gap=ci_gap,
            mean_percepts_a=m_pa, ci_percepts_a=ci_pa,
            mean_percepts_b=m_pb, ci_percepts_b=ci_pb,
            mean_percept_diff=m_diff, ci_percept_diff=ci_diff
        ))
    end
    agg = DataFrame(rows)

    # Figure 4: final risk and optimal risk vs weight ratio
    fig4 = Figure(size=(920, 540), fontsize=16)
    ax4 = Axis(fig4[1, 1],
        title="Final Risk vs Optimal Partition Risk",
        xlabel="Weight ratio (Category A / Category B)",
        ylabel="Risk")
    lines!(ax4, agg.weight_ratio, agg.mean_final_risk, color=:firebrick, linewidth=2.5,
        label="Evolved final risk")
    scatter!(ax4, agg.weight_ratio, agg.mean_final_risk, color=:firebrick, markersize=8)
    errorbars!(ax4, agg.weight_ratio, agg.mean_final_risk, agg.ci_final_risk, color=:firebrick, whiskerwidth=10)
    lines!(ax4, agg.weight_ratio, agg.mean_opt_risk, color=:black, linewidth=2.5, linestyle=:dash,
        label="Swap-search optimum risk")
    scatter!(ax4, agg.weight_ratio, agg.mean_opt_risk, color=:black, markersize=8)
    errorbars!(ax4, agg.weight_ratio, agg.mean_opt_risk, agg.ci_opt_risk, color=:black, whiskerwidth=10)
    ax4.xscale = log10
    axislegend(ax4, position=:rt)
    save(joinpath(FIG_DIR, "sim3_final_vs_optimal_risk.pdf"), fig4; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim3_final_vs_optimal_risk.png"), fig4; px_per_unit=3)

    # Figure 5: percept allocation by region vs weight ratio
    fig5 = Figure(size=(920, 540), fontsize=16)
    ax5 = Axis(fig5[1, 1],
        title="Distinct Percepts by Region",
        xlabel="Weight ratio (Category A / Category B)",
        ylabel="Mean distinct percept labels")
    lines!(ax5, agg.weight_ratio, agg.mean_percepts_a, color=:dodgerblue, linewidth=2.5,
        label="Region A")
    scatter!(ax5, agg.weight_ratio, agg.mean_percepts_a, color=:dodgerblue, markersize=8)
    errorbars!(ax5, agg.weight_ratio, agg.mean_percepts_a, agg.ci_percepts_a, color=:dodgerblue, whiskerwidth=10)
    lines!(ax5, agg.weight_ratio, agg.mean_percepts_b, color=:darkorange, linewidth=2.5,
        label="Region B")
    scatter!(ax5, agg.weight_ratio, agg.mean_percepts_b, color=:darkorange, markersize=8)
    errorbars!(ax5, agg.weight_ratio, agg.mean_percepts_b, agg.ci_percepts_b, color=:darkorange, whiskerwidth=10)
    ax5.xscale = log10
    axislegend(ax5, position=:rb)
    save(joinpath(FIG_DIR, "sim3_region_percept_allocation.pdf"), fig5; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim3_region_percept_allocation.png"), fig5; px_per_unit=3)

    # Figure 6: risk trajectories by weight ratio (replicate-averaged)
    fig6 = Figure(size=(980, 560), fontsize=16)
    ax6 = Axis(fig6[1, 1],
        title="Mean Risk Trajectory by Weight Ratio",
        xlabel="Generation",
        ylabel="Mean risk")
    palette = [:black, :steelblue, :seagreen4, :darkorange, :firebrick]
    for (k, wr) in enumerate(wrs)
        sub = results[Float64.(results.weight_ratio) .== wr, :]
        gdf = combine(groupby(sub, :generation), :mean_risk => mean => :mean_risk)
        lines!(ax6, gdf.generation, gdf.mean_risk, color=palette[k], linewidth=2.5,
            label="wr = " * string(Int(round(wr))))
    end
    axislegend(ax6, position=:rt)
    save(joinpath(FIG_DIR, "sim3_risk_trajectories_by_weight_ratio.pdf"), fig6; pt_per_unit=1)
    save(joinpath(FIG_DIR, "sim3_risk_trajectories_by_weight_ratio.png"), fig6; px_per_unit=3)

    x = log10.(Float64.(summary.weight_ratio))
    c_pa = corr(x, Float64.(summary.percepts_region_a))
    c_pb = corr(x, Float64.(summary.percepts_region_b))
    c_r = corr(x, Float64.(summary.final_mean_risk))
    gap = summary.final_mean_risk .- summary.optimal_risk

    println(io, "- sim3 loaded: $(nrow(results)) trajectory rows and $(nrow(summary)) summary rows.")
    println(io, "- sim3 correlation checks:")
    println(io, "  - corr(log10 weight_ratio, percepts_region_a) = $(round(c_pa, digits=4))")
    println(io, "  - corr(log10 weight_ratio, percepts_region_b) = $(round(c_pb, digits=4))")
    println(io, "  - corr(log10 weight_ratio, final_mean_risk) = $(round(c_r, digits=4))")
    println(io, "- sim3 mean final-optimal risk gap: $(round(mean(gap), digits=5)) ± $(round(std(gap), digits=5)).")
end

function summarise_sim2(io::IO)
    sim2_path = joinpath(DATA_DIR, "sim2_condition.jld2")
    if !isfile(sim2_path)
        println(io, "- sim2: `sim2_condition.jld2` not found yet (log exists only), so no verification/plots for Section 2-condition predictions yet.")
        return
    end
    results = load(sim2_path, "results")
    println(io, "- sim2 loaded: $(nrow(results)) rows. (Add dedicated condition-number analysis next.)")
end

open(REPORT_PATH, "w") do io
    println(io, "# Simulation Analysis Summary")
    println(io)
    println(io, "Generated: $(Dates.now())")
    println(io)
    println(io, "## Verification Notes")
    summarise_sim1(io)
    summarise_sim2(io)
    summarise_sim3(io)
    println(io)
    println(io, "## Output Figures")
    println(io, "- sim1_final_risk_vs_T.(pdf|png)")
    println(io, "- sim1_risk_trajectories_selected_T.(pdf|png)")
    println(io, "- sim1_pred_delta_vs_T.(pdf|png)")
    println(io, "- sim3_final_vs_optimal_risk.(pdf|png)")
    println(io, "- sim3_region_percept_allocation.(pdf|png)")
    println(io, "- sim3_risk_trajectories_by_weight_ratio.(pdf|png)")
end

println("Wrote analysis summary to: ", REPORT_PATH)
println("Saved plots in: ", FIG_DIR)
