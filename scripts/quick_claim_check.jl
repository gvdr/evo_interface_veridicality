"""
Quick verifier for the main claim using a controlled finite task family.

Idea:
- World has N=8 states (3-bit labels).
- Task ecology has m_active in {1,2,3} binary tasks, each reading one bit.
- This gives exact ecological complexity k_mu = 2^m_active.
- Compare capacities M=2 vs M=8 under the same evolutionary dynamics.

Expected qualitative pattern:
- M=8 (sufficient capacity): full veridicality should be selected when m_active=3.
- M=2 (insufficient when m_active>=2): ecological veridicality collapses and risk stays positive.

Run:
  julia --project=. scripts/quick_claim_check.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using CairoMakie
using DataFrames
using JLD2
using Random
using Statistics

struct BitTaskSampler <: TaskSampler
    N::Int
    B::Float64
    task_bank::Matrix{Float64}  # rows = tasks, cols = world states
    m_active::Int
end

function EvoPaper.sample_task(rng::AbstractRNG, s::BitTaskSampler)
    idx = rand(rng, 1:s.m_active)
    return vec(copy(view(s.task_bank, idx, :)))
end

function make_bit_task_bank(N::Int, nbits::Int)
    tasks = zeros(Float64, nbits, N)
    for bit in 1:nbits
        for i in 1:N
            w = i - 1
            tasks[bit, i] = ((w >>> (bit - 1)) & 0x01) == 0x01 ? 1.0 : 0.0
        end
    end
    return tasks
end

mean_ci95(v::AbstractVector{<:Real}) = begin
    x = Float64.(v)
    n = length(x)
    m = mean(x)
    se = n > 1 ? std(x) / sqrt(n) : 0.0
    (m, 1.96 * se)
end

function main()
    out_data = joinpath(@__DIR__, "..", "data", "quick_claim_check.jld2")
    out_fig_dir = joinpath(@__DIR__, "..", "figures", "quick_checks")
    out_fig_pdf = joinpath(out_fig_dir, "quick_claim_check_panels.pdf")
    out_fig_png = joinpath(out_fig_dir, "quick_claim_check_panels.png")
    mkpath(dirname(out_data))
    mkpath(out_fig_dir)

    # Small/fast setup
    N = 8
    nbits = 3
    m_values = [1, 2, 3]          # task diversity levels
    M_values = [2, 8]             # low vs sufficient capacity
    reps = 16

    K = 300
    T = 60                         # tasks per generation (good approximation to ecology)
    n_generations = 1500
    epsilon = 1e-4
    C = 2.0
    B = 1.0

    task_bank = make_bit_task_bank(N, nbits)
    rows = NamedTuple[]
    all_hist = DataFrame[]

    for M in M_values
        for m_active in m_values
            sampler = BitTaskSampler(N, B, task_bank, m_active)
            for rep in 1:reps
                seed = 10_000 * M + 1_000 * m_active + rep
                params = EvolutionParams(N, M, K, T, n_generations, epsilon, C, B)
                hist = run_evolution(params, sampler; seed=seed, show_progress=false)
                hist[!, :M] .= M
                hist[!, :m_active] .= m_active
                hist[!, :rep] .= rep
                push!(all_hist, hist)

                last = hist[end, :]
                push!(rows, (
                    M=M, m_active=m_active, rep=rep,
                    final_mean_risk=Float64(last.mean_risk),
                    final_frac_eco=Float64(last.frac_eco_veridical),
                    final_frac_full=Float64(last.frac_full_veridical),
                    final_mean_complexity=Float64(last.mean_used_complexity)
                ))
            end
        end
    end

    summary = DataFrame(rows)
    history = vcat(all_hist...)
    jldsave(out_data; summary=summary, history=history)

    # Aggregate for plotting
    agg_rows = NamedTuple[]
    for M in M_values, m_active in m_values
        sub = summary[(summary.M .== M) .& (summary.m_active .== m_active), :]
        mr, ci_r = mean_ci95(sub.final_mean_risk)
        me, ci_e = mean_ci95(sub.final_frac_eco)
        mf, ci_f = mean_ci95(sub.final_frac_full)
        mc, ci_c = mean_ci95(sub.final_mean_complexity)
        push!(agg_rows, (
            M=M, m_active=m_active,
            mean_risk=mr, ci_risk=ci_r,
            mean_eco=me, ci_eco=ci_e,
            mean_full=mf, ci_full=ci_f,
            mean_complexity=mc, ci_complexity=ci_c
        ))
    end
    agg = DataFrame(agg_rows)

    # 3-panel figure
    fig = Figure(size=(1400, 470), fontsize=14)

    ax1 = Axis(fig[1, 1], title="(A) Final Ecological Veridical Fraction",
        xlabel=L"Task diversity $m_\mathrm{active}$", ylabel="Eco-veridical fraction")
    for (M, color) in zip(M_values, [:firebrick, :dodgerblue])
        sub = agg[agg.M .== M, :]
        lines!(ax1, sub.m_active, sub.mean_eco, color=color, linewidth=2.5, label="M = " * string(M))
        scatter!(ax1, sub.m_active, sub.mean_eco, color=color, markersize=8)
        errorbars!(ax1, sub.m_active, sub.mean_eco, sub.ci_eco, color=color, whiskerwidth=10)
    end
    axislegend(ax1, position=:rb)
    ylims!(ax1, -0.02, 1.02)

    ax2 = Axis(fig[1, 2], title="(B) Final Full Veridical Fraction",
        xlabel=L"Task diversity $m_\mathrm{active}$", ylabel="Full-veridical fraction")
    for (M, color) in zip(M_values, [:firebrick, :dodgerblue])
        sub = agg[agg.M .== M, :]
        lines!(ax2, sub.m_active, sub.mean_full, color=color, linewidth=2.5, label="M = " * string(M))
        scatter!(ax2, sub.m_active, sub.mean_full, color=color, markersize=8)
        errorbars!(ax2, sub.m_active, sub.mean_full, sub.ci_full, color=color, whiskerwidth=10)
    end
    ylims!(ax2, -0.02, 1.02)

    ax3 = Axis(fig[1, 3], title="(C) Final Mean Risk",
        xlabel=L"Task diversity $m_\mathrm{active}$", ylabel="Mean Bayes risk")
    for (M, color) in zip(M_values, [:firebrick, :dodgerblue])
        sub = agg[agg.M .== M, :]
        lines!(ax3, sub.m_active, sub.mean_risk, color=color, linewidth=2.5, label="M = " * string(M))
        scatter!(ax3, sub.m_active, sub.mean_risk, color=color, markersize=8)
        errorbars!(ax3, sub.m_active, sub.mean_risk, sub.ci_risk, color=color, whiskerwidth=10)
    end

    save(out_fig_pdf, fig; pt_per_unit=1)
    save(out_fig_png, fig; px_per_unit=3)

    println("Saved quick verifier data: ", out_data)
    println("Saved quick verifier figure: ", out_fig_pdf)
    println()
    println("Aggregate means:")
    for M in M_values
        for m_active in m_values
            sub = agg[(agg.M .== M) .& (agg.m_active .== m_active), :]
            println("M=", M, " m=", m_active,
                " eco=", round(sub.mean_eco[1], digits=3),
                " full=", round(sub.mean_full[1], digits=3),
                " risk=", round(sub.mean_risk[1], digits=4))
        end
    end
end

main()
