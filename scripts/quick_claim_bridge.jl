"""
Bridge test for the quick claim setup (M=8, m_active=3).

Goal:
Check whether eco/full-veridical mass increases when we move closer to theorem
assumptions:
1) lower mutation,
2) larger population and longer runs,
3) reduced task-sampling noise (deterministic full task bank per generation).

Run:
  julia --project=. scripts/quick_claim_bridge.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using CairoMakie
using DataFrames
using JLD2
using Random
using Statistics

struct BitTaskSamplerBridge <: TaskSampler
    N::Int
    B::Float64
    task_bank::Matrix{Float64}  # rows = tasks, cols = world states
    m_active::Int
    deterministic_bank::Bool
end

function EvoPaper.sample_task(rng::AbstractRNG, s::BitTaskSamplerBridge)
    idx = rand(rng, 1:s.m_active)
    return vec(copy(view(s.task_bank, idx, :)))
end

function EvoPaper.build_task_matrix(rng::AbstractRNG, s::BitTaskSamplerBridge, T::Int)
    N = s.N
    F = Matrix{Float64}(undef, T, N)
    if s.deterministic_bank
        # Cycle through active tasks evenly -> no task-sampling noise.
        for t in 1:T
            idx = ((t - 1) % s.m_active) + 1
            @inbounds for i in 1:N
                F[t, i] = s.task_bank[idx, i]
            end
        end
    else
        for t in 1:T
            f = EvoPaper.sample_task(rng, s)
            @inbounds for i in 1:N
                F[t, i] = f[i]
            end
        end
    end
    return F
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
    out_data = joinpath(@__DIR__, "..", "data", "quick_claim_bridge.jld2")
    out_fig_dir = joinpath(@__DIR__, "..", "figures", "quick_checks")
    out_fig_pdf = joinpath(out_fig_dir, "quick_claim_bridge.pdf")
    out_fig_png = joinpath(out_fig_dir, "quick_claim_bridge.png")
    mkpath(dirname(out_data))
    mkpath(out_fig_dir)

    N = 8
    M = 8
    nbits = 3
    m_active = 3
    T = 60
    C = 2.0
    B = 1.0
    task_bank = make_bit_task_bank(N, nbits)

    # Four bridge conditions from weak -> strong approximation to theorem assumptions
    conds = [
        (name="A baseline", epsilon=1e-4, K=300, n_generations=1500, deterministic_bank=false, n_reps=12),
        (name="B low-mutation", epsilon=1e-6, K=300, n_generations=1500, deterministic_bank=false, n_reps=12),
        (name="C low-mutation + larger K/time", epsilon=1e-6, K=600, n_generations=3000, deterministic_bank=false, n_reps=8),
        (name="D + deterministic task bank", epsilon=1e-6, K=600, n_generations=3000, deterministic_bank=true, n_reps=8),
    ]

    hist_list = DataFrame[]
    summary_rows = NamedTuple[]

    for (ci, c) in enumerate(conds)
        sampler = BitTaskSamplerBridge(N, B, task_bank, m_active, c.deterministic_bank)
        for rep in 1:c.n_reps
            seed = 1_000_000 * ci + rep
            params = EvolutionParams(N, M, c.K, T, c.n_generations, c.epsilon, C, B)
            hist = run_evolution(params, sampler; seed=seed, show_progress=false)
            hist[!, :condition] .= c.name
            hist[!, :rep] .= rep
            push!(hist_list, hist)

            last = hist[end, :]
            push!(summary_rows, (
                condition=c.name, rep=rep,
                epsilon=c.epsilon, K=c.K, n_generations=c.n_generations,
                deterministic_bank=c.deterministic_bank,
                final_mean_risk=Float64(last.mean_risk),
                final_frac_eco=Float64(last.frac_eco_veridical),
                final_frac_full=Float64(last.frac_full_veridical),
                final_mean_complexity=Float64(last.mean_used_complexity)
            ))
        end
    end

    history = vcat(hist_list...)
    summary = DataFrame(summary_rows)
    jldsave(out_data; history=history, summary=summary)

    cond_names = [c.name for c in conds]
    x = 1:length(cond_names)

    agg_rows = NamedTuple[]
    for name in cond_names
        sub = summary[summary.condition .== name, :]
        mr, ci_r = mean_ci95(sub.final_mean_risk)
        me, ci_e = mean_ci95(sub.final_frac_eco)
        mf, ci_f = mean_ci95(sub.final_frac_full)
        push!(agg_rows, (
            condition=name,
            mean_risk=mr, ci_risk=ci_r,
            mean_eco=me, ci_eco=ci_e,
            mean_full=mf, ci_full=ci_f
        ))
    end
    agg = DataFrame(agg_rows)

    fig = Figure(size=(1400, 470), fontsize=14)

    ax1 = Axis(fig[1, 1], title="(A) Final Ecological-Veridical Fraction",
        xlabel="Bridge condition", ylabel="Eco-veridical fraction",
        xticks=(x, cond_names), xticklabelrotation=0.25)
    barplot!(ax1, x, agg.mean_eco, color=:dodgerblue)
    errorbars!(ax1, x, agg.mean_eco, agg.ci_eco, color=:black, whiskerwidth=10)
    ylims!(ax1, -0.02, 1.02)

    ax2 = Axis(fig[1, 2], title="(B) Final Full-Veridical Fraction",
        xlabel="Bridge condition", ylabel="Full-veridical fraction",
        xticks=(x, cond_names), xticklabelrotation=0.25)
    barplot!(ax2, x, agg.mean_full, color=:seagreen4)
    errorbars!(ax2, x, agg.mean_full, agg.ci_full, color=:black, whiskerwidth=10)
    ylims!(ax2, -0.02, 1.02)

    ax3 = Axis(fig[1, 3], title="(C) Final Mean Risk",
        xlabel="Bridge condition", ylabel="Mean Bayes risk",
        xticks=(x, cond_names), xticklabelrotation=0.25)
    barplot!(ax3, x, agg.mean_risk, color=:firebrick)
    errorbars!(ax3, x, agg.mean_risk, agg.ci_risk, color=:black, whiskerwidth=10)

    save(out_fig_pdf, fig; pt_per_unit=1)
    save(out_fig_png, fig; px_per_unit=3)

    println("Saved bridge-test data: ", out_data)
    println("Saved bridge-test figure: ", out_fig_pdf)
    println()
    println("Condition means:")
    for r in eachrow(agg)
        println(r.condition, " | eco=", round(r.mean_eco, digits=3),
            " full=", round(r.mean_full, digits=3),
            " risk=", round(r.mean_risk, digits=4))
    end
end

main()
