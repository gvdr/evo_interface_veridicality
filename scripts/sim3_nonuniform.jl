"""
Simulation 3: Non-Uniform mu and Adaptive Partition

Tests that non-uniform task weighting drives asymmetric percept allocation.
N=20, M=5, T=100, weight_ratios in [1,3,10,30,100], 30 reps.

Launch with: julia -t 16 --project=. scripts/sim3_nonuniform.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using DataFrames
using JLD2
using Random
using Dates

const N = 20
const M = 5
const K = 500
const EPSILON = 0.001
const N_GEN = 10000
const C = 2.0
const B = 1.0
const T_FIXED = 100
const WEIGHT_RATIOS = [1.0, 3.0, 10.0, 30.0, 100.0]
const N_REPS = 30

function make_weighted_sampler(N::Int, B::Float64, weight_ratio::Float64)
    # Category A ("predation"): Beta tasks with mode in [0.1, 0.5]
    #   mode = (a-1)/(a+b-2), so we pick a,b to get modes in that range
    cat_a = BetaTaskSampler(N, B; a_range=(1.5, 5.0), b_range=(3.0, 10.0))

    # Category B ("texture"): Beta tasks with mode in [0.5, 0.9]
    cat_b = BetaTaskSampler(N, B; a_range=(3.0, 10.0), b_range=(1.5, 5.0))

    w_a = weight_ratio
    w_b = 1.0
    total = w_a + w_b
    weights = [w_a / total, w_b / total]

    return WeightedTaskSampler(N, B, [cat_a, cat_b], weights)
end

function count_percepts_in_region(p::Vector{Int}, region_indices::Vector{Int})
    # Count how many distinct percept values are used in the given region
    return length(unique(p[i] for i in region_indices))
end

function main()
    t_start = now()

    # Region A = lower half of world states (modes ~ 0.1-0.5)
    # Region B = upper half
    region_a = collect(1:div(N, 2))
    region_b = collect((div(N, 2) + 1):N)
    pi = fill(1.0 / N, N)

    # Build work items: (weight_ratio, rep)
    work_items = [(wr, rep) for wr in WEIGHT_RATIOS for rep in 1:N_REPS]
    n_work = length(work_items)
    counter = Threads.Atomic{Int}(0)
    io_lock = ReentrantLock()

    logpath = joinpath(@__DIR__, "..", "data", "sim3_nonuniform.log")
    mkpath(dirname(logpath))
    logfile = open(logpath, "w")

    function log_msg(msg::String)
        lock(io_lock) do
            line = Dates.format(now(), "HH:MM:SS") * "  " * msg
            println(line)
            println(logfile, line)
            flush(logfile)
        end
    end

    log_msg("sim3_nonuniform started: " * string(n_work) * " work items, " *
            string(Threads.nthreads()) * " threads")
    log_msg("Params: N=" * string(N) * " M=" * string(M) * " K=" * string(K) *
            " n_gen=" * string(N_GEN) * " T=" * string(T_FIXED))
    log_msg("Weight ratios: " * string(WEIGHT_RATIOS))

    history_results = Vector{DataFrame}(undef, n_work)

    # Summary data per work item
    summary_data = Vector{NamedTuple{(:weight_ratio, :rep, :final_mean_risk,
                                       :final_eco_veridical, :percepts_region_a,
                                       :percepts_region_b, :optimal_risk),
                                      Tuple{Float64, Int, Float64, Float64,
                                            Int, Int, Float64}}}(undef, n_work)

    Threads.@threads for idx in eachindex(work_items)
        wr, rep = work_items[idx]
        sampler = make_weighted_sampler(N, B, wr)
        seed = Int(1000 * wr) + rep
        params = EvolutionParams(N, M, K, T_FIXED, N_GEN, EPSILON, C, B)

        history, final_pop = run_evolution(params, sampler;
                                           seed=seed, show_progress=false,
                                           return_population=true)
        history[!, :weight_ratio] .= wr
        history[!, :rep] .= rep
        history_results[idx] = history

        # Count percepts allocated to each region from dominant encoding
        enc_counts = Dict{Vector{Int}, Int}()
        for p in final_pop
            enc_counts[p] = get(enc_counts, p, 0) + 1
        end
        best_enc = argmax(enc_counts)
        pa = count_percepts_in_region(best_enc, region_a)
        pb = count_percepts_in_region(best_enc, region_b)

        # Compute optimal partition for comparison
        F_opt = build_task_matrix(MersenneTwister(seed + 999999), sampler, T_FIXED)
        _, opt_risk = optimal_partition_swap(F_opt, pi, M;
                                              rng=MersenneTwister(seed),
                                              n_restarts=50)

        last_row = history[end, :]
        summary_data[idx] = (
            weight_ratio=wr, rep=rep,
            final_mean_risk=last_row.mean_risk,
            final_eco_veridical=last_row.frac_eco_veridical,
            percepts_region_a=pa, percepts_region_b=pb,
            optimal_risk=opt_risk
        )

        prev = Threads.atomic_add!(counter, 1)
        done = prev + 1
        elapsed_s = Dates.value(now() - t_start) / 1000
        eta_s = done < n_work ? elapsed_s * (n_work - done) / done : 0.0
        log_msg("[" * string(done) * "/" * string(n_work) * "]  wr=" *
                string(wr) * " rep=" * string(rep) *
                "  regA=" * string(pa) * " regB=" * string(pb) *
                "  elapsed=" * string(round(elapsed_s; digits=1)) * "s" *
                "  ETA=" * string(round(eta_s; digits=1)) * "s")
    end

    all_results = vcat(history_results...)
    summary_results = DataFrame(summary_data)

    outpath = joinpath(@__DIR__, "..", "data", "sim3_nonuniform.jld2")
    jldsave(outpath; results=all_results, summary=summary_results)

    total_s = Dates.value(now() - t_start) / 1000
    log_msg("Saved results to " * outpath)
    log_msg("Done! Total time: " * string(round(total_s; digits=1)) * "s")
    close(logfile)
end

main()
