"""
Simulation 1: Berke Replication with Theoretical Overlay

Replicates Berke et al.'s main finding and tests quantitative predictions
of the theory (fitness gap, convergence rate, separation margin, cascade staircase).

Parameters: N=11, M=2, K=1000, epsilon=0.001, n_gen=5000, C=2.0, B=1.0
T values: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
50 independent runs per T.

Launch with: julia -t 16 --project=. scripts/sim1_berke.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using DataFrames
using JLD2
using Random
using Dates

const N = 11
const M = 2
const K = 1000
const EPSILON = 0.001
const N_GEN = 5000
const C = 2.0
const B = 1.0
const T_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
const N_REPS = 50

function main()
    t_start = now()

    work_items = [(T_val, rep) for T_val in T_VALUES for rep in 1:N_REPS]
    n_work = length(work_items)
    counter = Threads.Atomic{Int}(0)
    io_lock = ReentrantLock()

    logpath = joinpath(@__DIR__, "..", "data", "sim1_berke.log")
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

    log_msg("sim1_berke started: " * string(n_work) * " work items, " *
            string(Threads.nthreads()) * " threads")
    log_msg("Params: N=" * string(N) * " M=" * string(M) * " K=" * string(K) *
            " n_gen=" * string(N_GEN) * " eps=" * string(EPSILON))

    sampler = BetaTaskSampler(N, B)
    results = Vector{DataFrame}(undef, n_work)

    Threads.@threads for idx in eachindex(work_items)
        T_val, rep = work_items[idx]
        seed = 1000 * T_val + rep
        params = EvolutionParams(N, M, K, T_val, N_GEN, EPSILON, C, B)

        history = run_evolution(params, sampler; seed=seed, show_progress=false)
        history[!, :T] .= T_val
        history[!, :rep] .= rep

        # Compute theoretical predictions from a large reference sample
        rng = MersenneTwister(seed)
        F_ref = build_task_matrix(rng, sampler, T_val)
        preds = theoretical_predictions(F_ref, fill(1.0 / N, N), C)
        history[!, :pred_delta_mu] .= preds.delta_mu
        history[!, :pred_fitness_gap] .= preds.fitness_gap_lower
        history[!, :pred_kappa] .= preds.kappa
        history[!, :pred_k_T] .= preds.k_T

        results[idx] = history

        prev = Threads.atomic_add!(counter, 1)
        done = prev + 1
        elapsed_s = Dates.value(now() - t_start) / 1000
        eta_s = done < n_work ? elapsed_s * (n_work - done) / done : 0.0
        log_msg("[" * string(done) * "/" * string(n_work) * "]  T=" *
                string(T_val) * " rep=" * string(rep) *
                "  elapsed=" * string(round(elapsed_s; digits=1)) * "s" *
                "  ETA=" * string(round(eta_s; digits=1)) * "s")
    end

    all_results = vcat(results...)

    outpath = joinpath(@__DIR__, "..", "data", "sim1_berke.jld2")
    jldsave(outpath; results=all_results)

    total_s = Dates.value(now() - t_start) / 1000
    log_msg("Saved results to " * outpath)
    log_msg("Done! Total time: " * string(round(total_s; digits=1)) * "s")
    close(logfile)
end

main()
