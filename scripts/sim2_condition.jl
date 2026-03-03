"""
Simulation 2: Condition Number Prediction

Tests the novel prediction that transition width scales with kappa.
6 task families (Gaussian iso, Gaussian kappa in {10,100,1000}, Beta narrow, Beta wide),
M=11 primary + M=2 comparator, 13 T-values, 30 reps each.

Launch with: julia -t 16 --project=. scripts/sim2_condition.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using DataFrames
using JLD2
using LinearAlgebra
using Random
using Dates

const N = 11
const K = 500
const EPSILON = 0.001
const N_GEN = 5000
const C = 2.0
const B = 1.0
const T_VALUES = [1, 2, 3, 5, 8, 11, 15, 20, 30, 50, 100, 200, 500]
const N_REPS = 30
const M_VALUES = [11, 2]

function make_gaussian_sigma(N::Int, kappa_target::Float64)
    if kappa_target <= 1.0
        return Matrix{Float64}(I, N, N)
    end
    r = kappa_target^(1.0 / (N - 1))
    diag_vals = [r^(-(i - 1)) for i in 1:N]
    # Scale so max eigenvalue is 1, min is 1/kappa_target
    return diagm(diag_vals)
end

function build_families(N::Int, B::Float64)
    families = Dict{String, TaskSampler}()

    # Gaussian isotropic (kappa = 1)
    families["gauss_iso"] = GaussianTaskSampler(N, B;
        Sigma_c=Matrix{Float64}(I, N, N))

    # Gaussian with controlled kappa
    for kappa in [10.0, 100.0, 1000.0]
        label = "gauss_k" * string(Int(kappa))
        Sigma_c = make_gaussian_sigma(N, kappa)
        families[label] = GaussianTaskSampler(N, B; Sigma_c=Sigma_c)
    end

    # Beta narrow (low diversity, correlated tasks)
    families["beta_narrow"] = BetaTaskSampler(N, B;
        a_range=(1.0, 3.0), b_range=(1.0, 3.0))

    # Beta wide (high diversity)
    families["beta_wide"] = BetaTaskSampler(N, B;
        a_range=(0.1, 20.0), b_range=(0.1, 20.0))

    return families
end

function main()
    t_start = now()

    families = build_families(N, B)
    family_names = sort(collect(keys(families)))

    # Build flat work items: (family_name, M_val, T_val, rep)
    work_items = Tuple{String, Int, Int, Int}[]
    for family_name in family_names
        for M_val in M_VALUES
            for T_val in T_VALUES
                for rep in 1:N_REPS
                    push!(work_items, (family_name, M_val, T_val, rep))
                end
            end
        end
    end

    n_work = length(work_items)

    # Per-work-item checkpointing for crash-safe resume
    chunk_dir = joinpath(@__DIR__, "..", "data", "sim2_condition_chunks")
    mkpath(chunk_dir)

    chunk_path(family_name::String, M_val::Int, T_val::Int, rep::Int) =
        joinpath(chunk_dir,
                 family_name * "__M" * string(M_val) *
                 "__T" * string(T_val) * "__rep" * string(rep) * ".jld2")

    precomputed = 0
    for (family_name, M_val, T_val, rep) in work_items
        if isfile(chunk_path(family_name, M_val, T_val, rep))
            precomputed += 1
        end
    end

    io_lock = ReentrantLock()
    counter = Threads.Atomic{Int}(0)

    logpath = joinpath(@__DIR__, "..", "data", "sim2_condition.log")
    mkpath(dirname(logpath))
    logfile = open(logpath, "a")

    function log_msg(msg::String)
        lock(io_lock) do
            line = Dates.format(now(), "HH:MM:SS") * "  " * msg
            println(line)
            println(logfile, line)
            flush(logfile)
        end
    end

    log_msg("sim2_condition started: " * string(n_work) * " work items, " *
            string(Threads.nthreads()) * " threads")
    log_msg("Families: " * join(family_names, ", ") *
            "  M_values: " * string(M_VALUES))
    log_msg("Resume status: " * string(precomputed) * "/" * string(n_work) *
            " chunk files already present")

    Threads.@threads for idx in eachindex(work_items)
        family_name, M_val, T_val, rep = work_items[idx]
        out_chunk = chunk_path(family_name, M_val, T_val, rep)

        if isfile(out_chunk)
            # Already computed in a previous run; count as completed.
            prev = Threads.atomic_add!(counter, 1)
            done = prev + 1
            if (done % 100 == 0) || (done == n_work)
                elapsed_s = Dates.value(now() - t_start) / 1000
                eta_s = done < n_work ? elapsed_s * (n_work - done) / done : 0.0
                log_msg("[" * string(done) * "/" * string(n_work) * "]  " *
                        "(cached)  elapsed=" * string(round(elapsed_s; digits=1)) * "s" *
                        "  ETA=" * string(round(eta_s; digits=1)) * "s")
            end
            continue
        end

        sampler = families[family_name]
        seed = hash((family_name, M_val, T_val, rep)) % 1000000
        params = EvolutionParams(N, M_val, K, T_val, N_GEN, EPSILON, C, B)

        history = run_evolution(params, sampler;
                                seed=Int(seed), show_progress=false)
        history[!, :T] .= T_val
        history[!, :M] .= M_val
        history[!, :rep] .= rep
        history[!, :family] .= family_name

        # Theoretical predictions
        rng = MersenneTwister(Int(seed))
        F_ref = build_task_matrix(rng, sampler, T_val)
        preds = theoretical_predictions(F_ref, fill(1.0 / N, N), C)
        history[!, :pred_kappa] .= preds.kappa
        history[!, :pred_k_T] .= preds.k_T

        # Atomic write to avoid partial/corrupt checkpoint files on interruption.
        tmp_chunk = out_chunk * ".tmp"
        jldsave(tmp_chunk; result=history)
        mv(tmp_chunk, out_chunk; force=true)

        prev = Threads.atomic_add!(counter, 1)
        done = prev + 1
        elapsed_s = Dates.value(now() - t_start) / 1000
        eta_s = done < n_work ? elapsed_s * (n_work - done) / done : 0.0
        log_msg("[" * string(done) * "/" * string(n_work) * "]  " *
                family_name * " M=" * string(M_val) *
                " T=" * string(T_val) * " rep=" * string(rep) *
                "  elapsed=" * string(round(elapsed_s; digits=1)) * "s" *
                "  ETA=" * string(round(eta_s; digits=1)) * "s")
    end

    # Build lightweight per-work-item summary (memory-safe), and optionally
    # materialize full trajectories in batched part files.
    missing = Tuple{String, Int, Int, Int}[]
    summary_rows = NamedTuple{(:family, :M, :T, :rep, :final_mean_risk,
                               :final_frac_eco_veridical, :final_frac_full_veridical,
                               :final_mean_fitness, :pred_kappa, :pred_k_T),
                              Tuple{String, Int, Int, Int, Float64, Float64,
                                    Float64, Float64, Float64, Float64}}[]
    n_found = 0

    for (family_name, M_val, T_val, rep) in work_items
        path = chunk_path(family_name, M_val, T_val, rep)
        if !isfile(path)
            push!(missing, (family_name, M_val, T_val, rep))
            continue
        end
        n_found += 1
        hist = load(path, "result")
        last = hist[end, :]
        push!(summary_rows, (
            family=family_name,
            M=M_val,
            T=T_val,
            rep=rep,
            final_mean_risk=Float64(last.mean_risk),
            final_frac_eco_veridical=Float64(last.frac_eco_veridical),
            final_frac_full_veridical=Float64(last.frac_full_veridical),
            final_mean_fitness=Float64(last.mean_fitness),
            pred_kappa=Float64(last.pred_kappa),
            pred_k_T=Float64(last.pred_k_T)
        ))
    end

    if !isempty(missing)
        log_msg("WARNING: Missing " * string(length(missing)) *
                " chunk files after run")
    end

    summary_df = DataFrame(summary_rows)
    summary_path = joinpath(@__DIR__, "..", "data", "sim2_condition_summary.jld2")
    jldsave(summary_path; summary=summary_df, missing=missing)

    manifest_path = joinpath(@__DIR__, "..", "data", "sim2_condition_manifest.jld2")
    jldsave(manifest_path;
            chunk_dir=chunk_dir,
            n_work=n_work,
            n_found=n_found,
            n_missing=length(missing),
            missing=missing)

    # Optional: write full trajectory aggregate in parts to avoid OOM.
    # Enable with: SIM2_BUILD_FULL_AGG=1 julia ... scripts/sim2_condition.jl
    if get(ENV, "SIM2_BUILD_FULL_AGG", "0") == "1"
        parts_dir = joinpath(@__DIR__, "..", "data", "sim2_condition_agg_parts")
        mkpath(parts_dir)
        batch_size = 60
        part_idx = 0
        batch = DataFrame[]

        for (family_name, M_val, T_val, rep) in work_items
            path = chunk_path(family_name, M_val, T_val, rep)
            isfile(path) || continue
            push!(batch, load(path, "result"))
            if length(batch) >= batch_size
                part_idx += 1
                part_df = vcat(batch...)
                part_path = joinpath(parts_dir,
                                     "sim2_condition_part_" * lpad(string(part_idx), 4, "0") * ".jld2")
                jldsave(part_path; results=part_df)
                empty!(batch)
                log_msg("Wrote aggregate part " * string(part_idx) *
                        " (" * string(batch_size) * " chunks)")
            end
        end
        if !isempty(batch)
            part_idx += 1
            part_df = vcat(batch...)
            part_path = joinpath(parts_dir,
                                 "sim2_condition_part_" * lpad(string(part_idx), 4, "0") * ".jld2")
            jldsave(part_path; results=part_df)
            log_msg("Wrote aggregate part " * string(part_idx) *
                    " (final partial batch)")
        end
        log_msg("Full aggregate written in parts at " * parts_dir)
    else
        log_msg("Skipping full trajectory aggregate to avoid OOM " *
                "(set SIM2_BUILD_FULL_AGG=1 to enable part-wise export)")
    end

    total_s = Dates.value(now() - t_start) / 1000
    log_msg("Saved summary to " * summary_path)
    log_msg("Saved manifest to " * manifest_path)
    log_msg("Done! Total time: " * string(round(total_s; digits=1)) * "s")
    close(logfile)
end

main()
