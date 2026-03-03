"""
Wright-Fisher evolutionary dynamics engine.

Optimised for minimal allocation: double-buffered population, precomputed
cells, inlined Price/variance statistics, pre-allocated history vectors.
"""

function run_evolution(params::EvolutionParams, sampler::TaskSampler;
                       seed::Int=42, epsilon_tol::Float64=1e-10,
                       show_progress::Bool=true, return_population::Bool=false)
    (; N, M, K, T, n_generations, epsilon, C, B) = params
    rng = MersenneTwister(seed)
    pi = fill(1.0 / N, N)

    # Double-buffer population: avoids allocating K new vectors per generation
    pop_a = [Vector{Int}(undef, N) for _ in 1:K]
    pop_b = [Vector{Int}(undef, N) for _ in 1:K]
    for k in 1:K
        random_encoding!(rng, pop_a[k], M)
    end
    current_pop = pop_a
    next_pop = pop_b

    # Pre-allocate history columns
    h_generation       = Vector{Float64}(undef, n_generations)
    h_mean_risk        = Vector{Float64}(undef, n_generations)
    h_var_risk         = Vector{Float64}(undef, n_generations)
    h_mean_fitness     = Vector{Float64}(undef, n_generations)
    h_var_fitness      = Vector{Float64}(undef, n_generations)
    h_frac_full        = Vector{Float64}(undef, n_generations)
    h_frac_eco         = Vector{Float64}(undef, n_generations)
    h_mean_complexity  = Vector{Float64}(undef, n_generations)
    h_k_T             = Vector{Float64}(undef, n_generations)
    h_delta_mu         = Vector{Float64}(undef, n_generations)
    h_price_sel        = Vector{Float64}(undef, n_generations)
    h_price_cov        = Vector{Float64}(undef, n_generations)
    h_obs_delta        = Vector{Float64}(undef, n_generations)

    # Pre-allocate per-generation work vectors
    risks = Vector{Float64}(undef, K)
    fitnesses = Vector{Float64}(undef, K)

    prev_mean_risk = NaN
    prog = show_progress ? Progress(n_generations; desc="Evolving: ") : nothing

    for gen in 1:n_generations
        # Sample tasks for this generation
        F = build_task_matrix(rng, sampler, T)

        # Compute risk/fitness only for unique encodings, caching cells
        risk_map = Dict{Vector{Int}, Float64}()
        fitness_map = Dict{Vector{Int}, Float64}()
        cells_map = Dict{Vector{Int}, Vector{Vector{Int}}}()

        for k in 1:K
            p = current_pop[k]
            if !haskey(risk_map, p)
                cells = precompute_cells(p, M)
                cells_map[p] = cells
                R = multi_task_risk_cells(cells, F, pi)
                risk_map[p] = R
                fitness_map[p] = compute_fitness(R, T, C)
            end
        end

        # Fill per-individual arrays and compute means in one pass
        mean_r = 0.0
        mean_w = 0.0
        for k in 1:K
            p = current_pop[k]
            r = risk_map[p]
            w = fitness_map[p]
            risks[k] = r
            fitnesses[k] = w
            mean_r += r
            mean_w += w
        end
        mean_r /= K
        mean_w /= K

        # Theoretical quantities from this generation's task matrix
        sigma2 = task_distance_matrix(F)
        dm = separation_margin(sigma2)
        kt = equivalence_classes(sigma2, epsilon_tol)

        # Inlined Price equation + variance computation (single pass)
        var_r = 0.0
        var_w = 0.0
        cov_wr = 0.0
        for k in 1:K
            dr = risks[k] - mean_r
            dw = fitnesses[k] - mean_w
            var_r += dr * dr
            var_w += dw * dw
            cov_wr += dw * dr
        end
        var_r /= K
        var_w /= K
        cov_wr /= K
        price_sel = -cov_wr / mean_w

        # Observed risk change
        observed_delta = isnan(prev_mean_risk) ? 0.0 : mean_r - prev_mean_risk

        # Population statistics with cached cells for eco-veridical check
        eco_map = Dict{Vector{Int}, Bool}()
        n_full = 0
        n_eco = 0
        total_complexity = 0.0
        for k in 1:K
            p = current_pop[k]
            if is_injective(p)
                n_full += 1
            end
            if !haskey(eco_map, p)
                eco_map[p] = is_ecologically_veridical_cells(cells_map[p], F, epsilon_tol)
            end
            if eco_map[p]
                n_eco += 1
            end
            total_complexity += used_complexity(p)
        end

        # Record history by index (no push! overhead)
        h_generation[gen]      = Float64(gen)
        h_mean_risk[gen]       = mean_r
        h_var_risk[gen]        = var_r
        h_mean_fitness[gen]    = mean_w
        h_var_fitness[gen]     = var_w
        h_frac_full[gen]       = n_full / K
        h_frac_eco[gen]        = n_eco / K
        h_mean_complexity[gen] = total_complexity / K
        h_k_T[gen]            = Float64(kt)
        h_delta_mu[gen]        = dm
        h_price_sel[gen]       = price_sel
        h_price_cov[gen]       = cov_wr
        h_obs_delta[gen]       = observed_delta

        prev_mean_risk = mean_r

        # Selection: sample parent indices proportional to fitness
        total_fitness = sum(fitnesses)
        w = Weights(fitnesses ./ total_fitness)
        parent_indices = sample(rng, 1:K, w, K; replace=true)

        # Mutation: write into next_pop buffer (no new vector allocation)
        for k in 1:K
            mutate_into!(rng, next_pop[k], current_pop[parent_indices[k]], M, epsilon)
        end

        # Swap buffers
        current_pop, next_pop = next_pop, current_pop

        if prog !== nothing
            next!(prog)
        end
    end

    history = DataFrame(
        generation=h_generation, mean_risk=h_mean_risk, var_risk=h_var_risk,
        mean_fitness=h_mean_fitness, var_fitness=h_var_fitness,
        frac_full_veridical=h_frac_full, frac_eco_veridical=h_frac_eco,
        mean_used_complexity=h_mean_complexity, k_T=h_k_T, delta_mu=h_delta_mu,
        price_selection_term=h_price_sel, price_cov_wr=h_price_cov,
        observed_delta_risk=h_obs_delta
    )

    if return_population
        final_pop = [copy(p) for p in current_pop]
        return (history, final_pop)
    end
    return history
end
