"""
Theoretical quantities: task distance, separation margin, equivalence classes,
condition number, and theoretical predictions.
"""

function task_distance_matrix(F::Matrix{Float64})
    T, N = size(F)
    sigma2 = zeros(N, N)
    @inbounds for i in 1:N, j in (i+1):N
        d = 0.0
        for t in 1:T
            diff = F[t, i] - F[t, j]
            d += diff * diff
        end
        d /= T
        sigma2[i, j] = d
        sigma2[j, i] = d
    end
    return sigma2
end

function separation_margin(sigma2::Matrix{Float64})
    N = size(sigma2, 1)
    delta = Inf
    for i in 1:N, j in (i+1):N
        if sigma2[i, j] < delta
            delta = sigma2[i, j]
        end
    end
    return delta
end

function equivalence_classes(sigma2::Matrix{Float64}, epsilon_tol::Float64)
    N = size(sigma2, 1)
    # Union-find
    parent = collect(1:N)

    function find_root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function union!(a, b)
        ra = find_root(a)
        rb = find_root(b)
        if ra != rb
            parent[ra] = rb
        end
    end

    for i in 1:N, j in (i+1):N
        if sigma2[i, j] <= epsilon_tol
            union!(i, j)
        end
    end

    roots = Set{Int}()
    for i in 1:N
        push!(roots, find_root(i))
    end
    return length(roots)
end

function condition_number(F::Matrix{Float64})
    T = size(F, 1)
    G = Symmetric(F' * F / T)
    eigvals_G = eigvals(G)
    pos_eigvals = filter(>(1e-12), eigvals_G)
    if isempty(pos_eigvals)
        return Inf
    end
    return maximum(pos_eigvals) / minimum(pos_eigvals)
end

function theoretical_predictions(F::Matrix{Float64}, pi::Vector{Float64}, C::Float64;
                                  epsilon_tol::Float64=1e-10)
    T, N = size(F)
    sigma2 = task_distance_matrix(F)
    pi_min = minimum(pi)
    delta_mu = separation_margin(sigma2)
    k_T = equivalence_classes(sigma2, epsilon_tol)

    fitness_gap_lower = T * pi_min^2 * delta_mu
    convergence_rate_lower = pi_min^2 * delta_mu / (2 * C)

    G = Symmetric(F' * F / T)
    eigvals_G = eigvals(G)
    pos_eigvals = filter(>(1e-12), eigvals_G)
    kappa = isempty(pos_eigvals) ? Inf : maximum(pos_eigvals) / minimum(pos_eigvals)

    return TheoreticalPredictions(
        delta_mu,
        fitness_gap_lower,
        convergence_rate_lower,
        kappa,
        length(pos_eigvals),
        k_T
    )
end

function optimal_partition_swap(F::Matrix{Float64}, pi::Vector{Float64}, M::Int;
                                 rng::AbstractRNG=Random.default_rng(),
                                 n_restarts::Int=100, max_iter::Int=1000)
    # Greedy swap search for optimal M-partition minimising multi-task Bayes risk.
    # Used in Simulation 3 for comparison with evolved partitions.
    T, N = size(F)
    best_p = nothing
    best_risk = Inf

    for _ in 1:n_restarts
        p = [rand(rng, 1:M) for _ in 1:N]
        current_risk = multi_task_risk(p, F, pi, M)
        improved = true
        iter = 0
        while improved && iter < max_iter
            improved = false
            iter += 1
            for i in 1:N
                old_val = p[i]
                for m in 1:M
                    m == old_val && continue
                    p[i] = m
                    new_risk = multi_task_risk(p, F, pi, M)
                    if new_risk < current_risk
                        current_risk = new_risk
                        improved = true
                    else
                        p[i] = old_val
                    end
                end
            end
        end
        if current_risk < best_risk
            best_risk = current_risk
            best_p = copy(p)
        end
    end
    return best_p, best_risk
end
