"""
Bayes risk computation for encodings under tasks.
"""

function precompute_cells(p::Vector{Int}, M::Int)
    cells = [Int[] for _ in 1:M]
    @inbounds for (w, x) in enumerate(p)
        push!(cells[x], w)
    end
    return cells
end

function bayes_risk(p::Vector{Int}, f::AbstractVector{Float64}, pi::Vector{Float64}, M::Int)
    R = 0.0
    for x in 1:M
        cell = findall(==(x), p)
        isempty(cell) && continue
        pi_cell = sum(pi[w] for w in cell)
        f_hat = sum(pi[w] * f[w] for w in cell) / pi_cell
        for w in cell
            R += pi[w] * (f[w] - f_hat)^2
        end
    end
    return R
end

function multi_task_risk(p::Vector{Int}, F::Matrix{Float64}, pi::Vector{Float64}, M::Int)
    cells = precompute_cells(p, M)
    return multi_task_risk_cells(cells, F, pi)
end

function multi_task_risk_cells(cells::Vector{Vector{Int}}, F::Matrix{Float64},
                                pi::Vector{Float64})
    T = size(F, 1)
    total = 0.0
    @inbounds for t in 1:T
        for cell in cells
            isempty(cell) && continue
            pi_cell = 0.0
            for w in cell
                pi_cell += pi[w]
            end
            f_hat = 0.0
            for w in cell
                f_hat += pi[w] * F[t, w]
            end
            f_hat /= pi_cell
            for w in cell
                diff = F[t, w] - f_hat
                total += pi[w] * diff * diff
            end
        end
    end
    return total / T
end

function compute_fitness(risk::Float64, T::Int, C::Float64)
    return T * C - T * risk
end
