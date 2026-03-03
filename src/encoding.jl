"""
Encoding operations: creation, mutation, and property checks.
"""

function random_encoding(rng::AbstractRNG, N::Int, M::Int)
    return [rand(rng, 1:M) for _ in 1:N]
end

function random_encoding!(rng::AbstractRNG, p::Vector{Int}, M::Int)
    @inbounds for i in eachindex(p)
        p[i] = rand(rng, 1:M)
    end
    return p
end

function mutate(rng::AbstractRNG, p::Vector{Int}, M::Int, epsilon::Float64)
    p_new = copy(p)
    for i in eachindex(p_new)
        if rand(rng) < epsilon
            p_new[i] = rand(rng, 1:M)
        end
    end
    return p_new
end

function mutate_into!(rng::AbstractRNG, dst::Vector{Int}, src::Vector{Int},
                      M::Int, epsilon::Float64)
    @inbounds for i in eachindex(src)
        if rand(rng) < epsilon
            dst[i] = rand(rng, 1:M)
        else
            dst[i] = src[i]
        end
    end
    return dst
end

function is_injective(p::Vector{Int})
    return length(unique(p)) == length(p)
end

function used_complexity(p::Vector{Int})
    return length(unique(p))
end

function is_ecologically_veridical(p::Vector{Int}, F::Matrix{Float64}, epsilon_tol::Float64)
    T = size(F, 1)
    N = length(p)
    @inbounds for i in 1:N, j in (i+1):N
        if p[i] == p[j]
            d2 = 0.0
            for t in 1:T
                diff = F[t, i] - F[t, j]
                d2 += diff * diff
            end
            d2 /= T
            if d2 > epsilon_tol
                return false
            end
        end
    end
    return true
end

function is_ecologically_veridical_cells(cells::Vector{Vector{Int}},
                                          F::Matrix{Float64},
                                          epsilon_tol::Float64)
    T = size(F, 1)
    @inbounds for cell in cells
        nc = length(cell)
        nc <= 1 && continue
        for ci in 1:(nc-1), cj in (ci+1):nc
            i = cell[ci]
            j = cell[cj]
            d2 = 0.0
            for t in 1:T
                diff = F[t, i] - F[t, j]
                d2 += diff * diff
            end
            d2 /= T
            if d2 > epsilon_tol
                return false
            end
        end
    end
    return true
end
