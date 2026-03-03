"""
Task sampling: Beta, Gaussian, and Weighted task families.
"""

function sample_task(rng::AbstractRNG, s::BetaTaskSampler)
    a = s.a_lo + rand(rng) * (s.a_hi - s.a_lo)
    b = s.b_lo + rand(rng) * (s.b_hi - s.b_lo)
    d = Beta(a, b)
    f = Vector{Float64}(undef, s.N)
    for i in 1:s.N
        x_i = i / (s.N + 1)
        f[i] = pdf(d, x_i)
    end
    fmax = maximum(f)
    if fmax > 0
        for i in eachindex(f)
            f[i] = f[i] / fmax * s.B
        end
    end
    return f
end

function sample_task(rng::AbstractRNG, s::GaussianTaskSampler)
    c = rand(rng, s.mvn)
    f = s.phi' * c  # N-vector
    for i in eachindex(f)
        f[i] = clamp(f[i], -s.B, s.B)
    end
    return f
end

function sample_task(rng::AbstractRNG, s::WeightedTaskSampler)
    total = s.cumweights[end]
    r = rand(rng) * total
    idx = searchsortedfirst(s.cumweights, r)
    if idx > length(s.cumweights)
        idx = length(s.cumweights)
    end
    return sample_task(rng, s.category_samplers[idx])
end

function build_task_matrix(rng::AbstractRNG, sampler::TaskSampler, T::Int)
    N = sampler.N
    F = Matrix{Float64}(undef, T, N)
    for t in 1:T
        f = sample_task(rng, sampler)
        for i in 1:N
            F[t, i] = f[i]
        end
    end
    return F
end
