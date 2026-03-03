"""
Small numerical checks for core lemmas/theorems in interface_truth.md.

Run with:
  julia --project=. scripts/check_theorems.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using LinearAlgebra
using Random
using Statistics
using Test

function pairwise_risk_formula(p::Vector{Int}, F::Matrix{Float64},
                               pi::Vector{Float64}, M::Int)
    cells = precompute_cells(p, M)
    sigma2 = task_distance_matrix(F)
    R = 0.0
    for cell in cells
        isempty(cell) && continue
        pi_cell = sum(pi[i] for i in cell)
        accum = 0.0
        for i in cell, j in cell
            accum += pi[i] * pi[j] * sigma2[i, j]
        end
        R += 0.5 * accum / pi_cell
    end
    return R
end

function min_pairwise_distance(sigma2::Matrix{Float64})
    N = size(sigma2, 1)
    d = Inf
    for i in 1:N, j in (i + 1):N
        d = min(d, sigma2[i, j])
    end
    return d
end

function all_encodings(N::Int, M::Int)
    total = M^N
    encs = Vector{Vector{Int}}(undef, total)
    p = ones(Int, N)
    for idx in 1:total
        encs[idx] = copy(p)
        for k in 1:N
            p[k] += 1
            if p[k] <= M
                break
            end
            p[k] = 1
        end
    end
    return encs
end

function label_permute(p::Vector{Int}, sigma::Vector{Int})
    q = similar(p)
    @inbounds for i in eachindex(p)
        q[i] = sigma[p[i]]
    end
    return q
end

function normalize_prob(x::Vector{Float64})
    s = sum(x)
    s > 0 || error("Cannot normalize zero vector.")
    return x ./ s
end

function iterate_quasispecies(A::Matrix{Float64}, x0::Vector{Float64}, nsteps::Int)
    x = normalize_prob(copy(x0))
    history = Vector{Vector{Float64}}(undef, nsteps + 1)
    history[1] = copy(x)
    for g in 1:nsteps
        y = A * x
        x = normalize_prob(y)
        history[g + 1] = copy(x)
    end
    return history
end

function principal_right_eigenvector(A::Matrix{Float64})
    vals, vecs = eigen(A)
    idx = argmax(real.(vals))
    v = real.(vecs[:, idx])
    v = abs.(v)
    return normalize_prob(v)
end

@testset "Numerical theorem checks" begin
    @testset "Lemma 2.9 pairwise Bayes risk identity" begin
        rng = MersenneTwister(123)
        for _ in 1:20
            N = 6
            M = 3
            T = 11
            F = randn(rng, T, N)
            pi = rand(rng, N)
            pi ./= sum(pi)
            p = random_encoding(rng, N, M)
            r_direct = multi_task_risk(p, F, pi, M)
            r_pairwise = pairwise_risk_formula(p, F, pi, M)
            @test isapprox(r_direct, r_pairwise; atol=1e-11, rtol=1e-10)
        end
    end

    @testset "Theorem 4.1(c) and relabel invariance on small exhaustive case" begin
        rng = MersenneTwister(777)
        N = 4
        M = 4
        T = 7
        F = randn(rng, T, N)
        pi = rand(rng, N)
        pi ./= sum(pi)
        pi_min = minimum(pi)
        sigma2 = task_distance_matrix(F)
        delta_mu = min_pairwise_distance(sigma2)
        @test delta_mu > 1e-10
        lower_bound = pi_min^2 * delta_mu

        encs = all_encodings(N, M)
        for p in encs
            r = multi_task_risk(p, F, pi, M)
            if is_injective(p)
                @test abs(r) <= 1e-12
            else
                @test r + 1e-12 >= lower_bound
            end

            sigma = [2, 4, 1, 3]
            p_perm = label_permute(p, sigma)
            r_perm = multi_task_risk(p_perm, F, pi, M)
            @test isapprox(r, r_perm; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Proposition 4.6 monotone empirical complexity k_T" begin
        rng = MersenneTwister(99)
        N = 8
        T_max = 40
        sampler = BetaTaskSampler(N, 1.0)
        F = build_task_matrix(rng, sampler, T_max)

        k_vals = Int[]
        for T in 1:T_max
            sigma2 = task_distance_matrix(Matrix(view(F, 1:T, :)))
            kT = equivalence_classes(sigma2, 1e-12)
            push!(k_vals, kT)
        end

        @test all(k_vals[t + 1] >= k_vals[t] for t in 1:(T_max - 1))
        @test all(1 <= k <= N for k in k_vals)
    end

    @testset "Theorems 7.1-7.3 Price identities (selection and mutation)" begin
        rng = MersenneTwister(31415)
        L = 6
        x = normalize_prob(rand(rng, L))
        R = rand(rng, L)
        T = 5
        C = 2.0
        W = T .* (C .- R)
        @test all(W .> 0)

        # Selection only: x' = (W .* x) / W̄
        W_bar = dot(x, W)
        x_sel = (W .* x) ./ W_bar
        R_bar = dot(x, R)
        R_bar_sel = dot(x_sel, R)
        var_R = dot(x, (R .- R_bar).^2)
        @test isapprox(R_bar_sel - R_bar, -T * var_R / W_bar; atol=1e-12, rtol=1e-11)

        # Full Price equation with transmission term
        Q = rand(rng, L, L)
        Q ./= sum(Q; dims=2)
        x_next = Q' * x_sel
        z_bar_next = dot(x_next, R)
        cov_WR = dot(x, (W .- W_bar) .* (R .- R_bar))
        delta_R = Q * R .- R
        transmission = dot(x, W .* delta_R)
        lhs = W_bar * (z_bar_next - R_bar)
        rhs = cov_WR + transmission
        @test isapprox(lhs, rhs; atol=1e-12, rtol=1e-11)

        # Fisher selection-only identity
        W_bar_sel = dot(x_sel, W)
        var_W = dot(x, (W .- W_bar).^2)
        @test isapprox(W_bar_sel - W_bar, var_W / W_bar; atol=1e-12, rtol=1e-11)
    end

    @testset "Theorem 7.4 style class-conditional convergence" begin
        # Two closed classes: K1={1,2} (higher spectral radius), K2={3,4};
        # state 5 transient feeding both.
        W = [1.4, 1.3, 1.05, 1.0, 1.1]
        Q = [
            0.90 0.10 0.00 0.00 0.00
            0.20 0.80 0.00 0.00 0.00
            0.00 0.00 0.85 0.15 0.00
            0.00 0.00 0.25 0.75 0.00
            0.70 0.00 0.30 0.00 0.00
        ]
        A = Q' * Diagonal(W)

        λ1 = maximum(abs.(eigvals(A[1:2, 1:2])))
        λ2 = maximum(abs.(eigvals(A[3:4, 3:4])))
        @test λ1 > λ2

        # Reachable to both classes from transient, should select dominant class K1.
        x0 = [0.0, 0.0, 0.0, 0.0, 1.0]
        hist = iterate_quasispecies(A, x0, 250)
        x_end = hist[end]
        @test sum(x_end[1:2]) > 0.999
        @test x_end[5] < 1e-10

        # If initial support is only K2 (and no path to K1), stays in K2.
        x0_k2 = [0.0, 0.0, 0.6, 0.4, 0.0]
        hist_k2 = iterate_quasispecies(A, x0_k2, 250)
        x_end_k2 = hist_k2[end]
        @test sum(x_end_k2[1:2]) < 1e-12
        @test sum(x_end_k2[3:4]) > 0.999
    end

    @testset "Remark 7.4.3 periodic dominant block behavior" begin
        W = [1.4, 1.0]
        Q = [0.0 1.0; 1.0 0.0]   # irreducible period-2 mutation
        A = Q' * Diagonal(W)
        hist = iterate_quasispecies(A, [0.61, 0.39], 500)

        # Two-cycle: x_{t+2} close to x_t on tail, but x_{t+1} differs.
        tail = hist[401:end]
        d2 = mean(norm(tail[i] - tail[max(i - 2, 1)]) for i in 3:length(tail))
        d1 = mean(norm(tail[i] - tail[i - 1]) for i in 2:length(tail))
        @test d2 < 1e-12
        @test d1 > 1e-3

        # Cesaro average converges to Perron profile.
        x_cesaro = vec(mean(reduce(hcat, tail); dims=2))
        v_pf = principal_right_eigenvector(A)
        @test norm(x_cesaro - v_pf) < 2e-2
    end
end

println("All numerical theorem checks passed.")
