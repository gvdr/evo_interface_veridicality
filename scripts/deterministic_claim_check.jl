"""
Deterministic small-state verifier of the main claim.

Exact setup (no Monte Carlo task sampling):
- N=4 world states (2-bit labels).
- Task family = bit-readout tasks; task diversity m_active in {1,2}.
- Enumerate all encodings p: W -> X exactly.
- Compute exact expected Bayes risk R(p) under uniform task distribution.
- Evolve deterministic replicator-mutator on encoding frequencies.

This isolates the theorem logic:
1) If capacity M >= k_mu (ecological complexity), ecological veridicality is feasible and selected.
2) If M < k_mu, ecological veridicality is infeasible and equilibrium risk remains positive.

Run:
  julia --project=. scripts/deterministic_claim_check.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EvoPaper
using CairoMakie
using DataFrames
using JLD2
using LinearAlgebra
using Statistics

function all_encodings(N::Int, M::Int)
    total = M^N
    encs = Vector{Vector{Int}}(undef, total)
    p = ones(Int, N)
    for idx in 1:total
        encs[idx] = copy(p)
        for i in 1:N
            p[i] += 1
            if p[i] <= M
                break
            end
            p[i] = 1
        end
    end
    return encs
end

function bit_tasks(N::Int)
    # N=4 interpreted as states 0..3 with two bits.
    tasks = zeros(Float64, 2, N)
    for i in 1:N
        w = i - 1
        tasks[1, i] = ((w >>> 0) & 0x01) == 0x01 ? 1.0 : 0.0
        tasks[2, i] = ((w >>> 1) & 0x01) == 0x01 ? 1.0 : 0.0
    end
    return tasks
end

function task_distance_from_F(F::Matrix{Float64})
    return task_distance_matrix(F)
end

function is_eco_veridical_exact(p::Vector{Int}, sigma2::Matrix{Float64}; tol::Float64=1e-12)
    N = length(p)
    for i in 1:N, j in (i + 1):N
        if p[i] == p[j] && sigma2[i, j] > tol
            return false
        end
    end
    return true
end

function mutation_prob(p::Vector{Int}, q::Vector{Int}, M::Int, epsilon::Float64)
    # Matches mutate_into!: with prob epsilon, redraw uniformly in 1:M (can stay same).
    pr = 1.0
    psame = (1 - epsilon) + epsilon / M
    pdiff = epsilon / M
    @inbounds for i in eachindex(p)
        pr *= (p[i] == q[i]) ? psame : pdiff
    end
    return pr
end

function build_mutation_kernel(encs::Vector{Vector{Int}}, M::Int, epsilon::Float64)
    n = length(encs)
    Q = Matrix{Float64}(undef, n, n) # parent row -> child col
    for a in 1:n
        for b in 1:n
            Q[a, b] = mutation_prob(encs[a], encs[b], M, epsilon)
        end
        rs = sum(Q[a, :])
        Q[a, :] ./= rs
    end
    return Q
end

function run_deterministic_replicator_mutator(W::Vector{Float64}, Q::Matrix{Float64};
                                              nsteps::Int=4000)
    n = length(W)
    x = fill(1.0 / n, n) # uniform start
    hist_risk = Vector{Float64}(undef, nsteps + 1)
    hist_mass = Matrix{Float64}(undef, nsteps + 1, n) # optional diagnostics
    hist_mass[1, :] .= x
    hist_risk[1] = NaN
    for g in 1:nsteps
        y = transpose(Q) * (W .* x)
        x = y ./ sum(y)
        hist_mass[g + 1, :] .= x
    end
    return x, hist_mass
end

function scenario_metrics(M::Int, m_active::Int; epsilon::Float64=1e-3, C::Float64=2.0)
    N = 4
    pi = fill(1.0 / N, N)

    # Ecology: first m_active bit tasks.
    F_full = bit_tasks(N)
    F = F_full[1:m_active, :]
    sigma2 = task_distance_from_F(F)

    encs = all_encodings(N, M)
    n = length(encs)
    risks = zeros(Float64, n)
    eco = falses(n)
    full = falses(n)

    for i in 1:n
        p = encs[i]
        risks[i] = multi_task_risk(p, F, pi, M)
        eco[i] = is_eco_veridical_exact(p, sigma2)
        full[i] = is_injective(p)
    end

    W = C .- risks
    @assert all(W .> 0)
    Q = build_mutation_kernel(encs, M, epsilon)
    x_eq, _ = run_deterministic_replicator_mutator(W, Q)

    mean_risk = dot(x_eq, risks)
    eco_mass = sum(x_eq[eco])
    full_mass = sum(x_eq[full])
    min_risk = minimum(risks)
    eq_gap = mean_risk - min_risk
    k_mu = equivalence_classes(sigma2, 1e-12)

    return (
        M=M, m_active=m_active, k_mu=k_mu,
        mean_risk=mean_risk, min_risk=min_risk, eq_gap=eq_gap,
        eco_mass=eco_mass, full_mass=full_mass
    )
end

function main()
    out_data = joinpath(@__DIR__, "..", "data", "deterministic_claim_check.jld2")
    out_fig_dir = joinpath(@__DIR__, "..", "figures", "quick_checks")
    out_fig_pdf = joinpath(out_fig_dir, "deterministic_claim_check.pdf")
    out_fig_png = joinpath(out_fig_dir, "deterministic_claim_check.png")
    mkpath(dirname(out_data))
    mkpath(out_fig_dir)

    scenarios = [(2, 1), (2, 2), (4, 1), (4, 2)]
    rows = NamedTuple[]
    for (M, m_active) in scenarios
        push!(rows, scenario_metrics(M, m_active))
    end
    df = DataFrame(rows)
    jldsave(out_data; summary=df)

    labels = ["M=2,m=1", "M=2,m=2", "M=4,m=1", "M=4,m=2"]
    x = 1:length(labels)

    fig = Figure(size=(1200, 420), fontsize=14)

    ax1 = Axis(fig[1, 1], title="Equilibrium Ecological-Veridical Mass",
        xlabel="Scenario", ylabel="Mass on eco-veridical encodings",
        xticks=(x, labels))
    barplot!(ax1, x, df.eco_mass, color=[:firebrick, :firebrick, :dodgerblue, :dodgerblue])
    ylims!(ax1, -0.02, 1.02)

    ax2 = Axis(fig[1, 2], title="Equilibrium Full-Veridical Mass",
        xlabel="Scenario", ylabel="Mass on injective encodings",
        xticks=(x, labels))
    barplot!(ax2, x, df.full_mass, color=[:firebrick, :firebrick, :dodgerblue, :dodgerblue])
    ylims!(ax2, -0.02, 1.02)

    ax3 = Axis(fig[1, 3], title="Equilibrium Mean Risk",
        xlabel="Scenario", ylabel="Mean risk at equilibrium",
        xticks=(x, labels))
    barplot!(ax3, x, df.mean_risk, color=[:firebrick, :firebrick, :dodgerblue, :dodgerblue])

    save(out_fig_pdf, fig; pt_per_unit=1)
    save(out_fig_png, fig; px_per_unit=3)

    println("Saved deterministic claim-check data: ", out_data)
    println("Saved deterministic claim-check figure: ", out_fig_pdf)
    println()
    println(df)
end

main()
