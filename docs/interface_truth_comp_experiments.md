# Paper 1 Experiments

# Empirical Validation of "Between Interface and Truth"

## Overview

Three simulation studies to validate the theoretical predictions of the paper. All simulations implement the evolutionary perception game from Hoffman et al. (2015) and Berke et al. (2022), extended with measurements of quantities defined in our theory.

**Language:** Julia ≥ 1.10
**Libraries:** Distributions.jl, LinearAlgebra, StatsBase.jl, DataFrames.jl, CairoMakie.jl, JLD2.jl, ProgressMeter.jl

---

## Common Infrastructure

### 1. World and Encoding Representation

```
World states:  W = 1:N  (integers)
Prior:         π = ones(N) / N  (uniform unless otherwise specified)
Encoding:      p::Vector{Int} of length N, where p[w] ∈ 1:M
               p[w] = x means world state w maps to percept x
Admissible set: all functions p: W → X (not necessarily surjective)
Used complexity: m(p) = length(unique(p)) ≤ M
Full injectivity: p is injective iff length(unique(p)) == N (requires M ≥ N)
Ecological injectivity (empirical): p[w1] == p[w2] ⇒ σ²_T(w1,w2) ≈ 0 (tolerance ε_tol)
```

### 2. Task Generation

**Beta-function tasks (Berke replication):**
```
Parameters:  a ~ Uniform(0.5, 10),  b ~ Uniform(0.5, 10)
             (or Berke's original range — verify from their supplementary)
World states: evenly spaced on (0,1): x_i = i/(N+1) for i = 1,...,N
Task vector:  f[i] = pdf(Beta(a, b), x_i)
Normalise:    f = f / maximum(f) * B   (so f ∈ [0, B], we use B = 1)
```

**Gaussian-linear tasks (for Simulation 2 comparison):**
```
Feature vectors:  φ_i ∈ ℝ^D, one per world state (fixed)
                  Default: φ_i = e_i (standard basis), so D = N
Task coefficient: c ~ MvNormal(zeros(D), Σ_c)
Task vector:      f[i] = dot(c, φ_i)
Clamp:            f = clamp.(f, -B, B)
```

**Non-uniform μ (for Simulation 3):**
```
Define task categories with weights:
  Category 1 ("predation"): weight w₁, parameters (a,b) from range R₁
  Category 2 ("texture"):   weight w₂, parameters (a,b) from range R₂
Sample category with probability proportional to weights, then sample
task from that category's parameter range.
```

### 3. Bayes Risk Computation

For encoding p and task f:

```
function bayes_risk(p::Vector{Int}, f::Vector{Float64}, π::Vector{Float64}, M::Int)
    R = 0.0
    for x in 1:M
        cell = findall(==(x), p)
        isempty(cell) && continue
        π_cell = sum(π[cell])
        f_hat = sum(π[w] * f[w] for w in cell) / π_cell
        R += π_cell * sum(π[w]/π_cell * (f[w] - f_hat)^2 for w in cell)
    end
    return R
end
```

Multi-task Bayes risk: average over T sampled tasks.

### 4. Task Distance Matrix

```
function task_distance_matrix(F::Matrix{Float64})
    # F is T × N (rows = tasks, columns = world states)
    N = size(F, 2)
    σ² = zeros(N, N)
    for i in 1:N, j in i+1:N
        σ²[i,j] = mean((F[:,i] .- F[:,j]).^2)
        σ²[j,i] = σ²[i,j]
    end
    return σ²
end
```

From σ², extract:
- `δ_μ = minimum(σ²[i,j] for i in 1:N for j in i+1:N)` (separation margin)
- `k_T = number of equivalence classes` (cluster σ² with tolerance ε_tol ≈ 1e-10)
- Condition number: eigendecompose σ² (or F'F/T), take λ_max/λ_min of positive eigenvalues

### 5. Evolutionary Dynamics Engine

Wright-Fisher process matching Berke et al.:

```
struct EvolutionParams
    N::Int              # world states
    M::Int              # percepts
    K::Int              # population size
    T::Int              # tasks per generation
    n_generations::Int  # number of generations
    ε::Float64          # per-position mutation rate
    C::Float64          # payoff constant (ensures positive fitness)
    B::Float64          # task bound
end

function run_evolution(params, task_sampler; seed=42)
    (; N, M, K, T, n_generations, ε, C, B) = params
    rng = MersenneTwister(seed)

    population = [random_encoding(rng, N, M) for _ in 1:K]

    history = DataFrame(
        generation = Int[],
        mean_risk = Float64[],
        var_fitness = Float64[],
        frac_full_veridical = Float64[],
        frac_eco_veridical = Float64[],
        mean_used_complexity = Float64[],
        frac_best_partition = Float64[],
        mean_fitness = Float64[],
    )

    for gen in 1:n_generations
        tasks = [task_sampler(rng) for _ in 1:T]

        unique_encodings = unique(population)
        fitness_map = Dict{Vector{Int}, Float64}()
        risk_map = Dict{Vector{Int}, Float64}()
        π = fill(1.0/N, N)
        for p in unique_encodings
            R = mean(bayes_risk(p, f, π, M) for f in tasks)
            risk_map[p] = R
            fitness_map[p] = T * C - T * R
        end

        fitnesses = [fitness_map[p] for p in population]
        risks = [risk_map[p] for p in population]

        w_bar = mean(fitnesses)
        push!(history, (
            generation = gen,
            mean_risk = mean(risks),
            var_fitness = var(fitnesses),
            frac_full_veridical = count(p -> length(unique(p)) == N, population) / K,
            frac_eco_veridical = count(p -> is_ecologically_veridical(p, tasks, π, ε_tol), population) / K,
            mean_used_complexity = mean(length(unique(p)) for p in population),
            frac_best_partition = maximum(count(==(p), population) for p in unique_encodings) / K,
            mean_fitness = w_bar,
        ))

        weights = Weights(fitnesses ./ sum(fitnesses))
        parents = sample(rng, population, weights, K; replace=true)
        population = [mutate(rng, p, M, ε) for p in parents]
    end

    return history
end

function random_encoding(rng, N, M)
    # Sample uniformly over the function class Ω = X^W
    return [rand(rng, 1:M) for _ in 1:N]
end

function is_ecologically_veridical(p, tasks, π, ε_tol)
    # Empirical criterion: if two states are merged, their empirical task distance is ~0
    N = length(p)
    F = hcat(tasks...)'   # T × N
    for i in 1:N, j in i+1:N
        if p[i] == p[j]
            d2 = mean((F[:, i] .- F[:, j]).^2)
            if d2 > ε_tol
                return false
            end
        end
    end
    return true
end

function mutate(rng, p::Vector{Int}, M::Int, ε::Float64)
    p_new = copy(p)
    for i in eachindex(p_new)
        if rand(rng) < ε
            p_new[i] = rand(rng, 1:M)
        end
    end
    return p_new
end
```

### 6. Theoretical Predictions (for overlay on plots)

```
function theoretical_predictions(F::Matrix{Float64}, π::Vector{Float64}, C::Float64)
    T, N = size(F)
    σ² = task_distance_matrix(F)
    π_min = minimum(π)
    δ_μ = minimum(σ²[i,j] for i in 1:N for j in i+1:N if i ≠ j)

    Δw_lower_bound = T * π_min^2 * δ_μ
    γ_lower_bound = π_min^2 * δ_μ / (2 * C)

    G = F' * F / T
    eigvals_G = eigvals(Symmetric(G))
    pos_eigvals = filter(>(1e-12), eigvals_G)
    κ = length(pos_eigvals) > 0 ? maximum(pos_eigvals) / minimum(pos_eigvals) : Inf

    return (; δ_μ, Δw_lower_bound, γ_lower_bound, κ, effective_rank=length(pos_eigvals))
end
```

---

## Simulation 1: Berke Replication with Theoretical Overlay

### Purpose
Replicate Berke et al.'s main finding and test whether the *quantitative* predictions of our theory (fitness gap, convergence rate, separation margin, cascade staircase) hold.

### Parameters
```
N = 11, M = 2, K = 1000, ε = 0.001, n_gen = 5000, C = 2.0, B = 1.0
T ∈ [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

Beta task sampler:
  a ~ Uniform(0.5, 10), b ~ Uniform(0.5, 10)
  x_i = i / 12 for i = 1:11
  f[i] = pdf(Beta(a, b), x_i), normalised to [0, 1]
```

### Measurements (per T, per replicate)
**From simulation:** fraction at best encoding at gen 5000; mean Bayes risk trajectory R̄(gen); convergence generation τ (first gen where frac_best > 0.9); Var(w) trajectory.

**From theory:** empirical σ²(T), δ_μ(T), k_T, κ(T), predicted fitness gap π²_min·δ_μ, predicted convergence rate δ_μ/(2C).

### Replications
50 independent runs per T.

### Plots
- **1a:** Fraction ecological-veridical vs T (overlay on Berke-style transition curve)
- **1b:** Fraction full-veridical vs T (expected ~0 for M = 2, shown explicitly)
- **1c:** δ_μ vs T (new measurement, threshold diagnostics)
- **1d:** k_T vs T (staircase, Prop 4.6a)
- **1e:** Observed fitness gap vs π²_min·δ_μ (scatter, should be above y=x)
- **1f:** Convergence time τ vs 1/δ_μ (scatter, test linear relationship)
- **1g:** Price equation check: ΔR̄ vs -T·Var(R)/w̄ (single run, generation-by-generation)

---

## Simulation 2: Condition Number Prediction

### Purpose
Test the novel prediction that transition width scales with κ.

### Design
Three task family pairs, each varying κ while controlling other parameters:

**Pair A — Gaussian (closed-form control):**
- Low κ: Σ_c = I (isotropic), κ = 1
- High κ: Σ_c = diag(1, r, r², ..., r^{N-1}) with κ_target ∈ {10, 100, 1000}

**Pair B — Beta (biologically motivated):**
- Low κ: a, b ~ Uniform(1, 3) (narrow parameter range, correlated tasks)
- High κ: a, b ~ Uniform(0.1, 20) (wide range, diverse tasks)
- κ computed empirically from F'F/T

**Pair C — Mixed control:**
- Fixed beta tasks but subsample from principal components to control effective κ

### Parameters
```
N = 11, M = 11 (equal complexity), K = 500, ε = 0.001, n_gen = 5000
T ∈ [1, 2, 3, 5, 8, 11, 15, 20, 30, 50, 100, 200, 500]
```
Also run with M = 2 for comparison.

### Replications
30 runs per (task_family, T).

### Plots
- **2a:** Transition curves (fraction ecological-veridical vs T), one per family. Low κ = step; high κ = sigmoid.
- **2b:** Transition width (T₉₀ − T₁₀) vs κ using ecological-veridical fraction. Log-log, predict positive slope.
- **2c:** Beta families only (demonstrates prediction holds without closed forms).
- **2d:** For M = 11 only: fraction full-veridical vs T (capacity-sufficient reference panel).

---

## Simulation 3: Non-Uniform μ and Adaptive Partition

### Purpose
Test that non-uniform task weighting drives asymmetric percept allocation (Remark 2.3.2, Theorem 4.2).

### Design
```
N = 20, M = 5 (lossy: 20 → 5), x_i = i/21

Category A ("predation"): weight w_A
  Beta tasks with mode in [0.1, 0.5]
Category B ("texture"): weight w_B
  Beta tasks with mode in [0.5, 0.9]

Weight ratio w_A/w_B ∈ {1, 3, 10, 30, 100}
T = 100 (fixed)
```

### Parameters
```
K = 500, ε = 0.001, n_gen = 10000
```

### Replications
30 runs per weight ratio.

### Plots
- **3a:** Percepts allocated to region A vs weight ratio. Should increase from ~2.5 to ~4–5.
- **3b:** Evolved partition vs theoretical k-means optimum (coloured bars).
- **3c:** Bayes risk: evolved vs theoretical optimum vs uniform vs random partition.

---

## Validation Checks (all simulations)

**Price equation:** At each generation, verify ΔR̄ ≈ -T·Var(R)/w̄ + O(ε).
**Fisher's theorem:** Verify Δw̄ ≈ Var(w)/w̄ + O(ε).
**Monotonicity:** R̄(gen) non-increasing on average; k_T non-decreasing in T.
**Non-monotone finite-sample caveat:** empirical δ_μ(T) need not be monotone for sample averages; evaluate trend and threshold crossings instead of strict monotonicity.

---

## Computational Budget

| Simulation | Runs | Time per run | Total |
|---|---|---|---|
| Sim 1 (11 values of T × 50 reps) | 550 | ~2 min | ~18 hours |
| Sim 2 (13 values of T × 6 families × 30 reps) | 2,340 | ~2 min | ~78 hours |
| Sim 3 (5 weight ratios × 30 reps) | 150 | ~5 min | ~12.5 hours |
| **Total** | | | **~110 hours** |

Parallelisable across replications. On a 16-core machine: ~7 hours wall time.

---

## Expected Outcomes and Failure Modes

### What would confirm the theory

1. δ_μ(T) crosses zero and grows, k_T is a non-decreasing staircase (Prop 4.6)
2. Fitness gap tracks π²_min · δ_μ from below (Thm 4.1c is a lower bound)
3. Convergence time correlates with 1/δ_μ (Thm 7.4d)
4. Transition width correlates with κ (Remark 4.7)
5. Non-uniform μ drives asymmetric percept allocation matching k-means (Thm 4.2)
6. Price equation decomposition holds generation-by-generation (Thm 7.3)

### What would falsify or require revision

1. **Fitness gap systematically below π²_min · δ_μ.** Error in the lower bound derivation. (Should not happen — proven bound.)
2. **Convergence time uncorrelated with δ_μ.** Spectral gap analysis too loose or finite-population effects dominate.
3. **Transition width uncorrelated with κ.** Gaussian spectral analysis doesn't transfer to structured families. (Most likely partial failure — beta family's spectral structure may not be well-captured by the linear model of §6.)
4. **Evolved partition differs from k-means in Simulation 3.** Evolutionary dynamics trapped at local optima, supporting phylogenetic-inertia concern of §10.5. (Likely for large N, small M.)
5. **Price equation residual not O(ε).** Bug in simulation or violation of model assumptions.

### How to handle partial failures

If outcome (3) occurs: report honestly, note that quantitative spectral predictions are specific to the Gaussian model and serve as qualitative guides for structured families. The qualitative predictions (monotonicity, direction of selection) should still hold.

---

## Code Structure

```
paper1_sims/
├── src/
│   ├── types.jl            # EvolutionParams, TaskFamily, etc.
│   ├── tasks.jl            # Beta, Gaussian, non-uniform samplers
│   ├── encoding.jl         # random_encoding, mutate, is_injective, is_ecologically_veridical
│   ├── risk.jl             # bayes_risk, multi_task_risk
│   ├── theory.jl           # task_distance_matrix, δ_μ, k_T, κ, predictions
│   ├── evolution.jl        # run_evolution (the main loop)
│   └── price.jl            # Price equation verification utilities
├── scripts/
│   ├── sim1_berke.jl
│   ├── sim2_condition.jl
│   └── sim3_nonuniform.jl
├── data/                   # JLD2 output files
├── figures/                # PDF/SVG output
├── Project.toml
└── README.md
```
