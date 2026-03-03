# Between Interface and Truth: Multi-Task Selection Drives Ecologically Veridical Perception

## Abstract

When does optimisation for performance produce representations that are faithful to the structure of the world? This question arises across disciplines: in evolutionary biology (does natural selection favour accurate perception?), in machine learning (do multi-task networks learn truthful internal representations?), and in philosophy of mind (is perception a window onto reality or a species-specific interface?). We develop a mathematical theory that provides a unified answer. An agent equipped with a fixed encoding — a single representation that must serve across all tasks — is selected toward what we call *ecological veridicality*: the preservation of exactly those distinctions between world states that the task ecology demands, no more, no less. This is governed by a *separation condition* on the task distribution: pairs of states that are distinguished on a set of tasks of positive μ-measure must receive distinct internal representations; pairs distinguished only on μ-null sets may be merged without Bayes-risk penalty. We prove that deterministic mean-field evolutionary dynamics (Price decomposition + quasispecies recursion) converge to the capacity-aware ecological optimum *within the mutation-accessible region* of encoding space; global convergence to the global optimum is recovered under a strong connectivity (primitive mutation) assumption, and finite-population Wright-Fisher dynamics track this limit for large populations on finite horizons. Convergence rate is controlled by a spectral ratio that admits an explicit lower bound in terms of the fitness gap inside the dominant accessible class; in the full-separation feasible-veridical regime this recovers a bound in terms of the separation margin δ_μ. As the task ecology diversifies, the resolved ecological complexity k_T = |W/~_T| is monotone non-decreasing (graded cascade), while finite-T risk trends remain family- and capacity-dependent. The qualitative principle — that multi-task optimality selects for sufficient statistics shared across tasks — is known in statistical learning theory (Baxter 2000); our contribution is to embed it in the evolutionary perception framework, prove deterministic convergence under natural-selection dynamics, and derive the graded cascade linking task diversity to representational fidelity. Applied to the debate between Hoffman's Fitness-Beats-Truth theorem (which shows single-task selection favours non-veridical "interface" encodings) and Berke et al.'s simulations (which show multi-task selection reverses this), our theory resolves the tension: both are correct, about different parts of the task ecology. The framework recovers Hoffman's result as the single-task special case, explains Berke et al.'s simulations quantitatively, and offers a mathematical formalisation of the species-specific perceptual world (Umwelt), with a heuristic connection to Frank's (2025) force–metric–bias law.

---

## 1. Introduction

When an agent is optimised for performance — whether by natural selection, gradient descent, or Bayesian updating — does its internal representation come to reflect the true structure of the world? Or can high performance be achieved with representations that are systematically distorted, compressed, or even disconnected from reality? This question is fundamental across several disciplines. In evolutionary biology, it concerns whether natural selection produces accurate perception. In machine learning, it concerns whether deep networks trained on prediction tasks develop "truthful" internal representations. In philosophy of mind, it concerns the epistemic status of conscious experience. Despite its breadth, the question has lacked a precise mathematical treatment that identifies the *conditions* under which optimisation yields representational fidelity.

The sharpest formulation of the negative case is due to Hoffman, Singh, and Prakash (2015). Their Fitness-Beats-Truth (FBT) theorem demonstrates that, in an evolutionary game where organisms act on the basis of perceptual information, encodings tuned to fitness generically dominate encodings that faithfully represent the world. The argument is elegant: a fitness-tuned "interface" encoding collapses world states that yield equal fitness into a single percept, using its limited channel capacity to mark only the fitness-relevant distinctions. A veridical encoding wastes capacity distinguishing states that are equivalent from the organism's perspective. In simulations and under broad conditions, the interface wins (Prakash et al. 2021). The case was further strengthened by Prakash, Fields, Hoffman, Prentner, and Singh (2020), who proved that single-task payoff functions generically fail to preserve mathematical structures — total orders, permutation groups, cyclic groups, measurable spaces — with the probability of structure-preservation approaching zero as the state space grows. Hoffman (2019) drew the philosophical conclusion that perception is fundamentally non-veridical — that we see "icons on a desktop," not reality.

Berke, Walter-Terrill, Jara-Ettinger, and Scholl (2022) challenged this conclusion with a deceptively simple observation. Organisms do not face one task; they face many. A primate foraging for fruit also avoids predators, navigates terrain, selects mates, and monitors conspecifics. If the perceptual system is *cognitively impenetrable* — fixed across tasks, not re-tuned for each one — then the interface strategy fails. An encoding tuned to collapse fruit-irrelevant distinctions may merge precisely the states that predator-avoidance requires to be distinct. In evolutionary simulations with multiple tasks and a fixed encoding, Berke et al. found that the fittest organisms were those with veridical perception. The result was demonstrated by simulation only; the authors offered no analytical proof and explicitly called for "detailed mathematical models" to explain the phenomenon.

Anderson (2015) identified the core logical issue from a philosophical standpoint. He argued that Hoffman et al. had "only considered the problem of adaptation over evolutionary time scales" and had "ignored the need for (and demonstrated capacity of) animals to adjust their behavior to achieve homeostasis within ontogenetic time scales." When organisms face multiple homeostatic demands whose payoff functions are nonmonotonic, "the perceptual response needs to track resources monotonically so that an animal can know how to adapt its behavior to achieve homeostasis" (Anderson 2015, pp. 1508–1509). An interface encoding that maps perception directly onto payoffs provides no directional information — an animal can tell that its resource level is "off" but not *which direction it is off*. Anderson concluded that "it is possible to embrace the significance of fitness in shaping evolution without arriving at the epistemological and metaphysical conclusions of [interface theory]" (p. 1511), but offered no formal framework to substantiate this. Indeed, Hoffman and Singh (2012, §8.3) themselves raised the multi-task question explicitly: does a general-purpose encoding serving multiple fitness functions become more veridical as the number of tasks grows? They conjectured that it would not — "there is no principled reason why maximizing the channel capacity for the best-fit fitness signal should automatically maximize channel capacity for the truth signal" — and noted that "this must remain an open question until detailed mathematical models of this process are developed and studied."

This paper provides such models — one that applies beyond the Hoffman/Berke debate to any system where a fixed representation must serve multiple objectives. We prove that multi-task optimisation with a fixed encoding selects for what we call *ecological veridicality*: the preservation of exactly those distinctions between world states that the task ecology demands. The key mathematical object is the *separation condition* — a property of the task distribution μ that determines which pairs of world states are distinguished on sets of tasks of positive μ-measure. Pairs that are separated must receive distinct internal representations; pairs that differ only on μ-null task sets may be freely merged without Bayes-risk cost. The result is not absolute truth but a graded, task-relative fidelity whose resolution is set by the agent's task ecology.

The core results are: (i) a static optimality theorem showing that ecological veridicality minimises multi-task Bayes risk (uniquely up to percept-label symmetry), with explicit lower bounds in the full-separation regime (Theorem 4.1); (ii) a deterministic mean-field evolutionary convergence proof, using Price's equation for direction of selection and quasispecies theory for class-conditional convergence on the quotient encoding space under constrained mutation, with global convergence as a primitive-mutation special case and explicit spectral-rate bounds in terms of the fitness gap, plus a finite-population Wright-Fisher law-of-large-numbers link (Theorems 7.3–7.5); (iii) a graded separation cascade with monotone non-decreasing resolved complexity k_T as task ecology diversifies (Proposition 4.6); and (iv) the recovery of Hoffman's FBT as the single-task special case and Berke et al.'s simulations as a numerical instance, plus a heuristic FMB connection (Appendix C).

The qualitative principle that multi-task optimality selects for sufficient statistics is established in statistical learning theory (Baxter 2000, Maurer et al. 2016). Our contribution is to give this principle a precise formulation in terms of the separation condition and the graded cascade, to prove deterministic mean-field convergence under evolutionary dynamics via quasispecies theory (with finite-population approximation guarantees), and to show that the resulting framework resolves the Hoffman/Berke debate while providing a mathematical formalisation of aspects of von Uexküll's (1934) species-specific perceptual world (Umwelt). We view this paper as a bridge between statistical learning theory, evolutionary game theory, and the philosophy of perception — applicable wherever a shared representation must serve a diverse set of objectives.

The paper is organised as follows. Section 2 sets up the framework: world states, encodings, tasks, Bayes risk. Section 3 defines the separation condition and the task distance. Section 4 proves the static optimality theorems, including the main veridicality result, the lossy case, the phase transition, and the graded cascade. Section 5 provides finite-sample concentration bounds. Section 6 develops the Gaussian task model and spectral analysis. Section 7 proves mean-field evolutionary convergence via Price's equation and quasispecies theory to dominant mutation-accessible classes (global convergence under primitive mutation), and states the finite-population approximation on fixed horizons. Section 8 recovers prior results as special cases. Section 9 discusses the implications — including the distinction between static optimality and evolutionary dynamics (§9.3), the ecological Umwelt interpretation (§9.4), and limitations (§9.5), with philosophy-of-evolution considerations woven into those sections. Appendix C gives a heuristic connection to Frank's FMB law, and Appendix D gives the full reducible-case proof for the primitive-block setting plus a periodic-class extension.

---

## 1.1. Related Work and Intellectual Genealogy

The qualitative principle underlying our main result — that optimal multi-task representation preserves task-relevant distinctions — is established in statistical learning theory. Baxter (2000) proved that learning a shared representation across multiple tasks converges to the common sufficient statistic of the task family, with sample complexity that decreases as the number of tasks grows. Maurer et al. (2016) provided finite-sample PAC-Bayes bounds showing that shared representations recover ground-truth subspaces in multi-task settings. In the information-theoretic literature, the same principle appears as the statement that the minimal sufficient statistic for a family of distributions is the optimal data reduction.

Our contribution is not the discovery of this principle but its embedding in the evolutionary perception framework of Hoffman et al. (2015) and the resolution of specific open questions within that debate. The following table clarifies the correspondence:

| This paper | Multi-task learning theory |
|---|---|
| World states W | Input space X |
| Encoding p: W → X | Shared representation φ: X → Z |
| Task f: W → ℝ | Task-specific loss function ℓ_t |
| Separation condition (Def. 3.3) | Shared-representation condition |
| Equivalence classes [w]_μ | Learned invariances / null space of task family |
| Ecological veridicality | Sufficiency of shared representation |
| Separation margin δ_μ | Task diversity / eigenvalue gap |

What the learning theory literature does *not* provide, and what we contribute here, is:

1. **The connection to Hoffman's FBT theorem.** The FBT result is formulated in a specific evolutionary game with fitness functions, perception maps, and reproductive dynamics. Prakash et al. (2020) further proved that single-task payoff functions generically fail to preserve mathematical structures. Showing that multi-task optimality *reverses* this — that the composite selective pressure across tasks restores exactly the structure the task ecology demands — requires reformulating the learning-theoretic principle within this game, which involves different definitions, different loss structures, and different notions of optimality.

2. **Evolutionary convergence guarantees.** The learning theory results concern statistical risk minimisation by an algorithm. Our Theorems 7.3–7.4 analyse the deterministic mean-field natural-selection recursion (replicator-mutator on quotient encoding space), proving convergence to the dominant mutation-accessible optimum and, under primitive mutation, to the global optimum. Theorem 7.5 links this to finite-population Wright-Fisher dynamics via a law-of-large-numbers limit. Rate is controlled by the spectral ratio and hence by the fitness gap; in full-separation feasible-veridical primitive regimes this yields explicit δ_μ-dependent bounds. This requires quasispecies theory and spectral analysis, not PAC-Bayes bounds.

3. **The graded cascade and ecological Umwelt.** The learning theory does not address the evolutionary question of how veridicality increases as the task ecology diversifies over phylogenetic time. Proposition 4.6 and §9.4 provide this developmental-evolutionary picture.

4. **A heuristic bridge to Frank's (2025) FMB law.** Identifying Price's equation as a conceptual bridge between evolutionary dynamics and Frank's force-metric-bias decomposition is specific to the evolutionary setting and has no analogue in the learning theory.

We view this paper as a *bridge* between statistical learning theory, evolutionary game theory, and the philosophy of perception — not as a de novo derivation of principles already known in any one of these fields. Philosophically, we treat the theorems as conditional adaptation claims in the sense emphasized by Sober's evidential and methodological analyses (2008, 2024): evidential support is model-comparative, and population-level laws require initial conditions and auxiliaries before yielding concrete predictions. This also aligns with Pigliucci and Kaplan's argument that adaptationist explanation must be assessed against alternative causal pathways and developmental structure, not inferred directly from statistical fit alone (2006, prelude; chs. 2 and 5). Likewise, following Godfrey-Smith's model-based treatment of Darwinian populations (2009, 2024), the framework is an idealized map from ecological structure to selective pressure, not a complete biological reconstruction.

## 2. Framework

### 2.1. World, Encoding, Tasks

**Definition 2.1 (World).** Let W = {w₁, ..., w_N} be a finite set of *world states* equipped with a prior distribution π, where π(w) > 0 for all w ∈ W. Write π_min = min_w π(w).

**Definition 2.2 (Encoding).** Let X = {x₁, ..., x_M} be a finite set of *percepts*. An *encoding* is any function p: W → X (not necessarily surjective). The encoding induces a partition of W into non-empty fibres (cells) C = {C_x : x ∈ Im(p)}, where C_x = p⁻¹(x) = {w ∈ W : p(w) = x}. We write m(p) = |Im(p)| ≤ M for the number of percept values actually used by p.

**Definition 2.2.1 (Percept-label symmetry and quotient).** Let S_M be the permutation group on X. It acts on encodings by relabeling percepts:

  (σ · p)(w) = σ(p(w)),   σ ∈ S_M.

Define orbit equivalence p ~_X p' iff p' = σ · p for some σ ∈ S_M. The orbit space is

  Ω̄ = Ω / ~_X,   where Ω = X^W.

Elements of Ω̄ are encoding classes modulo percept-label names.

**Lemma 2.2.3 (Risk invariance under relabeling).** For every σ ∈ S_M and encoding p:

  R(σ · p) = R(p).

*Proof.* Relabeling only permutes cell names; the underlying partition {C_x} of W is unchanged. Equations (2), (3), and Lemma 2.9 depend on p only through this partition and associated π-masses, hence are invariant under σ. □

**Corollary 2.2.4 (Uniqueness on the quotient).** Any statement of uniqueness "up to percept-label permutation" is equivalent to ordinary uniqueness on Ω̄.

**Definition 2.2.2 (Veridicality hierarchy).**
- **Full veridicality:** p is injective on W (possible iff M ≥ N).
- **Ecological veridicality (relative to μ):** p is injective on the quotient W/~_μ, equivalently p(w₁) = p(w₂) implies w₁ ~_μ w₂.
- **Lossy-optimal encoding at capacity M:** any minimiser of R(p) over all functions p: W → X with |X| = M.

When μ separates all points, ecological veridicality coincides with full veridicality. When M < |W/~_μ|, ecological veridicality is infeasible and the relevant notion is lossy optimality.

**Definition 2.3 (Task).** A *task* is a bounded function f: W → [-B, B] for some B > 0. We identify tasks with vectors f ∈ ℝ^N. A *task distribution* is a probability measure μ on ℝ^N.

**Remark 2.3.1 (Scope of the task definition).** Defining tasks as scalar-valued functions f: W → ℝ restricts attention to point estimation of a single quantity. This is the framework used by Hoffman et al. (2015) and Prakash et al. (2021) in formulating the FBT theorem, and we adopt it for direct comparability. Real biological tasks — categorisation, relational judgment, motor planning — are richer. The extension to vector-valued tasks f: W → ℝ^D is straightforward: the task distance becomes σ²(w₁, w₂) = E_μ[‖f(w₁) - f(w₂)‖²], and the entire theory carries through with D-dimensional variance replacing scalar variance. Tasks with categorical or non-numeric outcomes require replacing squared error with an appropriate proper scoring rule (cf. Remark 4.1.1). The qualitative structure of the theory — separation, ecological veridicality, evolutionary convergence — does not depend on the scalar restriction.

**Remark 2.3.2 (Non-uniform task weighting).** The task distribution μ is an arbitrary probability measure, not a uniform one. Tasks that are ecologically more frequent, more fitness-consequential, or both, receive greater weight under μ. This weighting propagates through the entire theory: the task distance σ²(w₁, w₂) = E_{f∼μ}[(f(w₁) − f(w₂))²] is a μ-weighted average, so pairs of world states that high-weight tasks distinguish contribute more to separation than pairs that only low-weight tasks distinguish. In the lossy case (M < N, Theorem 4.2), the optimal partition allocates limited perceptual capacity preferentially to distinctions that high-weight tasks demand. For example, a raptor whose task ecology is dominated by predation (detecting prey against a cluttered background) will have an optimal encoding that finely discriminates motion-related features at the expense of texture distinctions among stationary objects — not because the encoding is "distorted," but because it is ecologically veridical with respect to a μ that heavily weights predation. The apparent distortion (e.g. telephoto acuity at the fovea, poor peripheral texture resolution) is the theory's prediction, not a deviation from it.

**Definition 2.4 (Cognitive impenetrability).** The encoding p is fixed across all tasks. Only the downstream *readout* (the mapping from percepts to actions) may vary per task.

### 2.2. Bayes Risk

Given encoding p and task f, an agent observing percept x = p(w) must estimate f(w). Under squared-error loss, the optimal estimate is the conditional expectation:

  f̂(x) = E[f(w) | p(w) = x] = Σ_{w ∈ C_x} π(w | C_x) f(w)         ... (1)

where π(w | C_x) = π(w)/π(C_x) and π(C_x) = Σ_{w ∈ C_x} π(w).

**Definition 2.5 (Single-task Bayes risk).** The Bayes risk of encoding p on task f is:

  R(p, f) = E_w[(f(w) - f̂(p(w)))²]
           = Σ_x π(C_x) · Var(f | C_x)                               ... (2)

where Var(f | C_x) = Σ_{w ∈ C_x} π(w | C_x) [f(w) - f̂(x)]² is the within-cell conditional variance.

**Lemma 2.6 (Variance decomposition).** For any encoding p and task f:

  Var(f) = Σ_x π(C_x) · Var(f | C_x) + Var(f̂(p(w)))

That is: total variance = within-cell variance (= Bayes risk) + between-cell variance (= explained variance).

*Proof.* This is the law of total variance: Var(Y) = E[Var(Y|Z)] + Var(E[Y|Z]) applied with Y = f(w) and Z = p(w). □

**Corollary 2.7.** R(p, f) = 0 if and only if f is constant on each cell C_x. That is, f is p-measurable.

*Proof.* R(p, f) = Σ_x π(C_x) Var(f | C_x) = 0 iff Var(f | C_x) = 0 for all x (since π(C_x) > 0), iff f is constant on each C_x. □

**Definition 2.8 (Multi-task Bayes risk).** The expected Bayes risk over the task distribution:

  R(p) = E_{f ~ μ}[R(p, f)] = Σ_x π(C_x) · E_μ[Var(f | C_x)]       ... (3)

### 2.3. The Pairwise Decomposition

The multi-task Bayes risk admits a revealing pairwise decomposition.

**Lemma 2.9 (Pairwise Bayes risk formula).** For any encoding p:

  R(p) = (1/2) Σ_x Σ_{w₁, w₂ ∈ C_x} π(w₁|C_x) π(w₂|C_x) π(C_x) σ²(w₁, w₂)

       = (1/2) Σ_x (1/π(C_x)) Σ_{w₁, w₂ ∈ C_x} π(w₁) π(w₂) σ²(w₁, w₂)

where σ²(w₁, w₂) = E_{f ~ μ}[(f(w₁) - f(w₂))²].

*Proof.* We use the identity: for any discrete random variable Z with values {z_i} and probabilities {p_i}:

  Var(Z) = (1/2) Σ_{i,j} p_i p_j (z_i - z_j)²                      ... (*)

Applied to f restricted to cell C_x with weights π(· | C_x):

  Var(f | C_x) = (1/2) Σ_{w₁, w₂ ∈ C_x} π(w₁|C_x) π(w₂|C_x) (f(w₁) - f(w₂))²

Taking E_μ and using linearity (the exchange of expectation and summation is justified by Fubini's theorem, since all terms are non-negative and the sums are finite):

  E_μ[Var(f | C_x)] = (1/2) Σ_{w₁, w₂ ∈ C_x} π(w₁|C_x) π(w₂|C_x) σ²(w₁, w₂)

Substituting into (3) and using π(w_i | C_x) = π(w_i)/π(C_x):

  R(p) = Σ_x π(C_x) · (1/2) Σ_{w₁,w₂ ∈ C_x} [π(w₁)π(w₂)/π(C_x)²] σ²(w₁,w₂)
       = (1/2) Σ_x (1/π(C_x)) Σ_{w₁,w₂ ∈ C_x} π(w₁) π(w₂) σ²(w₁,w₂)     □

**Interpretation.** The Bayes risk is a weighted sum of σ²(w₁, w₂) over all pairs of world states that are *merged* (placed in the same cell). The weight for a merged pair (w₁, w₂) ∈ C_x is π(w₁)π(w₂)/π(C_x), which is always positive. Every merged pair of task-distinguishable states contributes positively to Bayes risk.

---

## 3. The Separation Condition

**Definition 3.1 (Task distance).** For w₁, w₂ ∈ W, define:

  σ²(w₁, w₂) = E_{f ~ μ}[(f(w₁) - f(w₂))²]

This is the expected squared difference in task values between w₁ and w₂.

**Lemma 3.2.** The function d(w₁, w₂) = σ(w₁, w₂) is a pseudo-metric on W.

*Proof.* (i) d(w, w) = 0 since f(w) - f(w) = 0. (ii) d(w₁, w₂) = d(w₂, w₁) trivially. (iii) Triangle inequality: by Minkowski's inequality in L²(μ),

  ‖f(w₁) - f(w₃)‖_{L²(μ)} ≤ ‖f(w₁) - f(w₂)‖_{L²(μ)} + ‖f(w₂) - f(w₃)‖_{L²(μ)}

i.e. σ(w₁, w₃) ≤ σ(w₁, w₂) + σ(w₂, w₃). □

**Definition 3.3 (Separation).** μ *separates* w₁ from w₂ if σ²(w₁, w₂) > 0, equivalently if f(w₁) ≠ f(w₂) on a set of tasks of positive μ-measure. The distribution μ *separates points* (or is *point-separating*) if σ²(w₁, w₂) > 0 for all w₁ ≠ w₂.

**Definition 3.4 (Separation margin).** When μ separates points:

  δ_μ = min_{w₁ ≠ w₂} σ²(w₁, w₂) > 0

**Definition 3.5 (Task equivalence).** Define w₁ ~_μ w₂ iff σ²(w₁, w₂) = 0. This is an equivalence relation (reflexivity and symmetry are obvious; transitivity follows from the triangle inequality for σ). Equivalently, w₁ ~_μ w₂ iff f(w₁) = f(w₂) for μ-almost every task f. The equivalence classes [w]_μ partition W into groups of states indistinguishable up to μ-null task sets.

**Remark 3.6 (Algebraic characterisation).** Embed tasks as vectors: define Φ: W → L²(μ) by Φ(w) = [f ↦ f(w)]. Then:

  σ²(w₁, w₂) = ‖Φ(w₁) - Φ(w₂)‖²_{L²(μ)}

Separation means Φ is injective. The task distance is the L²(μ) distance between the images.

---

## 4. Static Optimality Theorems

### Theorem 4.1 (Multi-Task Veridicality — Equal Complexity)

Let k_μ = |W/~_μ|. Then:

  (a) R(p) = 0 if and only if for every cell C_x, all elements of C_x are in the same μ-equivalence class [w]_μ. Equivalently, p is ecologically veridical (injective on W/~_μ).

  (b) Hence R(p) = 0 is achievable if and only if M ≥ k_μ. In particular:
    - If μ separates all points (δ_μ > 0, so k_μ = N): R(p) = 0 iff p is injective on W (full veridicality).
    - If μ does not separate all points: zero-risk encodings are exactly those that merge only μ-equivalent states.

  (c) If μ separates points, then for any encoding p that merges at least one pair w₁ ≠ w₂:
        R(p) ≥ π²_min · δ_μ > 0

Consequently, under full separation and M ≥ N, the fully veridical encoding is the unique minimiser *up to a permutation of percept labels*.

*Proof.*

**(a), ⟹:** If every cell C_x contains only μ-equivalent states, then for any pair w₁, w₂ ∈ C_x we have σ²(w₁, w₂) = 0, i.e. f(w₁) = f(w₂) for μ-a.e. f. So Var(f | C_x) = 0 for μ-a.e. f, and hence E_μ[Var(f | C_x)] = 0 for every x. Thus R(p) = 0.

**(a), ⟸:** Suppose R(p) = 0. Then E_μ[Var(f | C_x)] = 0 for all x (since π(C_x) > 0). This means Var(f | C_x) = 0 for μ-a.e. f. For a cell C_x = {w₁, ..., w_k} with k ≥ 2, Var(f | C_x) = 0 requires f(w₁) = f(w₂) = ... = f(w_k) for μ-a.e. f. Taking expectations: σ²(w_i, w_j) = E_μ[(f(w_i) - f(w_j))²] = 0 for all pairs i, j within C_x. So every merged pair is μ-equivalent. □

**(c):** Assume μ separates points. Let p merge at least one distinct pair. Then ∃ distinct w₁, w₂ ∈ W with p(w₁) = p(w₂) = x₀. By Lemma 2.9, the contribution of this pair to R(p) is:

  R(p) ≥ [π(w₁)π(w₂)/π(C_{x₀})] · σ²(w₁, w₂)                      ... (4)

Since σ²(w₁, w₂) ≥ δ_μ > 0, equation (4) gives:

  R(p) ≥ π(w₁)π(w₂)/π(C_{x₀}) · δ_μ

Now π(w₁) ≥ π_min, π(w₂) ≥ π_min, and π(C_{x₀}) ≤ 1. So:

  R(p) ≥ π_min · π_min / 1 · δ_μ = π²_min · δ_μ                     □

**Remark 4.1.1 (Loss-function dependence).** The results in this paper divide into two categories with different degrees of generality:

*Loss-general results (hold for any strictly proper loss function):*
- Theorem 4.1(a): R(p) = 0 iff p merges only μ-equivalent states. This is an information-theoretic statement: under any strictly proper loss, the Bayes risk vanishes iff the task value is measurable with respect to the encoding partition, iff f is constant on each cell for μ-a.e. f.
- Proposition 4.3: Gauge freedom within equivalence classes.
- Theorem 4.4 and Proposition 4.6: The phase transition and graded cascade, which depend on the rank of the task matrix, not on the loss.
- The qualitative direction of selection (Theorem 7.3): merging separated states incurs positive Bayes risk under any proper loss, so selection always pushes against merging.

*Squared-error-specific results:*
- Theorem 4.1(c): The specific lower bound R(p) ≥ π²_min · δ_μ, where δ_μ is defined via the squared task distance σ²(w₁, w₂) = E_μ[(f(w₁) - f(w₂))²].
- The pairwise decomposition (Lemma 2.9) and the spectral analysis of §6.
- The Hoeffding concentration bounds of §5.
- The quantitative convergence rate in Theorem 7.4(d).

Under a different loss, the task "distance" between world states becomes a different divergence (e.g. total variation, Hellinger, or a loss-specific Bregman divergence), and the pairwise decomposition, spectral structure, and concentration bounds take different forms. The qualitative structure of the theory — separation condition, fitness gap, evolutionary convergence — is preserved, but the quantitative bounds are not portable. We work throughout with squared error because it is the standard benchmark in the FBT literature (Hoffman et al. 2015, Prakash et al. 2021), because it yields the cleanest algebra, and because it permits the spectral analysis of §6. Extending the quantitative theory to specific biologically motivated loss functions (e.g. asymmetric losses reflecting the "smoke detector principle") is a natural direction for future work; the key observation is that asymmetric losses that catastrophically penalise certain errors can only *increase* the cost of merging the relevant states, strengthening the selective pressure toward ecological veridicality for those distinctions.


### Theorem 4.2 (Lossy Case — Finite Partition Optimisation)

Let Π_M be the finite set of partitions of W into at most M non-empty cells. For P = {C_1, ..., C_m} ∈ Π_M define

  J(P) := (1/2) Σ_{C ∈ P} (1/π(C)) Σ_{w₁,w₂ ∈ C} π(w₁)π(w₂) σ²(w₁,w₂).   ... (4.2)

Then:

(a) For every encoding p: W → X with |X| = M, if P(p) is its induced partition, R(p) = J(P(p)).

(b) Conversely, for every P ∈ Π_M there exists an encoding p_P with P(p_P)=P and R(p_P)=J(P).

(c) Therefore

  min_{p:W→X, |X|=M} R(p) = min_{P ∈ Π_M} J(P),

and a minimiser exists (finite search space).

When M < k_μ (in particular M < N under full separation), every feasible encoding is lossy relative to μ and the optimum is the best M-cell partition under J.

*Proof.* (a) is exactly Lemma 2.9 rewritten by cells, so risk depends on p only through P(p). (b) Given P with m ≤ M cells, assign each cell a distinct label in X and map each w to its cell label; unused labels are allowed. Then P(p_P)=P and (a) gives R(p_P)=J(P). (c) (a)-(b) establish equivalence between optimisation over encodings and optimisation over Π_M. Since W is finite, Π_M is finite, so a minimiser exists. □

**Interpretation.** This is a finite weighted partitioning problem (an analogue of k-means distortion minimisation with fixed cluster count), where distortion is induced by the task distance and prior-weighted within-cell pair penalties.

**Remark 4.2.1 (Metabolic cost and channel capacity).** The cardinality M = |X| of the percept space is a structural constraint reflecting the organism's metabolic and channel-capacity limits. Maintaining M distinct perceptual states requires neural architecture — receptor diversity, channel bandwidth, cortical representational area — that carries a metabolic cost. Our framework parameterises this cost through fixed capacity M rather than through an explicit penalty term. Theorem 4.1 characterises when zero Bayes risk is feasible (M ≥ k_μ); Theorem 4.2 characterises the optimum at any constrained budget, especially M < k_μ. The question "what is the optimal M?" is a separate problem involving metabolic trade-offs that our framework does not address — but neither does Hoffman's FBT, which also fixes perceptual capacity. The lossy k-means structure of Theorem 4.2 is closely related to rate-distortion theory: the optimal M-partition minimises distortion (Bayes risk) at a given rate (log M bits of channel capacity). An explicit rate-distortion formulation, with a Lagrange multiplier β on the mutual information I(W; P), would yield the same qualitative structure — aggressive compression of task-equivalent states, faithful separation of task-distinguishable states — with β playing the role of metabolic cost per bit.

**Proposition 4.3 (Gauge degeneracy of ecological optima).** Let [w]_μ denote a task-equivalence class. Any encoding that is injective on W/~_μ has Bayes risk 0 (Theorem 4.1), and therefore all such encodings are global minimisers. In particular, when μ does not separate points, there is a nontrivial degenerate set of optimal encodings corresponding to arbitrary assignments *within* each equivalence class, subject only to not merging distinct classes.

*Proof.* By Theorem 4.1(a), R(p)=0 iff each cell contains only μ-equivalent states. Therefore every encoding injective on the quotient W/~_μ attains the same minimal value 0. If at least one class has cardinality > 1, there are multiple such encodings, giving degeneracy of minimisers. □

**Interpretation.** The equivalence classes [w]_μ define a flat optimum set: any encoding that separates distinct classes but is arbitrary within each class attains the same minimal risk 0. This degeneracy disappears when μ separates all points.

### Theorem 4.4 (Empirical Separation and Capacity Thresholds)

Let f₁, ..., f_T be T tasks. Form the *task matrix* F ∈ ℝ^{T × N} with F_{ti} = f_t(w_i). Define the empirical task distance:

  σ²_T(w_i, w_j) = (1/T) Σ_{t=1}^T (f_t(w_i) - f_t(w_j))²
                  = (1/T) ‖F_·i - F_·j‖²

where F_·i is the i-th column of F. Let ~_T be the empirical equivalence relation w_i ~_T w_j iff σ²_T(w_i, w_j) = 0, and k_T = |W/~_T|.

(a) σ²_T(w_i, w_j) > 0 for all i ≠ j iff the columns of F are pairwise distinct (equivalently, k_T = N).

(b) rank(F) = N is sufficient (not necessary) for pairwise distinct columns.

(c) Capacity criterion for zero empirical risk:
    - Zero empirical Bayes risk is achievable iff M ≥ k_T.
    - Full empirical veridicality requires both k_T = N and M ≥ N.

(d) If tasks are drawn from a distribution absolutely continuous on ℝ^N, then with probability 1 a single task has distinct values on all N states, so k₁ = N and the cascade is trivial. Thus biologically meaningful phase transitions require structured task families.

*Proof.*

(a) Immediate from σ²_T(w_i, w_j) = 0 iff F_·i = F_·j.

(b) Linear independence of columns implies pairwise distinctness.

(c) For empirical objective R_T(p) = (1/T)Σ_t R(p,f_t), we have R_T(p)=0 iff R(p,f_t)=0 for every t (nonnegative summands). By Corollary 2.7, this holds iff each task f_t is constant on every cell of p, equivalently iff each cell is contained in an empirical equivalence class of ~_T. Hence one needs at least one percept per ~_T-class, i.e. M ≥ k_T, and this is also sufficient by assigning one percept per class.

(d) If f₁ is absolutely continuous on ℝ^N, then P(f₁(w_i) = f₁(w_j)) = 0 for i ≠ j. Finite union bound gives pairwise distinctness almost surely. □

**Corollary 4.5 (Capacity-Aware Transition).** Let T ↦ k_T denote empirical ecological complexity. Then:

- The operative transition at fixed capacity M (as T increases) is from *ecologically veridical* (k_T ≤ M) to *lossy-optimal* (k_T > M).
- Full veridicality is a special case requiring k_T = N and M ≥ N.
- There is no universal critical task number T* independent of task family; instead,
      T_loss(M) = inf{T : k_T > M},
  and its value is family-dependent.

**Proposition 4.6 (Graded separation cascade).** Let f₁, f₂, ... be a sequence of tasks and define k_T as above. Then:

(a) k_T is non-decreasing in T: k_{T+1} ≥ k_T.

(b) 1 ≤ k_T ≤ N.

(c) For fixed M, no monotonic law holds in general for the *averaged* optimum R*(M, T) = min_p R_T(p) when R_T is the sample average over T tasks: adding one task changes both numerator and denominator. (A monotone statement does hold for the cumulative objective S_T(p)=Σ_{t≤T}R(p,f_t), for which min_p S_T is non-decreasing.)

*Proof.*

(a) If σ²_T(w_i, w_j) > 0, then
    σ²_{T+1}(w_i, w_j) = (T/(T+1))σ²_T(w_i, w_j) + (1/(T+1))(f_{T+1}(w_i)-f_{T+1}(w_j))² > 0.
    So distinctions cannot be lost.

(b) Immediate.

(c) For any fixed partition p, R_{T+1}(p) is the average of T+1 nonnegative terms and need not be ≤ R_T(p). Since the minimisation is over the same feasible set, monotonic decrease is not guaranteed. □

**Remark 4.6.1 (Family dependence of cascade shape).** Proposition 4.6 gives only monotonicity and bounds. The *shape* of the staircase T ↦ k_T is model-dependent: unstructured continuous task families tend to yield k₁ = N almost surely (Theorem 4.4(d), trivial cascade), while structured families can exhibit gradual growth, with transition width governed by geometric anisotropy of the task family.

**Remark 4.7 (Condition number and transition width).** In structured families, transition sharpness is controlled by anisotropy of task variation. In the Gaussian-linear model (§6), this is captured by κ = λ_max/λ_min of Σ_c on the relevant subspace: κ ≈ 1 gives sharp transitions; κ ≫ 1 gives broad transitions. This is a family-specific quantitative prediction, not a universal law across all task families.

---

## 5. Concentration and Finite-Sample Guarantees

### Theorem 5.1 (Exponential Convergence with T Tasks)

Let f₁, ..., f_T be drawn iid from μ, with |f(w)| ≤ B for all w, f. For any fixed encoding p with R(p) > 0:

  P(R̄_T(p) ≤ ε) ≤ exp(-2T(R(p) - ε)²/B⁴)     for ε < R(p)

where R̄_T(p) = (1/T) Σ_{t=1}^T R(p, f_t) is the empirical Bayes risk.

*Proof.* The random variables R(p, f_t) are iid with:
- E[R(p, f_t)] = R(p) > 0
- 0 ≤ R(p, f_t) ≤ B² (since Var(f | C_x) ≤ E[f² | C_x] ≤ B²)

By Hoeffding's inequality for bounded iid variables in [0, B²]:

  P(R̄_T - R(p) ≤ -(R(p) - ε)) ≤ exp(-2T(R(p) - ε)² / (B²)²)

which gives the result. □

### Corollary 5.2 (Uniform Bound over All Non-Veridical Encodings)

Assume full separation (δ_μ > 0, so k_μ = N) and |W| = |X| = N. Then the number of encodings is N^N, and the probability that any encoding that merges at least one separated pair has empirical Bayes risk below ε is:

  P(∃ p that merges at least one separated pair: R̄_T(p) ≤ ε) ≤ N^N · exp(-2T(π²_min δ_μ - ε)²/B⁴)

Setting the right side ≤ α and solving:

  T ≥ B⁴/(2(π²_min δ_μ - ε)²) · (N log N + log(1/α))

*Proof.* Union bound over at most N^N encodings, each satisfying Theorem 5.1 with R(p) ≥ π²_min δ_μ. □

**Remark 5.3 (Conservatism of the bound).** Under the full-separation assumptions of Corollary 5.2, this bound is deliberately worst-case: it uses the minimum Bayes risk gap π²_min δ_μ for every non-veridical encoding, whereas most non-veridical encodings have much larger Bayes risk (merging multiple well-separated pairs). It also uses the crude union bound over all N^N encodings, ignoring correlations between overlapping encodings. The bound is therefore most useful as a qualitative guarantee that exponentially many tasks are *not* required — polynomial in N suffices — rather than as a practical estimate of the transition point for specific task families. For the latter, the spectral analysis of §6 and Remark 4.7 provide sharper, problem-specific predictions.

---

## 6. Gaussian Task Model and Spectral Analysis

The results of §§3–5 hold for arbitrary task distributions μ. In this section, we develop a specific parametric example — Gaussian-linear tasks — to illustrate how the abstract quantities (σ², δ_μ, the separation condition) can be computed in closed form and related to the spectral structure of the task covariance. This is one tractable instantiation, chosen for its analytical transparency; other task families (e.g. the beta-function tasks of Berke et al.) yield the same qualitative structure but require numerical computation of σ² and δ_μ rather than closed-form expressions. Nothing in the core theory (Theorems 4.1, 4.2, 7.3, 7.4) depends on the Gaussian assumption.

### 6.1. Setup

Assign each world state a feature vector: w_i ↦ φ_i ∈ ℝ^D. Tasks are random linear functions:

  f(w_i) = c^T φ_i,   c ~ N(0, Σ_c)

where Σ_c ∈ ℝ^{D × D} is the task covariance matrix (positive semi-definite).

### Theorem 6.1 (Task Distance under Gaussian Model)

  σ²(w_i, w_j) = (φ_i - φ_j)^T Σ_c (φ_i - φ_j)

This is the squared Mahalanobis distance between φ_i and φ_j under Σ_c.

*Proof.* 

  σ²(w_i, w_j) = E[(c^T φ_i - c^T φ_j)²]
               = E[(c^T(φ_i - φ_j))²]
               = (φ_i - φ_j)^T E[cc^T] (φ_i - φ_j)
               = (φ_i - φ_j)^T Σ_c (φ_i - φ_j)                      □

### Theorem 6.2 (Separation Characterisation)

Let Δ = {φ_i - φ_j : i ≠ j} be the set of pairwise difference vectors, and let V = span(Δ) be the subspace they span (dimension r ≤ min(D, N-1)).

Exact pairwise condition:

μ separates all points iff

  (φ_i - φ_j)^T Σ_c (φ_i - φ_j) > 0   for all i ≠ j,

equivalently iff no nonzero pairwise difference vector lies in ker(Σ_c|_V).

A convenient sufficient condition is that Σ_c is positive definite on V, i.e.:

  λ_min(P_V Σ_c P_V) > 0

where P_V is the orthogonal projection onto V.

The separation margin is:

  δ_μ = min_{i ≠ j} (φ_i - φ_j)^T Σ_c (φ_i - φ_j)
      ≥ λ_min(Σ_c|_V) · min_{i ≠ j} ‖P_V(φ_i - φ_j)‖²
      = λ_min(Σ_c|_V) · d²_min

where λ_min(Σ_c|_V) is the smallest eigenvalue of Σ_c restricted to V, and d²_min = min_{i ≠ j} ‖φ_i - φ_j‖² is the squared minimum distance between world-state features.

*Proof.* By Theorem 6.1, σ²(w_i, w_j) = Δ_{ij}^T Σ_c Δ_{ij} with Δ_{ij} = φ_i - φ_j ∈ V. Thus μ separates all points iff Δ_{ij}^T Σ_c Δ_{ij} > 0 for all i ≠ j, proving the exact condition.

If Σ_c is positive definite on V, then v^T Σ_c v > 0 for every nonzero v ∈ V, hence in particular for every Δ_{ij} ≠ 0, so separation follows. For the margin bound, for any v ∈ V with ‖v‖ = 1:

  v^T Σ_c v ≥ λ_min(Σ_c|_V)

So σ²(w_i, w_j) = ‖Δ_{ij}‖² · (Δ_{ij}/‖Δ_{ij}‖)^T Σ_c (Δ_{ij}/‖Δ_{ij}‖) ≥ ‖Δ_{ij}‖² · λ_min(Σ_c|_V).

Taking the minimum over i ≠ j gives δ_μ ≥ λ_min(Σ_c|_V) · d²_min. □

### Corollary 6.3 (Spectral Control of Convergence)

In the full-separation feasible-veridical regime (μ separates points and M ≥ N), assume Σ_c is positive definite on V so that (Theorem 6.2)

  δ_μ ≥ λ_min(Σ_c|_V) · d²_min.

Then for any confidence level α ∈ (0,1) and tolerance ε with

  0 < ε < π²_min λ_min(Σ_c|_V) d²_min,

a sufficient condition for veridical empirical dominance (via Corollary 5.2) is

  T ≥ [B⁴ / (2(π²_min λ_min(Σ_c|_V)d²_min - ε)²)] · (N log N + log(1/α)).   ... (6.3)

Hence T scales polynomially in N and inversely with the squared spectral bottleneck λ_min(Σ_c|_V).

**Asymptotic reading.** For fixed α, ε and other constants, the leading dependence is

  T = O(N log N / (π⁴_min λ²_min(Σ_c|_V) d⁴_min)).

---

## 7. Evolutionary Dynamics via Price's Equation

We now analyse two coupled levels of evolutionary dynamics: (i) the finite-K stochastic Wright-Fisher process driven by realised fitness w_t; and (ii) its deterministic mean-field limit driven by expected fitness W. This is the dynamic complement to the static theorems of §4. The argument uses three tools with distinct roles:

- **Price's equation** (§7.2–7.4) provides a *decomposition* of evolutionary change into selection and transmission components. It identifies the *direction* of selection: mean Bayes risk decreases each generation (Theorem 7.3). However, Price's equation is a statistical identity — an exact partition of change — not a dynamical law. It cannot, by itself, predict long-term trajectories or prove convergence to an equilibrium.

- **Quasispecies theory** (§7.5) provides the *dynamical* result. In the frequency-independent setting, the deterministic update is a normalized positive linear map. Perron-Frobenius theory yields convergence to the dominant mutation-accessible asymptotic regime (equilibrium in primitive blocks, periodic limit in irreducible periodic blocks); unique global equilibrium is recovered when quotient mutation is primitive (Theorem 7.4, Remark 7.4.3).

- **Spectral perturbation theory** (§7.5, Theorem 7.4(d)) provides the *rate*. The convergence speed is governed by a dominant spectral ratio (within the dominant reachable class plus inter-class spectral separation). Under explicit small-mutation assumptions, this ratio is controlled by the class-wise capacity-aware fitness gap. A δ_μ-based bound is recovered as a corollary in the full-separation feasible-veridical primitive regime.

The causal mechanism throughout is ordinary natural selection: organisms with encodings that merge task-distinguishable world states incur positive Bayes risk, hence lower fitness, hence fewer offspring. Price describes one-generation change; the replicator-mutator recursion gives the mean-field dynamical law; Wright-Fisher gives the finite-population stochastic process around that law. This separation also aligns with the selection-for/selection-of distinction: covariance decomposition alone does not establish long-run adaptation, whereas the explicit dynamical model does. In Pigliucci and Kaplan's terms, statistical patterns underdetermine causal history unless they are embedded in explicit, competing causal models tested with additional evidence (2006, prelude; ch. 2).

### 7.1. Population Model

Consider a population of K agents at generation t. Agent i has encoding p_i: W → X. Let Ω denote the finite set of all encodings W → X (so |Ω| = M^N). For each p ∈ Ω, let n_t(p) be the number of agents with encoding p, and π_t(p) = n_t(p)/K the frequency.

**Fitness.** In generation t, all agents face the same sampled tasks f₁, ..., f_T. Define realised generational fitness

  w_t(p) = TC - Σ_{τ=1}^T R(p, f_τ),

with C chosen so w_t(p) > 0. Its expectation is

  W(p) = E[w_t(p)] = TC - T · R(p)                                     ... (5)

which is a decreasing linear function of R(p). Thus minimising Bayes risk is equivalent to maximising expected fitness.

**Reproduction.** At the end of each generation, agents reproduce with probability proportional to realised fitness: rate w_t(p)/w̄_t, where w̄_t = Σ_p π_t(p) w_t(p).

**Mutation.** Offspring encodings mutate according to a kernel Q_ε on Ω. We do **not** assume full support: biologically realistic viability/development constraints may forbid many transitions. A useful concrete family is local per-position mutation plus viability filtering, where only admissible transitions receive positive probability.

**Two dynamics and scope.** Theorems 7.3-7.4 are statements about the deterministic expected-fitness recursion (replace w_t by W). Theorem 7.5 is the stochastic finite-K Wright-Fisher model and links it to that deterministic limit via law of large numbers as K → ∞.

**Lemma 7.0 (Label-equivariance of mutation).** Assume the mutation kernel is percept-label equivariant: for every σ ∈ S_M and encodings p, p':

  Q_ε(σ · p → σ · p') = Q_ε(p → p').

Hence transition probabilities depend only on orbit classes in Ω̄ = Ω/~_X, so the process admits a well-defined quotient mutation kernel. If equivariance fails, all statements should be read on Ω directly rather than on Ω̄.

*Proof.* Immediate from the stated equivariance property. For uniform per-position mutation, equivariance holds because transition probabilities depend only on coordinatewise equality/inequality patterns, which relabeling preserves. □

### 7.2. Price's Equation

For a fixed generation, write w(p) := w_t(p) for notational simplicity.

**Theorem 7.1 (Price equation for encoding populations).** Let z: Ω → ℝ be any real-valued trait of encodings. The change in population mean of z across one generation satisfies:

  w̄ · Δz̄ = Cov_π(w, z) + E_π[w · δz]                              ... (6)

where:
- Δz̄ = z̄_{t+1} - z̄_t is the generational change in mean z
- Cov_π(w, z) = Σ_p π_t(p)(w(p) - w̄)(z(p) - z̄) is the selection covariance
- δz is the expected change in z due to mutation within a lineage
- The second term E_π[w · δz] is the transmission bias

*Proof.* This is the standard Price equation. Write π'(p) for the frequency after selection but before mutation:

  π'(p) = π_t(p) · w(p)/w̄

Then z̄' = Σ_p π'(p) z(p) = (1/w̄) Σ_p π_t(p) w(p) z(p). So:

  w̄ · z̄' = Σ_p π_t(p) w(p) z(p)
           = Σ_p π_t(p) w(p) z(p) - w̄ z̄ + w̄ z̄
           = Cov_π(w, z) + w̄ z̄

Thus w̄(z̄' - z̄) = Cov_π(w, z), which is the selection-only Price equation.

Including mutation: z̄_{t+1} = Σ_p π'(p) [z(p) + δz(p)] = z̄' + Σ_p π'(p) δz(p).

So: w̄(z̄_{t+1} - z̄_t) = w̄(z̄' - z̄_t) + w̄ · Σ_p π'(p) δz(p)
    = Cov_π(w, z) + Σ_p π_t(p) w(p) δz(p) = Cov_π(w, z) + E_π[w · δz]     □

### 7.3. Fisher's Fundamental Theorem

**Theorem 7.2 (Fitness increase under selection).** Apply Price's equation with z = w (the fitness trait itself):

  w̄ · Δw̄ = Var_π(w) + E_π[w · δw]

*Selection only (ε = 0):* Δw̄ = Var_π(w)/w̄ ≥ 0, with equality iff the population is monomorphic.

*With mutation (ε > 0):* Δw̄ = Var_π(w)/w̄ + E_π[w · δw]/w̄, where the mutation term E_π[w · δw] can be negative (mutations are typically deleterious).

*Proof.* Set z = w in equation (6). Then Cov_π(w, w) = Var_π(w). The result follows. □

**Remark.** For Berke's asexual haploid population (no recombination), the additive genetic variance equals the total phenotypic variance in fitness, so Var_π(w) is the full variance. In sexual diploid populations, Fisher's theorem involves only the additive component; our asexual setting avoids this complication.

### 7.4. Price Equation for Bayes Risk

**Theorem 7.3 (Bayes risk decrease in the deterministic mean-field dynamics).** In the deterministic expected-fitness dynamics (replace w by W from (5)), applying Price with z = R gives

  W̄ · ΔR̄ = Cov_π(W, R) + E_π[W · δR]

and since W(p) = TC - T R(p),

  ΔR̄ = -T · Var_π(R)/W̄ + E_π[W · δR]/W̄                            ... (7)

*Selection only:* ΔR̄ = -T · Var_π(R)/W̄ ≤ 0. Mean Bayes risk strictly decreases unless all types in support have equal Bayes risk.

*Proof.* Apply Theorem 7.1 with trait z = R and fitness w replaced by expected fitness W. Then

  W̄ · ΔR̄ = Cov_π(W, R) + E_π[W · δR].

Using W(p)=TC-TR(p),

  Cov_π(W,R) = Cov_π(TC-TR, R) = -T·Var_π(R),

which yields (7). Under selection only (δR ≡ 0), the right-hand side is -T·Var_π(R) ≤ 0, with equality iff R is constant on the support. □

**Remark (stochastic analogue).** For finite K with realised fitness, equation (7) holds with sampling noise terms; monotonic decrease is then in expectation / mean-field limit, not pathwise for every finite population trajectory.

### 7.5. Quotient-Space Convergence Dynamics

A key structural feature of our model is that expected fitness W(p) depends only on the encoding p, not on the population composition. This is the frequency-independent setting of quasispecies theory.

#### 7.5.1. The Quasispecies Recursion

For the deterministic generation-to-generation dynamics on a canonical representative set Ω_c ⊂ Ω (one encoding per percept-label orbit; equivalently Ω̄ = Ω/~_X), define D = diag(W(p₁), ..., W(p_{|Ω_c|})) and A_ε := Q_ε^T D. The replicator-mutator update is

  x_{g+1} = A_ε x_g / (1^T A_ε x_g)                                  ... (8)

where x_g ∈ Δ^{|Ω_c|} is the encoding-frequency vector at generation g.

By induction,

  x_g = A_ε^g x_0 / (1^T A_ε^g x_0).

Hence fixed points satisfy

  A_ε x* = λ x*                                                        ... (9)

for λ = 1^T A_ε x*. Thus fixed points are Perron eigenvectors of A_ε after simplex normalisation.

#### 7.5.2. The Perron-Frobenius Argument

By Lemma 7.0, recursion (8) descends to the quotient Ω̄ = Ω/~_X (Definition 2.2.1). To avoid label-symmetry multiplicity, we work on a canonical representative set Ω_c (one encoding per orbit), equivalently on Ω̄. In this subsection, all vectors/matrices and equations (8)-(9) are interpreted on Ω_c; for readability, we drop bars.

Define the directed mutation graph G_ε on Ω_c by edges p → q iff Q_ε(p → q) > 0. To avoid clash with the fitness offset constant C in W(p)=TC-TR(p), denote communicating classes by K. Let K₁, ..., K_m be the closed communicating classes. For each closed class K, define the class block

  A_{ε,K} := Q_{ε,K}^T D_K,

where Q_{ε,K} is the mutation kernel restricted to K and D_K is the fitness diagonal restricted to K.

For a given initial condition x_0, let R(x_0) be the set of closed classes reachable from supp(x_0) in G_ε. For each K ∈ R(x_0), define λ_K := ρ(A_{ε,K}) (spectral radius).

Fix a dominant reachable class K* satisfying

  λ_* := λ_{K*} > max_{K ∈ R(x_0)\{K*}} λ_K.                           ... (10)

Inside K*, define the class-wise capacity-aware optimum and gap:

  R*_K = min_{p ∈ K*} R(p),     P*_K = argmin_{p ∈ K*} R(p).

If P*_K ≠ K*, set

  ΔR_gap,K = min_{p ∈ K*\P*_K} [R(p) - R*_K] > 0;

if P*_K = K*, define ΔR_gap,K := 0 (degenerate no-gap case). Choose p*_K ∈ P*_K and define

  W*_K = W(p*_K),   W₂,K = max_{p ∈ K*\P*_K} W(p),   ΔW_K = W*_K - W₂,K = T·ΔR_gap,K,

and A_{0,K} := D_K.

**Theorem 7.4 (Constrained-mutation quasispecies convergence).** Assume:
  (i)  Frequency-independent expected fitness W(p) > 0 for all p ∈ Ω_c;
  (ii) For each closed class K ∈ R(x_0), the block A_{ε,K} is primitive;
  (iii) The dominant-class condition (10) holds.

Then:

(a) **Class-conditional convergence.** The recursion (8) converges to the Perron equilibrium of the dominant reachable class:

  x_g → x*_{K*},

where x*_{K*} is the normalized Perron right eigenvector of A_{ε,K*}, embedded in Ω_c (zero mass outside K*).

(b) **No global uniqueness without strong connectivity.** If multiple reachable closed classes tie for maximal λ_K, the limit need not be unique; asymptotically the process concentrates on the top reachable classes, with weights determined by initial condition and transient connectivity.

(c) **Small-mutation concentration within the dominant class.** If P*_K is a singleton {p*_K} and

  Q_{ε,K*} = I + ε̃(Q_{1,K*} - I) + O(ε²),   ε̃ = Nε(1 - 1/M),

then x*_{K*} → δ_{p*_K} as ε → 0 and, for p ∈ K*, p ≠ p*_K:

  x*_{K*}(p) = ε̃ · W(p*_K) Q_{1,K*}(p*_K → p) / (W(p*_K) - W(p)) + O(ε²).

(d) **Rate via dominant class spectral ratio.** Let λ₁,K > |λ₂,K| ≥ ... be eigenvalues of A_{ε,K*} by modulus and define

  ρ_{K*} = |λ₂,K|/λ₁,K,   η_{ε,K} = ‖A_{ε,K*} - A_{0,K}‖₂.

If η_{ε,K} < ΔW_K/2, then

  ρ_{K*} ≤ (W₂,K + η_{ε,K})/(W*_K - η_{ε,K}) < 1.

Define

  θ := max{ ρ_{K*}, max_{K ∈ R(x_0)\{K*}} (λ_K/λ_*) } < 1,         ... (11)

with the convention max over an empty set equals 0.

Then there exist C0>0 and integer m≥1 such that

  ‖x_g - x*_{K*}‖ ≤ C0 · g^{m-1} · θ^g.

If A_{ε,K*} is diagonalizable, m=1 for the within-class contribution.

If η_{ε,K} ≤ ΔW_K/4, then

  1 - ρ_{K*} ≥ ΔW_K/(2W*_K) ≥ ΔW_K/(2TC) = ΔR_gap,K/(2C).

*Proof.* See Appendix D (Lemma D.1, Lemma D.2, Proposition D.3, Proposition D.4, Lemma D.5, Proposition D.6, Corollary D.7). □

**Corollary 7.4.1 (Primitive special case: global convergence).** If the induced mutation kernel on Ω_c is primitive, there is a single closed class K* = Ω_c. Theorem 7.4 reduces to a unique interior fixed point and global convergence from every interior initial condition, with the same spectral-rate bound.

**Corollary 7.4.2 (δ_μ specialization).** In the full-separation feasible-veridical primitive regime (μ separates points and M ≥ N), define the global quantities

  R* = min_{p ∈ Ω_c} R(p),   P* = argmin_{p ∈ Ω_c} R(p),
  ΔR_gap = min_{p ∉ P*}[R(p)-R*],   A₀ = D,   η_ε := ‖A_ε - A₀‖₂,   ΔW := T·ΔR_gap.

Then R*=0 and ΔR_gap ≥ π²_minδ_μ by Theorem 4.1(c). Therefore, under η_ε ≤ ΔW/4:

  1 - ρ_ε ≥ π²_min δ_μ / (2C).

**Remark 7.4.3 (General irreducible-but-periodic extension).** Theorem 7.4 assumes primitivity of each relevant closed block to obtain pointwise convergence and a strict spectral gap ratio < 1. If a dominant closed block is only irreducible with period d>1, normalized iterates need not converge to a single fixed vector; instead, they approach a d-cycle (one limit point per residue class mod d), while Cesàro averages converge to the Perron profile. The accessible-class conclusion remains unchanged (selection still concentrates on dominant reachable classes), but the asymptotic object is periodic rather than stationary. Appendix D.8 records this extension.

![Numerical illustration of Theorem 7.4 and Remark 7.4.3. **(A)** Constrained mutation with two closed classes (K1 dominant, K2 subdominant) and one transient state. From the transient state (reachable to both classes), mean risk drops to the dominant-class asymptote R*_{K1}; from initial support restricted to K2, the population is trapped at R*_{K2}. Dashed lines: theoretical asymptotic risk from Perron eigenvectors of each block. **(B)** Quotient dynamics for the accessible case: mass on K1 (dominant) grows monotonically while K2 and the transient set are driven to zero, consistent with the spectral-radius ordering lambda_{K1} > lambda_{K2}. **(C)** Irreducible period-2 mutation (Remark 7.4.3): the raw trajectory oscillates indefinitely, but the Cesaro mean converges to the Perron profile (dashed).](figures/theorems/fig_theorem_dynamics_panels.png)

#### 7.5.3. Finite-Population Convergence

**Theorem 7.5 (Wright-Fisher under constrained mutation).** For the finite-population Wright-Fisher process with K agents, selection proportional to realised fitness w_t(p), and mutation kernel Q_ε:

(a) The composition chain is finite-state and Markov. Its communication structure is inherited from the mutation viability graph on encodings: the chain decomposes into transient sets and one or more closed communicating classes. Thus a stationary distribution always exists, but uniqueness holds only class-by-class (not globally unless the chain is irreducible).

(b) Restricted to any closed class that is aperiodic and irreducible, the chain has a unique stationary distribution and finite mixing time within that class. If a closed class is irreducible but periodic, the stationary distribution is still unique but trajectory convergence is periodic unless one time-averages.

(c) As K → ∞ (with iid task sampling each generation), the composition process converges on each fixed finite time horizon in probability to the deterministic expected-fitness replicator-mutator recursion (8) on Ω_c. For initial conditions satisfying Theorem 7.4 assumptions and with sufficiently large K, finite-horizon trajectories track the deterministic class-conditional limit (single equilibrium under primitive dominant class; periodic orbit under irreducible periodic dominant class; multi-modal behaviour when several classes are co-dominant).

*Proof.* (a)-(b) are standard finite Markov-chain facts once reducibility is allowed explicitly. For (c), apply the standard density-dependent population-process limit (Ethier & Kurtz 1986, Ch. 11; Etheridge 2009): conditional one-step drift is the expected-fitness map (8), sampling fluctuations are O(1/√K), and class-conditional asymptotics follow from Theorem 7.4 (primitive blocks) together with Remark 7.4.3 / Proposition D.8 (irreducible periodic dominant block). (Morandotti & Orlando 2025 gives a recent rigorous derivation for Moran-type processes in weak-selection scaling.) □

**Remark.** The deterministic timescale is governed by the dominant-class ratio in Theorem 7.4(d). Under primitive mutation this reduces to the global ratio ρ_ε. Corollary 7.4.2 provides the δ_μ specialization in the full-separation feasible-veridical primitive regime. Finite-population Wright-Fisher dynamics add O(1/√K) fluctuations; with constrained mutation, stationarity and mixing should be interpreted within reachable communicating classes.

### 7.6. Combining: Complete Evolutionary Picture

**Theorem 7.6 (Main synthesis).** Under multi-task selection with mutation rate ε > 0 and positive expected fitness:

(a) **Static optimality** (Theorem 4.1): in the full-separation, sufficient-capacity regime, the fully veridical encoding is optimal (unique up to percept-label permutation); in general, ecological optima are capacity-aware minimisers of R on Ω_c.

(b) **Directional selection decomposition** (Theorem 7.3):
  ΔR̄ = -T · Var_π(R)/W̄ + E_π[W·δR]/W̄.
Selection decreases mean risk; mutation contributes a transmission term.

(c) **Deterministic convergence on the quotient** (Theorem 7.4): with constrained mutation, the replicator-mutator recursion converges to the quasispecies equilibrium of the dominant reachable communicating class K* (when dominance is unique). The limiting optimum is therefore the best *accessible* optimum, not necessarily the global optimum on Ω_c.

(d) **Rate statement (explicit form):** convergence is exponential/geometric with ratio controlled by the dominant class (Theorem 7.4(d)). If η_{ε,K} := ‖A_{ε,K*} - A_{0,K}‖₂ ≤ ΔW_K/4, then
  1 - ρ_{K*} ≥ ΔR_gap,K/(2C).
Under primitive mutation, this reduces to the global bound (with the corresponding global gap ΔR_gap)
  1 - ρ_ε ≥ ΔR_gap/(2C).
In the full-separation feasible-veridical primitive regime, Corollary 7.4.2 yields
  1 - ρ_ε ≥ π²_min δ_μ/(2C).

(e) **Finite populations** (Theorem 7.5): with iid task sampling, Wright-Fisher dynamics converge on each fixed finite horizon in probability to the deterministic limit as K → ∞, with O(1/√K) fluctuations; ergodic behaviour is class-conditional unless mutation renders the quotient chain irreducible, and periodic classes require time-averaged interpretation.

---

## 8. Recovery of Prior Results

### 8.1. Hoffman's FBT as the T = 1 Special Case

**Proposition 8.1.** For a single task f, the interface encoding p_f defined by:

  p_f(w₁) = p_f(w₂) iff f(w₁) = f(w₂)

achieves R(p_f, f) = 0 while being non-injective whenever f is non-injective (i.e., whenever two world states yield the same fitness). Any veridical encoding also achieves R(p, f) = 0 but "wastes" percepts distinguishing states with equal fitness.

In the competition, the interface encoding wins in Hoffman's regime not because it has lower Bayes risk (both can attain zero for the single task) but through mutational/neutral-network effects among equal-fitness encodings. Which interface family dominates is then controlled by accessibility and mutation structure, not by a Bayes-risk advantage over all veridical encodings per se.

This is exactly Hoffman's insight: for a single non-injective task, there is no selective pressure for full veridicality. Our framework recovers this as the case where task-equivalence classes are nontrivial (and typically δ_μ is not point-separating).

### 8.2. Berke et al.'s Simulations as Numerical Instance

**Proposition 8.2.** Berke et al.'s setting: N = 11, M = 2, tasks are discretised beta functions.

Our capacity-aware framework predicts a transition in ecological fit as k_T grows with T. Full veridicality is impossible (M < N); the relevant target is the best 2-cell ecological partition. In structured beta families, k_T grows gradually rather than collapsing to N at T = 1, producing a broad crossover rather than a universal sharp threshold.

Berke observe:
- T = 1: 92% interface (matches Proposition 8.1)
- T = 5: early movement away from single-task interface-like encodings
- T = 200: majority near the ecologically optimal coarse partition
- T = 2000: strong concentration near that coarse optimum

These observations are consistent with Theorem 4.4/Corollary 4.5 (family-dependent, capacity-aware transition) and Theorem 7.6 (selection toward the capacity-aware optimum in the mutation-accessible class when separation pressure is present).

| Berke recovery: final risk vs task count | Predicted separation margin vs task count |
| --- | --- |
| ![Simulation 1: final mean Bayes risk versus number of sampled tasks per generation (log T axis). Error bars are run-level standard deviation across replicates; risk decreases from the single-task regime and then plateaus at the M=2 capacity floor.](figures/sim1_sim3/sim1_final_risk_vs_T.png) | ![Simulation 1: estimated empirical separation margin delta_T versus number of tasks. The trend is non-decreasing in expectation, matching Proposition 4.6's graded-cascade prediction.](figures/sim1_sim3/sim1_pred_delta_vs_T.png) |

**Figure 8.1.** Two complementary views of Berke-style recovery. Panel A shows the empirical risk transition with increasing task diversity; Panel B tracks the mechanism (growth of empirical separation pressure) predicted by Proposition 4.6.

### 8.3. Task-Family Geometry and Capacity Threshold

The main transition claim in Theorem 4.4 is family-dependent rather than universal. Figure 8.2A shows that families with better-conditioned task geometry attain much higher full-veridical fractions at the same `T`; for example, in our runs with `M=11`, `gauss_iso` reaches mean full-veridical fraction `0.6265` for `T>=11`, while `beta_narrow` remains near `0.009`. Figure 8.2B isolates the capacity effect: with `M=2`, full veridicality is identically zero across all conditions; with `M=11`, it emerges broadly once separation is sufficiently rich.

| Family geometry effect (M=11) | Capacity contrast (M=2 vs M=11) |
| --- | --- |
| ![Simulation 2: full-veridical fraction versus T by task family. Curves separate strongly by family geometry, matching the condition-number prediction in Theorem 4.4.](figures/sim2/sim2_family_full_vs_T.png) | ![Simulation 2: direct capacity comparison. M=2 cannot realize full veridicality; M=11 supports a clear transition as T grows.](figures/sim2/sim2_capacity_contrast.png) |

**Figure 8.2.** Geometry and capacity jointly control the veridicality transition.

A compact regime summary makes the effect size explicit (Figure 8.2C): `M=2` gives mean full-veridical fraction `0.000` (`n=2340`), while in favorable `M=11` regimes (`T>=11`, `log10(kappa)<=2`) the mean rises to `0.3128` (`n=530`), and for `gauss_iso` with `T>=11` it reaches `0.6265` (`n=240`). For reference, the null random-injective baseline at `N=11` is `11!/11^11 ≈ 1.399e-4`, far below the observed favorable-regime fractions.

![Simulation 2: regime-summary bars for full-veridical fraction. This quantifies the capacity-plus-geometry transition and separates it from a diffusion-like null baseline.](figures/sim2/sim2_strong_claim_regimes.png)

**Figure 8.2C.** Quantitative regime contrast for the strong-claim subsets.

### 8.4. Ecological Weighting in the Lossy Regime

Theorem 4.2 predicts that when full injectivity is impossible, optimal encodings allocate resolution toward task-heavy regions. Figure 8.3A shows this reallocation directly as weight ratios vary; Figure 8.3B shows that evolved final risk tracks the theoretical lossy optimum closely (mean absolute gap about `0.032` in our sweep), supporting the weighted-partition prediction beyond equal-weight settings.

| Region-percept allocation under non-uniform weights | Final evolved risk vs theoretical optimum |
| --- | --- |
| ![Simulation 3: percept allocation by world region as ecological weights change. More heavily weighted regions receive finer representational allocation.](figures/sim1_sim3/sim3_region_percept_allocation.png) | ![Simulation 3: final evolved Bayes risk against theoretical optimum for each weight scenario; points near the diagonal indicate close agreement with Theorem 4.2 predictions.](figures/sim1_sim3/sim3_final_vs_optimal_risk.png) |

**Figure 8.3.** Weighted-task ecology shifts the optimal coarse-graining, and evolutionary outcomes remain near the corresponding optimum.

### 8.5. Heuristic FMB Correspondence

The conceptual correspondence to Frank's force-metric-bias framework is discussed in Appendix C. It is explicitly heuristic and not used by any theorem in the main text.

---

## 9. Discussion

### 9.1. Summary of Contributions

1. **The separation theorem** (Theorem 4.1): Multi-task Bayes risk vanishes iff encoding is injective across task-equivalence classes. Zero risk is feasible iff M ≥ k_μ. Under full separation (δ_μ > 0, so k_μ = N) and sufficient capacity (M ≥ N), full veridicality is optimal (unique up to percept-label symmetry).

2. **The fitness gap** (Theorem 4.1(c)): Quantitative lower bound R(p) ≥ π²_min · δ_μ for every encoding that merges at least one μ-separated pair, proving strict selective pressure against ecologically invalid merges.

3. **Evolutionary convergence** (Theorems 7.3–7.4 and Remark 7.4.3): Price's equation identifies the direction of selection (mean Bayes risk decreases); quasispecies analysis via Perron-Frobenius proves class-conditional convergence of deterministic selection-mutation dynamics to the dominant reachable asymptotic regime on quotient encoding space (equilibrium under primitive blocks, periodic cycle in irreducible periodic blocks, global uniqueness only under primitive mutation); spectral analysis shows rate is governed by the dominant spectral ratio in the primitive case, with explicit lower bound 1-ρ_{K*} ≥ ΔR_gap,K/(2C) under η_{ε,K} := ‖A_{ε,K*}-A_{0,K}‖₂ ≤ ΔW_K/4, and δ_μ specialization in the primitive full-separation regime via Corollary 7.4.2.

4. **Graded separation cascade** (Proposition 4.6, Theorem 4.4): the resolved ecological complexity k_T is monotone non-decreasing as tasks are added. For structured task families (the biologically relevant case), this produces gradual transitions whose width is controlled by the condition number of the task family — a quantitatively testable prediction. Full veridicality is recovered only when both separation is complete and capacity is sufficient.

5. **Spectral control** (Theorem 6.2): In the Gaussian model, everything reduces to the spectrum of the task covariance Σ_c. The bottleneck direction — the task dimension with least diversity — controls both the fitness gap and the convergence rate.

6. **Ecological Umwelt** (§9.4): The equivalence classes [w]_μ provide a mathematical framework that gives formal content to the concept of a species-specific perceptual world. Veridicality is always relative to the task ecology, resolving the Hoffman-Berke debate.

7. **Unification**: Hoffman's FBT and Berke's simulations are recovered as special cases / limiting regimes, with a heuristic bridge to Frank's FMB in Appendix C.

### 9.2. The Role of Cognitive Impenetrability

The entire argument depends on the encoding being *fixed* across tasks. If the organism could switch encodings per task, it could maintain a different (non-veridical) interface for each, and Hoffman's FBT would apply separately to each. Cognitive impenetrability is therefore the mechanism that creates selective pressure for richer shared representations; under mutation-development constraints this pressure can terminate at accessible local optima rather than a global optimum.

This connects to a deep point in information theory. A *sufficient statistic* T(X) for a family of parameters {θ} is one that preserves all information about every θ. A fixed encoding that must serve all tasks is being pressed toward sufficiency for the entire task family. Under full separation (k_μ = N) and sufficient capacity (M ≥ N), the minimum-dimension sufficient encoding is the identity up to relabeling (veridical encoding). Outside that regime, sufficiency is only relative to W/~_μ (ecological veridicality).

### 9.3. Levels of Analysis: Static Optimality, Evolution, and Development

The results in this paper operate at two distinct levels, and conflating them leads to errors.

**Static optimality (§§3–6)** characterises the *destination*: which encoding minimises multi-task Bayes risk, and by how much. These results are properties of the objective function R(p), not of any particular optimisation process. They hold regardless of whether p is found by natural selection, gradient descent, reinforcement learning, exhaustive search, or divine fiat. Any system that approximately minimises R(p) — by whatever mechanism — will arrive at an approximately ecologically veridical encoding. This generality is the reason the framework applies beyond biology.

**Evolutionary dynamics (§7)** proves that one *specific* optimisation process — natural selection modelled at mean-field level as a replicator-mutator recursion on a population of organisms — converges to the dominant mutation-accessible asymptotic regime (equilibrium under primitive mutation blocks; periodic cycle in irreducible periodic blocks; global optimum only under primitive global connectivity). The finite-population Wright-Fisher process tracks this deterministic flow on fixed finite horizons as K grows. The entities undergoing evolution are whole organisms, each carrying an encoding as a heritable trait. Selection operates on whole-organism fitness across all tasks. The Price equation, quasispecies theory, and spectral analysis are tools for analysing this particular population process. They do not apply to within-organism processes such as neural development, learning, or synaptic plasticity.

The distinction matters because a different optimisation process — such as stochastic gradient descent training a neural network — may also converge to an encoding that satisfies the static optimality conditions, but the *convergence proof* would require entirely different mathematics (e.g. the theory of implicit regularisation in gradient descent). The static theorems would still characterise the destination; only the proof of arrival would change. In evolutionary biology, this parallels the standard distinction between the *optimality model* (what should evolution produce?) and the *population-genetic model* (does the evolutionary process actually get there?). Our §§3–6 answer the first question; §7 answers the second, for the specific case of frequency-independent selection with mutation.

This two-level structure reflects a fundamental point about evolutionary theory that Sober (2008, pp. 362–363) articulates with particular clarity:

> The distinction between laws and initial conditions also is important in evolutionary biology. The "laws of motion of populations" are general statements that are conditional in form. They say that if a population has a given set of properties at time t₁ and is subject to this or that evolutionary process then it has various probabilities of exhibiting different properties at time t₂. These laws make no predictions until initial conditions are specified. Duhem's thesis applies to evolutionary biology no less than it applies to physics, though it, of course, needs to be understood probabilistically.

Our theorem layout follows this distinction exactly: §§3–6 state conditional laws (if the task ecology has property X, then the optimal encoding has property Y); §7 adds the initial conditions (mutation structure, starting distribution, population size) that convert these conditional laws into dynamical predictions. The same adaptationist methodology is stressed in Sober (2024): optimality models are methodologically informative when paired with explicit process models and explicit constraint statements — objective landscape first, process on that landscape second, scope limits third (§9.5).

This separation of concerns also clarifies the scope of the Baldwin effect (Baldwin 1896, Hinton & Nowlan 1987): if organisms can *learn* a good encoding within their lifetime (developmental optimisation), this learning does not replace evolutionary convergence but can accelerate it, because learned encodings that improve fitness increase the organism's reproductive success and hence the heritability of the architectural traits that enable such learning. The static optimality results characterise what both development and evolution are converging toward; the dynamic results establish mean-field convergence and its spectral rate, plus finite-population approximation on fixed horizons.

### 9.4. Relative Veridicality and the Ecological Umwelt

A natural objection arises: no species is ever exposed to the full universe of possible tasks. A lineage of primates has never had to navigate the echolocation tasks of bats, the magnetoreception tasks of migratory birds, or the infrared detection tasks of pit vipers. Even over evolutionary time, the task distribution μ that a lineage encounters is a proper subset of all conceivable tasks. Does our theorem then require an unrealistic assumption of universal task exposure?

It does not. The theorem's conclusion is always *relative to the task distribution μ actually encountered*. This is not a weakness but the central insight.

**What the mathematics actually says.** The task distribution μ induces an equivalence relation ~_μ on W (Definition 3.5): world states w₁ and w₂ are equivalent iff they are equal on μ-almost every task (equivalently σ²(w₁, w₂) = 0). The equivalence classes [w]_μ partition the world into groups of states that are, from the organism's ecological standpoint, *functionally identical*. Theorem 4.1 and its generalisation (Theorem 4.2) then say:

- **Across equivalence classes:** The risk-minimising encoding is injective. Distinct classes must receive distinct percepts, because some task distinguishes them and merging incurs positive Bayes risk.
- **Within equivalence classes:** The encoding is *free*. No task distinguishes states within a class, so any assignment is equally fit. This is the gauge freedom of Proposition 4.3.

The result is not absolute veridicality but *ecological veridicality*: the encoding is injective up to the resolution of the task ecology. It preserves every distinction that matters for any task the lineage faces, and nothing more.

**The graded separation cascade.** As the task ecology expands — a species enters a new niche, develops new behaviours, or faces new selective pressures — previously equivalent states may become distinguished. This is a monotone process: once a pair is separated it remains separated (adding tasks can only increase σ²). The number of effective percept categories k_μ = |W/~_μ| grows as a staircase function of ecological complexity. Each step in the staircase represents the emergence of a new perceptual distinction driven by a new task demand. The encoding tracks this cascade: it remains optimally adapted to the current task ecology, becoming progressively more veridical as that ecology diversifies.

**Full veridicality as a limiting case.** The condition δ_μ > 0 for all pairs — which gives full injectivity — is the limit of maximal task diversity. It is an idealisation, just as an ideal gas or a frictionless surface is an idealisation. In practice, every organism inhabits a niche with finite task diversity, and its perception is veridical only up to the resolution of that niche. The theory predicts exactly which distinctions are represented and which are not: the equivalence classes [w]_μ are a formal characterisation of the organism's *perceptual grain*.

**Connection to Umwelt theory.** This framework offers a mathematical formalisation of aspects of von Uexküll's (1934) concept of the *Umwelt* — the species-specific perceptual world. In our model, each species' Umwelt is determined by its task ecology μ. A bat and a primate sharing a forest canopy have different μ, hence different equivalence classes, hence different optimal encodings. Each is ecologically veridical — veridical with respect to its own tasks — without either having privileged access to mind-independent reality. The gauge freedom within equivalence classes provides a formal analogue of what it means for a species to be "blind" to certain aspects of the world: not that it misrepresents them, but that no task exerts selective pressure to represent them at all. We note that this is a formalisation within a specific mathematical model, not a philosophical derivation of the Umwelt concept in its full richness.

**Resolving the philosophical tension.** This framing clarifies the apparent conflict between Hoffman and Berke. Hoffman is correct that fitness-maximisation does not require ontological truth — and indeed, within the unresolved equivalence classes, the encoding is unconstrained by selection (any assignment of percepts to states within a class yields equal fitness). Berke is correct that multi-task selection with cognitive impenetrability creates directional pressure toward preserving task-relevant world structure — and indeed, across resolved classes, the encoding must respect the separation structure induced by σ². Both are right, about different parts of the partition. Our contribution is to show that the partition itself — which classes are resolved and which are not — is determined by a single object: the task distribution μ. The mathematical freedom within equivalence classes is a statement about the fitness landscape (multiple encodings are equally fit), not a metaphysical claim about the organism's experience.

Anderson (2015, p. 1509) arrived at a closely related conclusion by philosophical argument. He proposed that veridicality should not be understood as correspondence with an inaccessible world-in-itself, but rather as congruence between different sets of observables:

> If the fundamental "elements" to be compared in assessing veridicality are observables, the issue of veridicality becomes determining whether the equivalence classes generated by perceptual experience map onto (or into) the set of observables generated by some other given procedure of measurement. In this context, veridicality is a measure of the congruence between different sets of observables; it is not a measure of discordancy between perception and truth, because the latter has no meaning apart from an inaccessible God's eye view.

Our equivalence classes [w]_μ and Theorem 4.1 give this intuition precise mathematical form: ecological veridicality is exactly the condition that the perceptual partition (the encoding's equivalence classes) is at least as fine as the task-induced partition (the equivalence relation ~_μ). The "other procedure of measurement" in Anderson's formulation corresponds, in our framework, to the task ecology itself — the set of fitness-relevant observables that the organism's lineage has encountered.

### 9.5. Limitations and Shared Assumptions

Our convergence theorems (§7) separate two claims that are often conflated. Under constrained mutation, selection provably pushes populations toward the best *accessible* optimum (dominant reachable communicating class). Reaching the global optimum additionally requires strong connectivity of the mutation graph on quotient encoding space (primitive mutation), which is a modelling assumption that may fail under evo-devo viability constraints, canalisation, or strong genotype-phenotype restrictions. The space of possible encodings is combinatorially vast (|Ω| = M^N for unrestricted maps), and realistic mutational neighbourhoods are sparse relative to this space. This is the same constraint-sensitive lesson stressed in philosophy-of-biology critiques of simple adaptationism: treat adaptive hypotheses as one component in a plural causal analysis including developmental and historical alternatives (Pigliucci & Kaplan 2006, ch. 5), and avoid reducing evolutionary explanation to a single cause when selection may be "main but not exclusive" (Sober 2008, p. 361). More broadly, our simulation results (decreasing risk trajectories, increasing veridicality fractions) are what Pigliucci and Kaplan (2006, pp. 4–5) call "statistical shadows" — patterns compatible with the causal model we propose but not, by themselves, proof of that model over alternatives. As they emphasise, "the statistical shadows cannot be used as direct supporting evidence for any particular causal model"; what *can* be done is to derive alternative causal hypotheses, project their expected statistical patterns, and compare these against the observed data. Our analytic strategy follows this counsel: we derive predictions under distinct causal regimes (primitive vs constrained mutation, full-separation vs lossy capacity) and test each against simulation, rather than inferring mechanism from a single goodness-of-fit.

This limitation is shared with every optimality argument in the literature, including Hoffman's FBT. Hoffman's theorem proves that interface encodings are fitter than veridical ones for single tasks, but any dynamic conclusion still depends on a mutation-accessibility model (what transitions are viable) and population process assumptions. If developmental bottlenecks prevent access to some veridical optima, they can equally prevent access to some interface optima. The question "does evolution reach the theoretical optimum?" remains distinct from "what *is* the theoretical optimum?" — our static theorems (§4) answer the latter, while §7 answers the former conditionally on explicit mutation-accessibility assumptions.

A related concern is that cognitive impenetrability — the fixity of encoding across tasks — may itself be an idealisation. If organisms can partially adapt their encoding to the current task (cognitive penetration, attentional modulation, context-dependent gain control), the effective task diversity is reduced and the pressure toward veridicality weakens. Our framework accommodates this as a reduction in the effective separation margin δ_μ: a partially adaptive encoding faces a smaller effective task set (only those tasks for which adaptation is too slow or too costly), yielding ecological veridicality at a coarser grain. Full cognitive penetrability (instantaneous, costless re-encoding per task) recovers Hoffman's FBT as the limiting case, with each task selecting its own interface independently.

Finally, one might object that the separation condition is circular — that the theory predicts veridicality only by assuming, via the task distribution, that world-state distinctions matter. This misreads the logical structure. The separation condition σ²(w₁, w₂) > 0 is an *empirical property* of the task ecology, not a theoretical assumption chosen for convenience. The theorem is a conditional: IF the task ecology separates a pair of states, THEN the optimal encoding distinguishes them. The direction of explanation runs from observable facts about fitness landscapes to predictions about encoding structure, not the other way around. The theory does not tell you which pairs are separated — that is an empirical input, just as Newtonian mechanics does not tell you where the masses are. What the theory provides is the machinery to convert a measured task ecology into precise predictions about which perceptual distinctions are maintained and which are not. Moreover, complete non-separation — the condition under which the theory would have nothing to say — is biologically implausible for any organism with more than one behavioural goal. Any two tasks that rank world states differently will separate at least some pairs. The only question is how many pairs and by how much, which is precisely what δ_μ quantifies.

An additional limitation concerns ecological endogeneity. We treat the task distribution μ as externally given, but in many lineages the task ecology is partly constructed by the organisms themselves (niche construction, environmental engineering). In that regime μ can be history-dependent (μ_t), and mutational accessibility can co-evolve with developmental architecture (Q_{ε,t}). Recent work emphasizing organism-environment feedback (Godfrey-Smith 2024) makes this extension biologically important. Our current theorems are then interpreted piecewise-conditionally: at each ecological-developmental regime, they characterize the direction and accessible asymptotics of selection for that regime.

---

## Appendix A: Proof Details for the Pairwise Variance Identity

**Lemma A.1.** For a discrete random variable Z with P(Z = z_i) = p_i:

  Var(Z) = (1/2) Σ_{i,j} p_i p_j (z_i - z_j)²

*Proof.* 

  Var(Z) = E[Z²] - (E[Z])²
         = Σ_i p_i z_i² - (Σ_i p_i z_i)²
         = Σ_i p_i z_i² · (Σ_j p_j) - (Σ_i p_i z_i)(Σ_j p_j z_j)
         = Σ_{i,j} p_i p_j z_i² - Σ_{i,j} p_i p_j z_i z_j
         = (1/2) Σ_{i,j} p_i p_j (z_i² - 2z_i z_j + z_j²)
         = (1/2) Σ_{i,j} p_i p_j (z_i - z_j)²

(The last line uses the symmetry of the double sum to combine z_i² and z_j² terms.) □

---

## Appendix B: Hoeffding's Inequality

**Theorem (Hoeffding 1963).** Let Y₁, ..., Y_n be independent with a_i ≤ Y_i ≤ b_i. Then:

  P(Ȳ - E[Ȳ] ≤ -t) ≤ exp(-2n²t²/Σ_i(b_i - a_i)²)

For iid variables in [a, b]: P(Ȳ - E[Ȳ] ≤ -t) ≤ exp(-2nt²/(b-a)²).

---

## Appendix C: Heuristic Connection to Frank's FMB Law

This appendix is interpretive rather than theorem-proving. It maps the dynamics of §7 onto Frank's force-metric-bias (FMB) template:

  Δθ = M f + b + ξ

with the following correspondences:
- `f`: local gradient-like selection pressure (`-∇R` under smooth parameterisations of encodings)
- `M`: geometry/metric term (natural-gradient-style preconditioning, e.g. Fisher metric in softmax coordinates)
- `b`: systematic non-gradient drift terms (e.g. momentum/transport terms in algorithmic analogues)
- `ξ`: mutation or sampling noise

Why heuristic only:
- The main model in §§2-7 is discrete over finite encoding classes; FMB is a differential parameter-space description.
- Softmax/Fisher parameterisations introduce gauge/degeneracy subtleties unless explicitly fixed.
- A full equivalence proof would require a separate manifold-level setup and regularity assumptions not needed for the core results.

What is rigorous in the main text is unaffected: static optimality (Theorems 4.1-4.2), concentration (§5), and quotient-space evolutionary convergence with spectral rates (Theorems 7.3-7.6). Appendix C is a conceptual bridge to adjacent learning-dynamics frameworks.

---

## Appendix D: Full Proof of Theorem 7.4 (Reducible, Primitive-Block Case)

We provide the full proof of Theorem 7.4 in the constrained-mutation setting, where A_ε can be reducible.

### D.1. Setup and Frobenius Form

Fix ε>0 and write A := A_ε = Q_ε^T D on Ω_c (dimension n). Since Q_ε ≥ 0 and D is positive diagonal, A is nonnegative and has the same directed support graph as Q_ε.

**Lemma D.1 (Frobenius decomposition and closed-part representation).** There exists a permutation matrix P (ordering transient classes first) such that

  PAP^T =
  [ T   0 ]
  [ U   B ],                                                     ... (D.1)

where T is the transient block (possibly empty) and B = diag(B_1,...,B_m) collects closed communicating-class blocks K_1,...,K_m.

For y_0 := Px_0 = (y_0^tr, y_0^cl), define y_g := PA^g x_0 and z_g := (y_g)_{cl}. Then

  z_g = B^g y_0^cl + Σ_{t=0}^{g-1} B^{g-1-t} U T^t y_0^tr.      ... (D.2)

*Proof.* Frobenius normal form for nonnegative matrices gives (D.1) after a suitable permutation of classes. Expanding powers of a block lower triangular matrix yields

  (PAP^T)^g =
  [ T^g                 0              ]
  [ Σ_{t=0}^{g-1} B^{g-1-t} U T^t   B^g ].

Left-multiplying by y_0 and taking the closed component gives (D.2) after reindexing. □

### D.2. Perron Expansion on Closed Blocks

**Lemma D.2 (Block Perron expansions).** For each closed block B_i, define λ_i := ρ(B_i). Under Theorem 7.4(ii), B_i is primitive, so there exist positive vectors v_i,u_i (u_i^T v_i=1), constants C_i>0, and integer s_i≥1 such that

  B_i^g = λ_i^g v_i u_i^T + R_i(g),                               ... (D.3)
  ‖R_i(g)‖ ≤ C_i g^{s_i-1} r_i^g,

where r_i := max_{k≥2}|λ_{k,i}| < λ_i.

Define

  I_max := { i : K_i reachable from supp(x_0), λ_i = λ_max },
  λ_max := max_{i reachable} λ_i.                                 ... (D.4)

*Proof.* Primitive Perron-Frobenius gives a simple dominant eigenvalue and positive eigenvectors; Jordan decomposition gives the remainder bound with polynomial factor g^{s_i-1}. □

### D.3. Unique Dominant Reachable Class (Theorem 7.4(a))

**Proposition D.3 (Unique dominant reachable class convergence).** Assume Theorem 7.4(iii): there is a unique dominant reachable class K* with λ_* > λ_i for all other reachable i. Then there exist c_*>0, C>0, s≥1, θ∈(0,1), and embedded Perron vector \bar v_* (v_* on K*, zero elsewhere) such that

  A^g x_0 = λ_*^g (c_* \bar v_* + e_g),                           ... (D.5)
  ‖e_g‖ ≤ C g^{s-1} θ^g.

Consequently,

  x_g := A^g x_0/(1^T A^g x_0) → \bar v_*/(1^T\bar v_*) = x*_{K*}. ... (D.6)

*Proof.* Combine Lemma D.1 with Lemma D.2 on each reachable closed block. Every contribution has exponential rate λ_i^g times at most polynomial factors. Uniqueness of λ_* makes all non-K* terms exponentially smaller; their maximum relative rate is absorbed into θ<1. Reachability of K* from supp(x_0) implies positive mass transfer into K* via (D.2), so the Perron projection coefficient c_*>0. Normalize (D.5) to obtain (D.6). □

### D.4. Tie Case (Theorem 7.4(b))

**Proposition D.4 (Tie-case limit set).** If |I_max|>1, then

  A^g x_0 = λ_max^g (Σ_{i∈I_max} c_i \bar v_i + r_g),             ... (D.7)

with c_i≥0 (not all zero) and ‖r_g‖/λ_max^g→0. Therefore every subsequential limit of normalized iterates lies in

  conv{ \bar v_i/(1^T\bar v_i) : i∈I_max }.                       ... (D.8)

In general, the limit is not unique.

*Proof.* Apply Lemmas D.1-D.2 with equal top rates λ_i=λ_max for i∈I_max. Lower-rate reachable blocks vanish relatively. Normalization converts nonnegative leading combinations into convex combinations of normalized Perron rays. Coefficients depend on initial projections/feeding in (D.2). □

### D.5. Small-Mutation Expansion in Dominant Class (Theorem 7.4(c))

**Lemma D.5 (Small-mutation dominant-block perturbation).** In dominant block K*, assume P*_K={p*_K} and

  Q_{ε,K*} = I + ε̃(Q_{1,K*}-I) + O(ε²).

Then A_{ε,K*}=Q_{ε,K*}^T D_{K*} has a simple Perron eigenpair near ε=0, and the normalized Perron vector satisfies

  x*_{K*}(p) =
  ε̃ · W(p*_K)Q_{1,K*}(p*_K→p)/(W(p*_K)-W(p)) + O(ε²),  p≠p*_K,  ... (D.9)

with x*_{K*}→δ_{p*_K} as ε→0.

*Proof.* At ε=0, A_{0,K}=D_K is diagonal with simple top eigenvalue W(p*_K). Analytic perturbation of simple eigenpairs (Kato 1995; Greenbaum-Li-Overton 2020) yields analyticity and first-order correction (D.9). □

### D.6. Spectral-Ratio Bound in Dominant Class (Theorem 7.4(d))

**Proposition D.6 (Dominant-class spectral ratio and convergence rate).** Set A_{0,K}=D_K and η_{ε,K}:=‖A_{ε,K*}-A_{0,K}‖₂. If η_{ε,K}<ΔW_K/2, then

  ρ_{K*}:=|λ_{2,K}|/λ_{1,K}
  ≤ (W₂,K+η_{ε,K})/(W*_K-η_{ε,K}) < 1.                              ... (D.10)

With

  θ := max{ρ_{K*}, max_{i reachable,i≠*}(λ_i/λ_*)} < 1,             ... (D.11)

with the convention max over an empty set equals 0.

there exist C_0>0 and m≥1 such that

  ‖x_g-x*_{K*}‖ ≤ C_0 g^{m-1} θ^g.                                   ... (D.12)

If A_{ε,K*} is diagonalizable, the within-class polynomial factor is absent.

If η_{ε,K}≤ΔW_K/4, then

  1-ρ_{K*} ≥ ΔW_K/(2W*_K) ≥ ΔW_K/(2TC) = ΔR_gap,K/(2C).             ... (D.13)

*Proof.* Since A_{0,K} is diagonal/normal, Bauer-Fike gives

  min_j |λ-W(p_j)| ≤ η_{ε,K}

for every eigenvalue λ of A_{ε,K*}. If η_{ε,K}<ΔW_K/2, disks around W*_K and the rest are disjoint, implying

  λ_{1,K}≥W*_K-η_{ε,K},   |λ_{2,K}|≤W₂,K+η_{ε,K},

which yields (D.10). Proposition D.3 gives dominant-class convergence; replacing its abstract relative rate by the explicit max in (D.11) gives (D.12). The inequality (D.13) follows by direct algebra:

  1-ρ_{K*}
  ≥ 1-(W*_K-ΔW_K+η_{ε,K})/(W*_K-η_{ε,K})
  = (ΔW_K-2η_{ε,K})/(W*_K-η_{ε,K})
  ≥ (ΔW_K/2)/W*_K,

then W*_K≤TC. □

### D.7. Primitive Special Case (Corollaries 7.4.1-7.4.2)

**Corollary D.7 (Primitive global case).** If quotient mutation is primitive on Ω_c, there is one closed class K*=Ω_c. Then Theorem 7.4 reduces to standard primitive PF global convergence. Corollary 7.4.2 follows by substituting Theorem 4.1(c): ΔR_gap ≥ π²_min δ_μ. □

### D.8. Irreducible but Periodic Dominant Class (Extension of Remark 7.4.3)

**Proposition D.8 (Periodic dominant block).** Keep the setup of Theorem 7.4 except allow the dominant reachable block B_* to be irreducible with period d>1 (not primitive). Then:

(a) Normalized iterates need not converge to a single fixed point.

(b) There exist d limit points x^{(0)},...,x^{(d-1)} such that along each residue class, x_{nd+r} → x^{(r)}.

(c) The Cesàro average converges:

  (1/G) Σ_{g=0}^{G-1} x_g → x^{PF}_*,

where x^{PF}_* is the Perron profile on K* (embedded in Ω_c).

*Proof.* Frobenius cyclic decomposition gives a permutation of B_* into d cyclic blocks; B_*^d is block diagonal with primitive diagonal blocks. Applying Proposition D.3 to B_*^d on each residue class yields subsequence limits x^{(r)}. Averaging over residues cancels periodic phases and converges to the Perron profile (standard imprimitive Perron-Frobenius asymptotics; Seneta 2006, Ch. 1). □

---

## Appendix E: Supplementary Computational Figures

The main text carries only the figures needed for the argument arc. Additional corroboration, diagnostics, and finite-size checks are listed here for reproducibility and robustness inspection.

| Supplementary figure | Main purpose | Generated by |
| --- | --- | --- |
| `figures/sim1_sim3/sim1_risk_trajectories_selected_T.png` | Trajectory-level variance and stabilization across `T` in Sim 1 | `julia --project=. evo/scripts/analyze_and_plot_results.jl` |
| `figures/sim2/sim2_family_risk_vs_T.png` | Risk-view companion to the full-veridical transition by family | `julia --project=. evo/scripts/analyze_sim2_summary.jl` |
| `figures/sim2/sim2_kappa_scatter_M11.png` | Scatter-level check of condition-number vs veridicality trend | `julia --project=. evo/scripts/analyze_sim2_summary.jl` |
| `figures/sim1_sim3/sim3_risk_trajectories_by_weight_ratio.png` | Convergence diagnostics for non-uniform weighting scenarios | `julia --project=. evo/scripts/analyze_and_plot_results.jl` |
| `figures/quick_checks/quick_claim_check_panels.png` | Fast stochastic sanity check of the main transition claim | `julia --project=. evo/scripts/quick_claim_check.jl` |
| `figures/quick_checks/deterministic_claim_check.png` | Deterministic replicator-mutator bridge to theorem-level dynamics | `julia --project=. evo/scripts/deterministic_claim_check.jl` |
| `figures/quick_checks/quick_claim_bridge.png` | Parameter-sensitivity bridge (mutation/population/time horizon effects) | `julia --project=. evo/scripts/quick_claim_bridge.jl` |

Numerical summaries for these figures are recorded in `evo/data/README.md` and the figure-specific notes in `evo/figures/*/README.md`.

---

## References

Anderson, B. L. (2015). Where does fitness fit in theories of perception? *Psychonomic Bulletin & Review*, 22, 1507–1511.

Bamieh, B. (2020). A tutorial on matrix perturbation theory (using compact matrix notation). arXiv:2002.05001.

Baxter, J. (2000). A model of inductive bias learning. *Journal of Artificial Intelligence Research*, 12, 149–198.

Baldwin, J. M. (1896). A new factor in evolution. *American Naturalist*, 30, 441–451.

Berke, M. D., Walter-Terrill, R., Jara-Ettinger, J., & Scholl, B. J. (2022). Flexible goals require that inflexible perceptual systems produce veridical representations. *Cognitive Science*, 46(10), e13195.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.

Eigen, M. (1971). Self-organization of matter and the evolution of biological macromolecules. *Naturwissenschaften*, 58, 465–523.

Hinton, G. E., & Nowlan, S. J. (1987). How learning can guide evolution. *Complex Systems*, 1, 495–502.

Etheridge, A. (2009). Some mathematical models from population genetics. Lecture notes, Saint-Flour Summer School in Probability. Available at https://www.stats.ox.ac.uk/~etheridg/orsay/notes1.pdf.

Ethier, S. N., & Kurtz, T. G. (1986). *Markov Processes: Characterization and Convergence*. Wiley.

Fisher, R. A. (1930). *The Genetical Theory of Natural Selection*. Clarendon Press.

Frank, S. A. (2025). The Price equation reveals a universal force–metric–bias law of algorithmic learning and natural selection. *Entropy*, 27, 1129.

Glynn, P. W., & Desai, P. Y. (2018). A probabilistic proof of the Perron-Frobenius theorem. arXiv:1808.04964.

Godfrey-Smith, P. (2009). *Darwinian Populations and Natural Selection*. Oxford University Press.

Godfrey-Smith, P. (2024). *Living on Earth: Forests, Corals, Consciousness, and the Making of the World*. Farrar, Straus and Giroux.

Greenbaum, A., Li, R.-C., & Overton, M. L. (2020). First-order perturbation theory for eigenvalues and eigenvectors. *SIAM Review*, 62(2), 463–482. arXiv:1903.00785.

Hofbauer, J., & Sigmund, K. (1998). *Evolutionary Games and Population Dynamics*. Cambridge University Press.

Hoffman, D. D. (2019). *The Case Against Reality*. W. W. Norton.

Hoffman, D. D., Singh, M., & Prakash, C. (2015). The interface theory of perception. *Psychonomic Bulletin & Review*, 22, 1480–1506.

Hoffman, D. D., & Singh, M. (2012). Computational evolutionary perception. *Perception*, 41, 1073–1091.

Jones, B. L. (1976). Some models for selection of biological macromolecules with time varying constraints. *Bulletin of Mathematical Biology*, 38, 15–28.

Kato, T. (1995). *Perturbation Theory for Linear Operators*. Springer.

Maurer, A., Pontil, M., & Romera-Paredes, B. (2016). The benefit of multitask representation learning. *JMLR*, 17, 1–32.

Morandotti, M., & Orlando, G. (2025). Replicator dynamics as the large population limit of a discrete Moran process in the weak selection regime. arXiv:2501.12688.

Nowak, M. A. (2006). *Evolutionary Dynamics*. Harvard University Press.

Pigliucci, M. (2008). Is evolvability evolvable? *Nature Reviews Genetics*, 9, 75–82.

Pigliucci, M., & Kaplan, J. (2006). *Making Sense of Evolution: The Conceptual Foundations of Evolutionary Biology*. University of Chicago Press.

Prakash, C., Fields, C., Hoffman, D. D., Prentner, R., & Singh, M. (2020). Fact, fiction, and fitness. *Entropy*, 22(5), 514.

Prakash, C., Stephens, K. D., Hoffman, D. D., Singh, M., & Fields, C. (2021). Fitness beats truth in the evolution of perception. *Acta Biotheoretica*, 69, 319–341.

Price, G. R. (1970). Selection and covariance. *Nature*, 227, 520–521.

Price, G. R. (1972). Extension of covariance selection mathematics. *Annals of Human Genetics*, 35, 485–490.

Seneta, E. (2006). *Non-negative Matrices and Markov Chains*. Springer.

Sober, E. (1984). *The Nature of Selection: Evolutionary Theory in Philosophical Focus*. MIT Press.

Sober, E. (1993). *Philosophy of Biology*. Westview Press.

Sober, E. (2008). *Evidence and Evolution: The Logic Behind the Science*. Cambridge University Press.

Sober, E. (2024). *The Philosophy of Evolutionary Theory*. Cambridge University Press.

Sternberg, S. (2010). The Perron-Frobenius theorem. Ch. 9 in *Dynamical Systems*. Dover. Available at https://www.math.miami.edu/~armstrong/685fa12/sternberg_perron_frobenius.pdf.

Thompson, C. J., & McBride, J. L. (1974). On Eigen's theory of the self-organization of matter and the evolution of biological macromolecules. *Mathematical Biosciences*, 21, 127–142.

von Uexküll, J. (1934). *A Foray into the Worlds of Animals and Humans* (trans. J. D. O'Neil, 2010). University of Minnesota Press.
