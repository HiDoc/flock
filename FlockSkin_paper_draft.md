# FlockSkin: Skinning Weight Fields as Attractors of Learned Local Flocking Dynamics

**Draft v0.1 — internal review copy.** Target format: short paper / workshop track (graphics or self-organization venue). Section 5 (Results) is a pre-registered stub to be filled by the V0 experimental campaign. Citations marked `[⋆]` need verification or completion.

Authors: Constantine ⟨affiliation⟩, ⟨co-authors TBD⟩

---

## Abstract

Modern learned rigging and skinning systems increasingly rely on large autoregressive models that predict a rig in a single global pass. We investigate the opposite regime: whether a skinning weight field can arise as the **attractor of a small, shared-weight recurrent cell** applying the same local rules at every vertex — a learned analogue of Reynolds' flocking rules, in the lineage of neural cellular automata. Each vertex holds a hidden state and a distribution over candidate bones; at every step it exchanges messages with its mesh neighbors and candidate bones, and applies a **damped delta to its weight logits**, so that fixed points of the dynamics are exactly the states where the cell chooses to stop. We frame convergence — *quiescence* — as a fourth flocking rule that Reynolds' boids lack and that must be learned. Rather than claiming state of the art, we **pre-register three falsifiable properties**: a recurrence gain over an equally deep feed-forward network with unshared weights, attractor stability under extended iteration, and localized repair of corrupted weights without collateral damage — together with quantitative thresholds, baselines that include hand-coded flocking rules, and targeted ablations. The complete study, on procedural shapes and decimated character meshes with ground-truth skeletons, is designed to run on a single consumer laptop (Apple M4 Pro, MLX), with each experiment completing in under one hour.

**Keywords:** skinning, rigging, self-organization, neural cellular automata, graph neural networks, flocking, iterative refinement.

---

## 1. Introduction

Automatic rigging and skinning have recently been dominated by increasingly large learned models that map a mesh to a rig in one global inference pass, often autoregressively [RigNet; SkinTokens ⋆; UniRig ⋆; Puppeteer ⋆]. These systems are effective but structurally monolithic: the solution is emitted, not maintained. They offer no native notion of *convergence*, no graceful response to local perturbation, and no mechanism by which an artist's local correction propagates while the rest of the solution is left untouched.

This paper studies a different computational substrate for the same problem. In Reynolds' classical flocking model [Reynolds 1987], three purely local steering rules — separation, alignment, cohesion — produce coherent global motion with no central coordinator, and the collective reacts to perturbation (a predator strike) by locally reorganizing and reconverging. We ask whether a skinning weight field admits the same description: a global order (a smooth, sparse, anatomically separated weight field) produced and *maintained* by simple local rules, iterated to a fixed point. Two departures from Reynolds are deliberate. First, the rules are **learned** rather than hand-coded, placing the system in the lineage that runs from cellular automata through Neural Cellular Automata [Mordvintsev et al. 2020] to Graph NCA [Grattarola et al. 2021]. Second, the target regime is inverted: a murmuration never converges, whereas our system must — we treat *quiescence* as a fourth rule, absent from boids, that the dynamics must acquire.

We deliberately scope this first study to the cleanest falsifiable setting: **skinning weight refinement with a ground-truth skeleton**, on meshes of ~1k vertices. The scientific question is not "can a network skin a mesh" — feed-forward graph networks already can — but whether *recurrent application of one shared local rule* buys measurable properties that depth alone does not.

Our contributions are: **(1)** a formulation of skinning as the attractor of a learned local dynamical system, with a parameterization (damped deltas on bone logits, gated state updates) chosen so that fixed points are representable and reachable; **(2)** a reading of the learned dynamics in terms of the three Reynolds rules plus quiescence, made operational through a hand-coded flocking baseline that serves as a necessity test for learning; **(3)** a pre-registered experimental protocol — thresholds, baselines, ablations, and a candidate-coverage oracle — that separates the value of recurrence from raw depth and can refute the hypothesis quickly; **(4)** an implementation designed for a single consumer laptop (MLX / Apple Silicon), making every experiment in the paper reproducible in about one hour of compute.

## 2. Background and Related Work

**Flocking and self-organization.** Reynolds' boids [1987] established that separation, alignment, and cohesion — each computed from immediate neighbors — suffice for emergent collective motion; the model underpins decades of work in crowd animation, swarm robotics, and multi-agent systems, and flock-like dynamics are documented in human crowds [⋆]. Neural Cellular Automata [Mordvintsev et al. 2020] learn local update rules that grow and *stabilize* a target pattern, introducing the pool-based training and stochastic updates we adapt; Graph NCA [Grattarola et al. 2021] transports the idea to arbitrary graphs. Our target regime — convergence to a task-defined fixed point — is the morphogenetic/consensus regime of this family rather than the perpetual-motion regime of boids.

**Iterative refinement and fixed points in learning.** Weight-shared recurrent refinement of a field with a gated cell and delta updates is the mechanism behind RAFT's optical-flow estimator [Teed & Deng 2020], which we regard as the closest architectural precedent in another domain. Deep Equilibrium Models [Bai et al. 2019] make the fixed point itself the object of training; we do not use implicit differentiation, but our stability loss and random-horizon supervision pursue the same attractor structure with cheaper machinery.

**Rigging and skinning.** Classical geometric methods bind vertices to bones by heat diffusion [Baran & Popović 2007], geodesic voxel binding [Dionne & de Lasa 2013], or bounded biharmonic weights [Jacobson et al. 2011]; they remain the practitioner's baseline and supply our initializations. RigNet [Xu et al. 2020] introduced the standard learned benchmark and its joint/bone metrics. Recent large autoregressive systems [SkinTokens ⋆; UniRig ⋆; Puppeteer ⋆] represent the current state of the art at scale; we position our study as orthogonal — a controlled test of a different substrate — and defer any comparison at scale to future work. Equivariant message passing [Satorras et al. 2021] becomes relevant in our roadmap when joints, not weights, are the moving agents.

## 3. Method

### 3.1 Problem statement and notation

Let a mesh $M$ have vertices $x_v \in \mathbb{R}^3$, $v \in \{1,\dots,V\}$, with a fixed neighbor set $\mathcal{N}_v$ ($K$ = 8, one-ring capped, KNN-completed) and a ground-truth skeleton whose bones are indexed by $b$. Each vertex is assigned a **candidate set** $\mathcal{B}_v$ of $B$ = 8 bones (§3.3). Skinning weights $w_v \in \Delta^{B-1}$ live on the simplex over $\mathcal{B}_v$. Given an initial weight field $w^{(0)}$ — corrupted ground truth, a naive geometric binding, or near-uniform — the system must produce a trajectory $w^{(0)} \to w^{(1)} \to \dots$ that converges to a field matching the ground truth $w^{*}$ *in deformation space*, remains there under further iteration, and returns there after localized perturbation.

### 3.2 State and parameterization

Each vertex carries a hidden state $h_v \in \mathbb{R}^{32}$ (initialized to zero) and logits $z_v \in \mathbb{R}^{B}$ over its candidates, with

$$w_v = \mathrm{softmax}(z_v / \tau), \qquad z_v^{(0)} = \log(w_v^{(0)} + \varepsilon), \quad \tau = 1 .$$

Two parameterization choices are load-bearing. **(i) Damped deltas, not absolute predictions.** The cell emits an increment $\Delta z_v$ applied as $z \leftarrow z + \alpha\,\Delta z$ with fixed $\alpha = 0.25$. Predicting $z$ absolutely would let the final iteration overwrite the trajectory, collapsing the system into a feed-forward network in disguise; with deltas, a fixed point is precisely a state where the cell outputs $\Delta z \to 0$ — quiescence is representable in the action space. **(ii) A gated state update.** Bare NCA-style additive updates diverge easily; a GRU gate gives the system a learned mechanism for *not* changing.

### 3.3 Precomputed geometric context

All geometry is computed offline, once per mesh, on CPU. Per vertex: normal invariants, approximate curvature, local area. Per edge $(u,v)$: length and the invariant dot products between edge direction and the two normals. Per pair $(v,b)$: Euclidean point-to-segment distance, **diffusion distance** from vertex to bone (heat method [Crane et al. 2013 ⋆]), clamped abscissa along the bone, normal-to-axis angle, bone length, and relative hierarchy depth. Candidate sets $\mathcal{B}_v$ are the top-$B$ bones by diffusion distance. This is where anatomical structure enters: intrinsic (geodesic) proximity is *injected* as a feature, not expected to emerge from Euclidean rules — local dynamics cannot exclude a bone that is Euclidean-close but geodesically remote unless the metric can see the difference (§3.5, separation). Only invariant features are used; equivariance is obtained at the level of the task by random rotation augmentation.

### 3.4 The recurrent cell

One iteration applies the following, with all functions shared across vertices and across iterations ($\phi_\cdot$ are 2-layer MLPs of width 96; total ≈ 40–60k parameters):

$$m_{u\to v} = \phi_m\big(h_u, h_v, e_{uv}\big), \qquad m_v = \big[\textstyle\mathrm{mean}_{u\in\mathcal{N}_v},\ \max_{u\in\mathcal{N}_v}\big]\, m_{u\to v} \tag{1}$$

$$q_{vb} = \phi_b\big(h_v, f_{vb}, w_{vb}\big), \qquad q_v = \mathrm{mean}_{b\in\mathcal{B}_v}\, q_{vb} \tag{2}$$

$$h_v \leftarrow \mathrm{GRU}\big(h_v,\ [\,m_v, q_v\,]\big) \tag{3}$$

$$\Delta z_{vb} = \phi_o\big(h_v, f_{vb}, w_{vb}\big), \qquad z_v \leftarrow z_v + \alpha\, \Delta z_v \tag{4}$$

$$w_v = \mathrm{softmax}(z_v/\tau) \tag{5}$$

Equation (1) is diffusion along the surface; equation (2) is a long-range shortcut to the relevant bones, which compensates the limited receptive field of a 1-ring automaton (after $T$ steps, information travels $T$ rings, while a limb spans 15–30 at this resolution). An optional stochastic per-vertex update mask (fire rate 0.5, as in NCA practice) is evaluated as an ablation.

### 3.5 A flocking reading of the dynamics

The learned cell is intended to occupy the behavioral space that Reynolds' rules span, transported from positions to weight distributions. **Separation** — short-range repulsion — becomes the maintenance of sharp boundaries between regions of influence: mass must not bleed between geodesically distinct parts, which is only expressible because the metric in $f_{vb}$ is intrinsic (§3.3). **Alignment** becomes local agreement of neighboring weight distributions — the smoothness of the field — carried by the mesh channel (1). **Cohesion** becomes the attraction of weight mass toward the relevant bones and the convergence of a limb's vertices onto a common support, carried by the bone channel (2). The fourth behavior, **quiescence**, has no counterpart in boids: a flock never stops, whereas our system must, and the damped-delta action space (4), the gate (3), and the stability loss (8) exist to make stopping both representable and rewarded. This reading is made falsifiable twice over: a hand-coded implementation of the three rules serves as a baseline (§4.2, B5) — if fixed rules suffice, learning is unnecessary — and a post-hoc probe correlates the learned $\Delta z$ with the outputs of those fixed rules, asking whether the learned behavior decomposes into the Reynolds taxonomy at all.

### 3.6 Training as attractor shaping

Training must produce not a map but a *vector field* with the right fixed points. Three mechanisms cooperate.

**Pool-based truncated BPTT.** A pool of $P{=}1024$ persistent states $(h, z)$ spans instances of all "ages" (0 to ~100 lifetime steps). Each optimizer step samples a batch of 32 pool entries, re-initializes a fraction with fresh corruptions, re-corrupts a fraction of already-converged entries locally, injects a fraction of clean ground-truth states, unrolls $k{=}4$ steps with gradients, and writes detached states back. The model thereby experiences long dynamics while backpropagating through only $k$ steps [after Mordvintsev et al. 2020].

**Algorithm 1 — one training step**

```
sample 32 entries from pool
1/8 of batch  ← fresh corruption at level C0–C4        (curriculum)
1/8 of batch  ← local re-corruption of converged state  (repair)
1/16 of batch ← clean ground-truth state                (do no harm)
unroll k = 4 cell iterations with gradients
loss (6)–(9) at the final unrolled step (+ overflow steps for (8))
AdamW update; write detached (h, z) back to pool
```

**Random-horizon supervision.** The supervised step index is effectively $T \sim \mathcal{U}\{4,\dots,12\}$ through pool ages, and the loss is applied at the *final* unrolled step only. Supervising every intermediate step teaches the network to finish in one step — a degenerate dynamics; supervising at a random horizon shapes a basin instead.

**Corruption curriculum.** Inputs are drawn from: C0 global logit noise; C1 randomized patches (5–15% of vertices, BFS-grown); C2 mass transfer between hierarchy-adjacent bones on a patch; C3 naive inverse-square-distance binding; C4 near-uniform. Clean states are always present so that the identity on correct inputs is itself a training target.

**Losses.** With $w^{(T)}$ the terminal weights and $\Theta_p$ pose $p$ from a precomputed per-mesh bank (forward kinematics offline, joint perturbations ±25°):

$$\mathcal{L}_{w} = \big\| w^{(T)} - w^{*} \big\|_1 \tag{6}$$

$$\mathcal{L}_{\mathrm{def}} = \textstyle\sum_{p} \big\| \mathrm{LBS}(x, w^{(T)}; \Theta_p) - \mathrm{LBS}(x, w^{*}; \Theta_p) \big\|_2 \tag{7}$$

$$\mathcal{L}_{\mathrm{stab}} = \big\| w^{(T+1)} - w^{(T)} \big\|_1 \quad \text{(overflow steps, unsupervised toward } w^*\text{)} \tag{8}$$

$$\mathcal{L} = \mathcal{L}_w + \lambda_d\, \mathcal{L}_{\mathrm{def}} + \lambda_s\, \mathcal{L}_{\mathrm{stab}}, \qquad \lambda_d = 1 \text{ (after warmup)},\ \lambda_s = 0.05 \tag{9}$$

Equation (7) is the reference objective: artist weights are non-unique — distinct fields yield near-identical deformations — so (6) alone punishes correct solutions and rewards memorized conventions; deformation error is what the task is *for*, and with precomputed transforms it costs one einsum. No Laplacian smoothing is imposed: regularity must be produced by the alignment channel, and its absence would be a finding, not a nuisance.

### 3.7 Implementation

The study is designed for a single Apple M4 Pro laptop (24 GB unified memory) under MLX. All tensor shapes are fixed ($V{=}1024$ padded with masks, $K{=}B{=}8$); message passing is gather-only (index tables $[V,K]$ and $[V,B]$), the entire train step (unroll, loss, gradients) is `mx.compile`d, the dataset is memory-resident, and evaluation synchronizes once per optimizer step. At width 96 the cell holds ≈ 40–60k parameters; a forward iteration costs ≈ 0.4 GFLOP per mesh, giving ≈ 0.2 TFLOP per optimizer step (batch 32, $k{=}4$, backward included) — 5–15 steps/s conservatively, i.e. 10–15k optimizer steps within the one-hour budget. The binding constraint is host-side overhead, not arithmetic, which motivates the fixed-shape, gather-only, single-eval discipline above.

## 4. Experimental Protocol (pre-registered)

### 4.1 Datasets

**D0 — procedural** (200–500 shapes): capsule chains and trees (3–8 bones), tubular meshes, *analytic* ground truth $w^* \propto \mathrm{softmax}(-d_{\mathrm{diff}}/\sigma)$, with difficulty controlled through $\sigma$, joint angles, and deliberately near-touching limbs. D0 provides a clean-room setting where the ground truth has no artist noise. **D1 — characters** (~80): Mixamo-style rigged humanoids decimated to ~1k vertices (QEM), weights reprojected and renormalized, homogeneous skeleton conventions. Splits are 80/20 by mesh plus one held-out category for a weak generalization signal; the object of study is the dynamics, not generalization. A leakage checklist is enforced: candidates and features computed without $w^*$; corruption applied after feature extraction; test poses disjoint from training poses.

### 4.2 Baselines

**B0** the initialization itself (absolute floor); **B1** inverse-square-distance binding over candidates; **B2** a per-vertex MLP (features → weights) with no message passing, equal budget; **B3** a feed-forward GNN of **8 unshared layers**, same width (≈ 8× the parameters) — the critical control for the recurrence claim; **B4** the trained cell truncated to $T{=}1$; **B5** **hand-coded flocking**: iterated fixed rules (alignment = neighbor diffusion of weights; cohesion = attraction toward diffusion-near bones; separation = geodesic attenuation), step size grid-searched, zero learning — the necessity test for learning the rules at all.

### 4.3 Metrics

All metrics are reported **as functions of the iteration count** $T \in \{0,1,2,4,8,16,32\}$: weight L1 (control metric); **deformation error** under 8 held-out poses, normalized by mesh height (reference metric); support precision/recall/Dice at a declared threshold ($w > 10^{-3}$) and mean influence count; **bleeding**, defined operationally as weight mass on bones whose diffusion distance to the vertex exceeds $d_{\max}$; **stability**, the median $\|w^{(t+1)} - w^{(t)}\|_1$ for $t > 8$ and the late-step variance (oscillation detection); **repair**, after corrupting a region $\Omega$ at a converged state: the fraction of induced error absorbed within 8 steps, and the **collateral damage** $\Delta$L1 on $M \setminus \Omega$.

### 4.4 Ablations

(A1) evaluation horizon $T \in \{1,\dots,32\}$; (A2) shared vs. unshared iteration weights; (A3) with/without the state pool; (A4) delta vs. absolute logit prediction; (A5) bone channel off (pure mesh diffusion); (A6) diffusion features off (Euclidean candidates — the direct test of the separation claim); (A7) stochastic update mask; (A8) $\mathcal{L}_{\mathrm{def}}$ off; (A9) hidden size {16, 32, 64}. A1/A2/A4/A6 constitute the minimum decisive set.

### 4.5 Pre-registered hypotheses and thresholds

**H1 (recurrence).** L1 at $T{=}8$ ≤ 0.8 × L1 at $T{=}1$, *and* the shared-weight cell matches B3 within 5% relative at ≈ 8× fewer parameters. Failure refutes the central claim and reduces the system to a feed-forward GNN. **H2 (attractor).** Relative degradation < 5% between $T{=}8$ and $T{=}32$; median update norm decreasing. **H3 (repair).** ≥ 80% of induced error in $\Omega$ absorbed within 8 steps; collateral damage < 10% relative. **H4 (utility).** From the naive initialization (C3), deformation error strictly below B1 *and* B5 — a system beaten by fixed rules does not justify learning. Three seeds per configuration; all per-iteration curves reported for a fixed 8-mesh probe set (2 easy procedural, 2 adversarial procedural with near-touching limbs, 4 characters).

### 4.6 Candidate-coverage oracle

Before any training, we report $\mathrm{cov}(v) = \sum_{b \in \mathcal{B}_v} w^{*}_{vb}$ (mean and 5th percentile per mesh) and the oracle L1 of the best weights restricted to candidates. Coverage below 0.98 mean / 0.90 p5 caps achievable performance regardless of the model and mandates fixing candidate selection first; the oracle floor is drawn on every result curve.

## 5. Results

*To be completed by the V0 campaign.* Planned exhibits: **Table 1** — all methods × {L1, deformation error, Dice, bleeding} at $T{=}8$, D0 and D1; **Figure 2** — per-iteration curves for the cell vs. B3/B4/B5 (H1, H2); **Figure 3** — repair episode: error in $\Omega$ and in $M\setminus\Omega$ vs. steps after corruption (H3); **Table 2** — ablations A1–A9; **Figure 4** — qualitative weight fields along the trajectory, including a failure case. Negative results on any of H1–H4 will be reported as such; the protocol is designed so that a negative outcome is informative about *which* ingredient fails (pool, deltas, intrinsic metric, or recurrence itself).

## 6. Limitations and Scope

The study deliberately assumes a ground-truth skeleton, ~1k-vertex meshes, fixed candidate sets, and softmax weights (no exact zeros; support metrics are threshold-dependent). Datasets are procedural and convention-homogeneous; nothing here claims generalization across artist conventions or categories, nor competitiveness with large autoregressive systems at scale — the claims are strictly the pre-registered properties H1–H4. Free skeletal topology, joint prediction, and hierarchy inference are out of scope.

## 7. Outlook

If H1–H4 hold, the same substrate extends naturally to the skeleton itself: joints become the system's true boids — mobile agents in $\mathbb{R}^3$ updated by an equivariant, EGNN-style steering rule (a learned weighted sum of directions to neighbors), with separation acting between joints and cohesion drawing them toward the medial axis of their support region. Scaling proceeds through precomputed mesh coarsening rather than global attention, and joint birth/death through over-provisioned joints with existence gates rather than literal topological growth. The property that motivates the program is the one no single-pass model offers: a rig that *reconverges* around an artist's local edit.

## References (to complete)

* Reynolds, C. 1987. Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH*.
* Mordvintsev, A. et al. 2020. Growing Neural Cellular Automata. *Distill*.
* Grattarola, D. et al. 2021. Learning Graph Cellular Automata. *NeurIPS*.
* Teed, Z., Deng, J. 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. *ECCV*.
* Bai, S., Kolter, J.Z., Koltun, V. 2019. Deep Equilibrium Models. *NeurIPS*.
* Xu, Z. et al. 2020. RigNet: Neural Rigging for Articulated Characters. *SIGGRAPH*.
* Baran, I., Popović, J. 2007. Automatic rigging and animation of 3D characters. *SIGGRAPH*.
* Dionne, O., de Lasa, M. 2013. Geodesic voxel binding for production character meshes. *SCA*.
* Jacobson, A. et al. 2011. Bounded biharmonic weights for real-time deformation. *SIGGRAPH*.
* Satorras, V.G., Hoogeboom, E., Welling, M. 2021. E(n) Equivariant Graph Neural Networks. *ICML*.
* Crane, K. et al. 2013. Geodesics in heat. *ACM TOG*. `[⋆ verify]`
* SkinTokens `[⋆ add full citation]` · UniRig `[⋆ add full citation]` · Puppeteer `[⋆ add full citation]`
