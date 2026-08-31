# FlockRig / FlockSkin — Concrete R&D plan, V0 → V1

**Target hardware:** MacBook Pro M4 Pro, 24 GB unified memory, MLX/Metal.
**Budget per experiment:** ≈ 1 h of useful training.
**Status:** working document, a refined version of the initial brief.

> **Translation note.** This is the canonical English translation
> ([ADR-0004](adr/0004-english-ubiquitous-language.md)); the original French is
> preserved at [spec.fr.md](spec.fr.md) and remains the document of record for
> the reasoning. Section numbering is identical, so a code comment citing "§4.4"
> resolves in either. One deviation from the text above is recorded in
> [ADR-0003](adr/0003-fixed-shapes-and-npz-schema.md): the development machine is
> an Apple M4 (base), not an M4 Pro, so the §3.4 throughput budget is measured at
> G0.5 rather than inherited.

---

## 0. Theoretical foundation and thesis

### 0.1 Heritage: from Reynolds flocking to a learned dynamics

The project is explicitly anchored in the theory of **flocking** — the
self-organised collective motion of bird flocks, fish schools and swarms, where
coordination emerges without a conductor. Craig Reynolds' founding model (1987,
the *boids*) shows that three purely local steering rules suffice:
**separation** (short-range repulsion — avoid colliding with neighbours),
**alignment** (adopt the average heading and speed of neighbours), and
**cohesion** (long-range attraction towards the neighbours' centre of mass).
From these rules emerge complex global patterns — starling murmurations, the
near-instantaneous collective response to a predator — that nobody programmed.
The same schema describes human crowd dynamics (a few individuals suffice to
bend the whole group) and has fed animation, swarm robotics and multi-agent
systems for forty years.

The bet of FlockRig/FlockSkin is that rigging and skinning belong to this class
of problem: a global order (a coherent weight field, a well-placed skeleton) can
be produced by simple local rules, iterated, with no central coordinator — with
two differences from Reynolds: the rules are **learned** rather than hand-coded,
and the "agents" are the vertices and the joints. The technical lineage is
direct: boids (1987) → cellular automata → Neural Cellular Automata → Graph NCA,
the family this project belongs to.

Correspondence between Reynolds' three rules and the system's mechanisms:

| Reynolds rule | FlockSkin analogue (weights) | FlockRig analogue (joints, V1) | Mechanism in the architecture |
|---|---|---|---|
| **Separation** | sharp boundaries between influence regions; anti-*bleeding* between geodesically distant parts | joints repel each other (anti-redundancy, body coverage) | diffusion features (§2.2) + learned inhibitory deltas (Δz < 0) |
| **Alignment** | a vertex aligns its weight distribution with its neighbours' → regularity of the weight field | the bone axis aligns with the limb structure | mesh messages m_uv + aggregation (§2.3) |
| **Cohesion** | weight mass is drawn towards the relevant bones; vertices of the same limb converge on a common support | joints are drawn towards the medial axis of their support region | the vertex↔bone channel q_vb; in V1, an EGNN update — literally a learned steering rule (§7.3) |

Two properties of flocking are exactly the ones the project seeks to reproduce:
**emergence** (the global order is written nowhere in the rules) and **response
to perturbation** — the flock that scatters and reforms after a predator's attack
is the direct analogue of repair property P3.

### 0.2 Limits of the analogy — and the fourth rule

The analogy is a design guide, not an identity. Three divergences must stay in
mind:

* **The agents do not move (in V0).** Boids travel through space; here the
  vertices are fixed on the mesh and it is their *state* that moves — across the
  weight simplex, for FlockSkin. FlockRig's joints (V1), by contrast, really do
  move in R³: they are the system's true boids.
* **The target regime is the inverse of Reynolds'.** A murmuration never
  converges: flocking lives in perpetual motion. Here, success is **convergence
  to a fixed point** — the regime of consensus and of NCA morphogenesis (grow,
  then stabilise), not that of flight.
* **Reynolds is missing a rule: stopping.** Boids have no quiescent behaviour;
  our system must learn one. This is precisely what justifies the architecture's
  stabilisation choices — damped deltas, GRU gate, stability loss (§2.3, §3.3) —
  and what property P2 tests.

A practical consequence of this grounding: Reynolds' rules **hand-coded** become
a baseline in their own right (B5, §4.2). If three well-tuned fixed rules
suffice, learning the cell is not justified — the theoretical anchoring thus
supplies its own test of necessity.

### 0.3 Thesis and falsification criterion

The thesis is **not** "a small network can skin" — a feed-forward GNN already
does that. The thesis is:

> A local recurrent dynamics with shared weights — a learned heir of the flocking
> rules, of Graph Neural Cellular Automaton type — converges to an **attractor**
> that is a valid skinning solution, **stabilises** there, and **repairs** it
> after local perturbation.

Three falsifiable properties follow:

* **P1 — Useful recurrence:** T iterations of the same cell beat a single pass,
  and equal or beat a feed-forward GNN of equivalent depth with unshared weights
  (hence ~T× larger).
* **P2 — Attractor:** iterations beyond convergence do not degrade the result.
* **P3 — Repair:** a local corruption is resolved without significant collateral
  damage to the correct regions.

If P1 fails, the project reduces to "yet another GNN" and must pivot or stop. The
whole V0 plan is built to settle P1–P3 as quickly as possible.

---

## 1. Critical analysis of the initial brief

### 1.1 Fragile or incorrect hypotheses

**H1 — "Anatomical separation will emerge from local rules."** Fragile, probably
false as stated. Local rules see Euclidean distances; two regions close in
Euclidean but far in geodesic terms (thigh against thigh, fingers, arm against
torso) will produce structural *bleeding* if candidate bones are selected in
Euclidean space: the wrong bone remains a candidate with a strong prior and the
local dynamics has no signal by which to exclude it — in flocking vocabulary, the
*separation* rule cannot operate on a metric blind to mesh topology.
**Correction:** geodesy does not have to emerge, it must be **injected offline** —
precomputed vertex↔bone diffusion distances (heat method, once, at preprocessing,
which respects the "no geodesic recomputed during training" constraint), with
candidate selection on that basis. What V0 tests for emergence is the *refinement
dynamics*, not the mesh's intrinsic geometry.

**H2 — "4–8 steps suffice."** The receptive field of a 1-ring GNCA after T steps
is T rings; a limb spans 15–30 at 1k vertices. This is viable only because the
vertex↔bone channel provides a long-range shortcut (each vertex sees the relevant
bones directly). If ablations show the receptive field to be limiting, the V0
remedy is to add a few **precomputed long edges** (random sampling, or by
diffusion-distance level) — not to bring out multi-scale.

**H3 — "L1 against GT measures quality."** Artist weights are non-unique: several
weight fields give near-identical deformations. L1 alone punishes correct
solutions and rewards memorising conventions. The loss and the reference metric
must be **LBS deformation error under sampled poses** — differentiable, nearly
free with precomputed transforms, and aligned with what actually matters.

**H4 — "It must learn to stop."** Not necessary in V0. A fixed T at evaluation
plus a stability metric (‖w_{t+1} − w_t‖) suffice to test the attractor's
existence. Learned stopping (ACT-style) is a V2 refinement, expensive to debug
and with no falsification value.

**H5 — Softmax and sparsity are in tension.** A softmax never produces exact
zeros: measured sparsity will depend on an arbitrary threshold. In V0, accept
softmax + threshold (reported in the metrics); in V0.5, test sparsemax/entmax if
the *support* is the weak point.

**H6 — "1–2 M parameters."** That is an upper bound, not a target. With H=32 and
MLPs of width 96–128, the cell weighs **~0.1–0.3 M parameters**. This is good
news: the efficiency argument against large autoregressive models becomes
stronger, and there is headroom to widen if underfitting.

**H7 — Premature comparison to SkinTokens/UniRig/Puppeteer.** Moot before V1:
scales, datasets and objectives are incomparable in V0. V0 compares itself to
geometric heuristics and to feed-forward GNNs at equal budget.

### 1.2 Technical risks, by severity

| # | Risk | Symptom | Mitigation |
|---|---|---|---|
| R1 | **Insufficient candidate coverage**: GT mass lives outside the B candidate bones → an unbeatable performance ceiling | L1 stalls at a high floor | **Day-0 oracle** diagnostic (§4.4); fix candidate selection before any training |
| R2 | Divergent or oscillating dynamics (classic for NCAs without stabilisation) | metrics degrade after T_conv | state pool, **damped delta** updates, stability loss, supervision at random T |
| R3 | Collapse to the identity (the model learns to do nothing if the init is close to GT) | no repair, L1 ≈ L1(init) | varied corruption curriculum **including already-correct states** (the "do no harm" objective) |
| R4 | Intermediate supervision → trivial dynamics (all the work in the first step) | per-iteration curve flat after T=1 | supervise at a **random** number of steps (NCA style), not at every step |
| R5 | Throughput killed by Python/kernel-launch overhead (small tensors) | GPU underused, < 2 steps/s | `mx.compile` on the full rollout, fixed shapes, gather only, no `.item()` in the loop |
| R6 | Heterogeneous GT (artist conventions) blurring the signal | noisy loss, no convergence | start procedural (analytic GT) then Mixamo (homogeneous conventions) |
| R7 | GT information leaking into features or candidates | results too good, not reproducible | checklist: candidates and features computed **without** GT weights; corruption applied afterwards |

### 1.3 To cut from V0 (unnecessary complexity)

* Strict equivariance → replaced by **invariant features** (distances, dot
  products, abscissae along bones) + random-rotation augmentation. EGNN-style
  equivariance becomes necessary only when predicting positions (FlockRig, V1).
* Mesh multi-scale → pointless at 1k vertices; it is a tool for scaling up (V1).
* Per-joint cached memory → V0 has no active joint nodes; per-vertex state only.
* Learned stopping → see H4.
* Sparsemax / entmax → V0.5, only if support is the weak point.
* Any form of global attention, RL, or topological growth → consistent with the
  brief, confirmed out of scope.

---

## 2. Minimal V0 architecture — FlockSkin alone, ground-truth skeleton

### 2.1 System state

Per vertex v:

* `h_v ∈ R^32` — hidden state, zero-initialised;
* `z_v ∈ R^B` — logits over the B=8 candidate bones, initialised from the input
  skinning (naive or corrupted): `z_init = log(w_init + ε)`;
* `w_v = softmax(z_v / τ)` — current weights, τ ≈ 1 (temperature fixed in V0).

Everything else (graph, features) is static and precomputed.

### 2.2 Precomputed features (offline, CPU, a few minutes for the whole dataset)

| Scope | Features | Dim ≈ |
|---|---|---|
| Vertex | normal invariants and approximate curvature (local neighbourhood eigenvalues), local area | 4–6 |
| Edge (u,v) | length, dot(n_u, n_v), dot(dir_uv, n_v), dot(dir_uv, n_u) | 4 |
| Pair (v,b) | Euclidean point-segment distance, **precomputed diffusion distance**, abscissa t∈[0,1] along the bone (clamped), normal/bone-axis angle, bone length, relative hierarchy depth of the bone vs the nearest bone | 6–8 |

Graph: one-ring capped at K=8 neighbours (topped up by KNN if valence < 8),
indices `[V, K]` with a mask. Candidates: top-B bones by diffusion distance,
indices `[V, B]` with a mask. **Fixed shapes everywhere**: V=1024 (padding +
mask), K=8, B=8.

### 2.3 The cell (one iteration, weights shared across all iterations)

```
# mesh messages (gather over [V, K])
m_uv   = MLP_msg([h_u, h_v, e_uv])            # width 96, output 32
m_v    = concat(mean_K(m_uv), max_K(m_uv))     # double aggregation, 64

# bone messages (over [V, B])
q_vb   = MLP_bone([h_v, f_vb, w_vb])           # width 96, output 32
q_v    = mean_B(q_vb)                          # 32

# state update — GRU-style gate (stability)
h_v    ← GRUCell(h_v, [m_v, q_v])              # state 32

# weight update — damped DELTAS on the logits
Δz_vb  = MLP_out([h_v, f_vb, w_vb])            # one scalar per pair (v,b)
z_v    ← z_v + α · Δz_v                        # α = 0.25 fixed in V0
w_v    = softmax(z_v / τ)
```

Two structural choices, to be defended by ablation:

* **Deltas, not absolute prediction.** Predicting `z` directly turns the system
  into a disguised feed-forward GNN (the last iteration overwrites everything).
  Damped deltas make the fixed point natural (`Δz → 0` at convergence) and are
  the canonical form of a dynamical system.
* **GRU gate on the state.** Bare NCAs diverge easily; the gate gives the system
  a learned means of *not* changing.

An option to keep in reserve (ablation A7): **stochastic** per-vertex update
(*fire rate* ~0.5, as in the NCAs of Mordvintsev et al.), known to improve
attractor robustness.

### 2.4 Model size

MLP_msg ≈ 10k, MLP_bone ≈ 10k, GRUCell(96→32) ≈ 12k, MLP_out ≈ 4k →
**~40–60k parameters** in the base configuration; ~0.3 M if widened to 128–192.
Far below the brief's bound — deliberately.

### 2.5 Compiled loop (MLX skeleton)

```python
def rollout(params, state, static, k):        # k steps, fixed shapes
    h, z = state
    for _ in range(k):                        # unrolled, k small (2–4)
        h, z = cell(params, h, z, static)
    return h, z

step = mx.compile(train_step)                 # forward k steps + loss + grads
# mx.eval() only at the end of the optimiser step; logging every N steps
```

---

## 3. Training

### 3.1 State pool + truncated BPTT (precise mechanics)

Pool of P=1024 entries; each entry = (mesh id, h, z, metadata: number of steps
already lived, corruption type).

At each optimiser step:

1. sample 32 entries from the pool;
2. **reinitialise** ~1/8 of the batch with fresh corruption (curriculum §3.2) and
   ~1/16 with a clean GT state (do-no-harm);
3. **locally recorrupt** ~1/8 of the already-converged states (repair training);
4. roll out k=4 steps with gradients (truncated BPTT);
5. loss, backward, update;
6. write the **detached** states back into the pool.

Effect: the model sees states at every age (0 to ~100 lived steps) and learns
long dynamics while backpropagating through only 4 steps. This is the central
mechanism for P2 and P3.

### 3.2 Corruption curriculum

| Level | Input corruption | Phase |
|---|---|---|
| C0 | light Gaussian noise on the GT logits (global) | V0 |
| C1 | local corruption: BFS patch of 5–15% of vertices, weights randomised/permuted | V0 |
| C2 | mass transfer from one bone to a hierarchical neighbour over a patch | V0 |
| C3 | naive geometric init (inverse-distance² over candidates) | V0.5 |
| C4 | near-uniform init over candidates + noise | V0.5 |

Mixed per batch, with proportions shifted progressively towards the harder
levels. The batch **always** contains a fraction of clean states (target: leave
them intact).

### 3.3 Losses

| Loss | Definition | Weight | Notes |
|---|---|---|---|
| L_w | L1(w_T, w_GT), T sampled ~ U{4…12} from the pool state, supervision **at the last rollout step only** | 1.0 | supervision at random T = attractor shaping, anti-R4 |
| L_deform | Σ poses ‖LBS(V, w_T) − LBS(V, w_GT)‖₂ ; 2–4 poses drawn from a bank of 16–32 precomputed poses per mesh (offline FK) | 1.0 after ~1k warmup steps | the loss "that counts" (H3); one einsum, nearly free |
| L_stab | ‖w_{t+1} − w_t‖₁ over 1–2 *overflow* steps beyond the supervised step | 0.05 | forces Δ→0 at the fixed point |
| L_support (opt.) | BCE of a support head σ(s_vb) vs GT support | 0.1 | V0.5, only if Dice is low |

No Laplacian regularisation in V0: message diffusion is supposed to produce the
regularity; if it does not, that is information, not a defect to be masked.

### 3.4 Hyperparameters and quantified budget (M4 Pro, 24 GB)

| Parameter | V0 value |
|---|---|
| V / K / B / H | 1024 / 8 / 8 / 32 |
| MLP width | 96 (fallback 128–192) |
| Batch | 32 meshes |
| k (BPTT) | 4 |
| Training T | ~ U{4…12} (via pool age) |
| Pool | 1024 states |
| Optimiser | AdamW, lr 3e-4, cosine, warmup 500, clip 1.0 |
| Precision | fp32 first; bf16 only if profiling shows a throughput wall |
| Target steps | 10–15k optimiser steps in 40–50 min |

Order of magnitude of the cost: ~0.4 GFLOP forward per mesh per step at this
size → ~0.2 TFLOP per optimiser step (batch 32, k=4, backward included). Even at
1 TFLOPS effective (a pessimistic assumption for small kernels on an M4 Pro),
that holds **5–15 steps/s**, i.e. 15–50k steps/h. Compute is not the bottleneck;
**Python overhead is** — hence §3.5. Memory is a non-issue: full dataset + pool +
model < 1 GB.

### 3.5 Apple Silicon / MLX specific optimisations

* `mx.compile` on the full training step (k-step rollout + loss + grads);
  **fixed shapes mandatory** to avoid recompilation — hence V=1024 padding with
  masks.
* **Gather only, never scatter**: neighbourhoods `[V,K]` and candidates `[V,B]`
  as fixed indices → aggregation is a `take` plus a reduction, the most
  favourable pattern for MLX/Metal.
* Lazy evaluation: a single `mx.eval` per optimiser step; no `.item()` / print in
  the hot loop (GPU synchronisation); batched logging every 50–100 steps.
* Unified memory: the entire dataset resident in RAM as MLX tensors, no loading
  pipeline, no host↔device copy.
* LBS vectorised as one einsum over precomputed `[poses, J, 3, 4]` transforms.
* GRUCell written by hand (a few lines) rather than depending on the framework's
  recurrent layers — full control over what is compiled.
* Profile early: if < 5 steps/s, look first for a silent recompilation (a varying
  shape hiding somewhere) before optimising anything else.

---

## 4. Evaluation

### 4.1 Metrics (all traced **as a function of iteration count**: T = 0, 1, 2, 4, 8, 16, 32)

* **Weight L1** vs GT (control metric, not the reference — see H3).
* **Deformation error**: L2 of the vertices under 8 fixed test poses, normalised
  by mesh height. **The reference metric.**
* **Support**: precision / recall / Dice at a declared threshold (w > 1e-3), mean
  number of influences per vertex vs GT.
* **Bleeding**: weight mass carried by bones whose diffusion distance to the
  vertex exceeds a threshold d_max — an operational definition, independent of
  the GT support.
* **Stability**: median of ‖w_{t+1} − w_t‖₁ for t > 8; variance of the weights
  over late steps (oscillation detection).
* **Repair**: after corrupting a region Ω at the converged state, (i) fraction of
  the induced error in Ω resolved after 8 steps; (ii) **collateral damage** = ΔL1
  on M∖Ω.

### 4.2 Baselines

| # | Baseline | What it tests |
|---|---|---|
| B0 | the initialisation itself (no refinement) | absolute floor; failing to beat it is immediate failure |
| B1 | normalised inverse-distance² over the candidates | the standard geometric heuristic |
| B2 | per-vertex MLP (features → w), no messages, equal budget | do the neighbours serve any purpose? |
| B3 | feed-forward GNN, 8 **unshared** layers, same width (≈ 8× the parameters) | **the critical scientific baseline for P1** |
| B4 | the cell itself at T=1 | recurrence vs a single pass |
| B5 | **hand-coded flocking**: iteration of the three transposed fixed Reynolds rules — alignment (weight diffusion between neighbours), cohesion (attraction to bones near in diffusion distance), separation (geodesic attenuation) — step α tuned by grid search, zero learning | do Reynolds' rules suffice as they are, or must they be learned? the test of necessity for the theoretical anchoring (§0.2) |

### 4.3 Indispensable ablations

| # | Ablation | Question settled |
|---|---|---|
| A1 | T ∈ {1, 2, 4, 8, 16, 32} at eval | does recurrence give a monotone gain? (P1, P2) |
| A2 | shared vs unshared weights | iterating one rule vs raw depth |
| A3 | with / without the state pool | is the pool what creates the attractor? |
| A4 | Δz vs absolute logit prediction | the heart of the §2.3 design |
| A5 | bone channel on/off (pure mesh diffusion) | where does long-range information come from? (H2) |
| A6 | diffusion features on/off (Euclidean candidates) | direct test of H1 (anatomical separation) |
| A7 | stochastic update (fire rate 0.5) on/off | attractor robustness |
| A8 | L_deform on/off | H3 |
| A9 | H ∈ {16, 32, 64} | capacity headroom |

**Vital minimum to settle (days 3–5): A1, A2, A4, A6.** The rest can follow.

### 4.4 Mandatory prior diagnostic: candidate coverage oracle

Before any training, compute per vertex the GT weight mass carried by the B
candidates: `coverage(v) = Σ_{b∈cand(v)} w_GT(v,b)`. Report the mean and 5th
percentile per mesh. **If mean < 0.98 or p5 < 0.90, fix candidate selection
before training anything**: this ceiling is unbeatable by the model (R1) and
would contaminate every conclusion. Also report the oracle L1 (the best possible
weights restricted to the candidates) as the theoretical floor on every curve.

---

## 5. Minimal dataset

| Tier | Content | Role | Size |
|---|---|---|---|
| D0 — procedural | capsule chains and trees (3–8 bones), tube meshing, **analytic** GT `w ∝ softmax(−d_diff/σ)` with controlled difficulty (σ, angles, deliberate proximity between limbs) | perfect falsifiability: clean GT, unit tests of the dynamics, no downloads | 200–500 generated |
| D1 — Mixamo | 50–100 characters, decimated to ~1k vertices, homogeneous skeleton | first real data, consistent conventions | ~80 meshes |
| D2 — RigNet dataset | ModelsResource, 2703 models, standard splits | V1 only (benchmark) | — |

Preprocessing (once, CPU, libigl/potpourri3d or equivalent): QEM decimation →
1024, normalisation (unit height, centring), one-ring/KNN, normals, vertex↔bone
diffusion distances, top-8 candidates, reprojection + renormalisation of the GT
weights onto the decimated mesh, a bank of 16–32 poses per mesh (joint
perturbations ±25° + FK). Serialisation to fixed-shape `.npz`.

Splits: 80/20 by mesh, plus **one held-out category** (e.g. D0 quadrupeds) for a
weak generalisation signal. In V0 the question is the *dynamics*, not
generalisation — 100–500 meshes are ample.

Anti-leakage checklist (R7): candidates and features computed without the GT
weights; corruption applied after feature computation; test poses distinct from
training poses.

---

## 6. Quantified go / no-go criteria

| Gate | Criterion | Threshold | If it fails |
|---|---|---|---|
| **G0** | coverage oracle (§4.4) | mean ≥ 0.98; p5 ≥ 0.90 | fix the candidates, do not train |
| **G0.5** | sanity: overfit a single mesh | L1 → ~0 in < 10 min | loop/gradient bug, do not continue |
| **G1** | recurrence (P1) | L1(T=8) ≤ 0.8 × L1(T=1) **and** recurrent model ≥ B3 within ±5% relative with ~8× fewer parameters | 2 days maximum of fixes (smaller α, random T, pool, fire rate); otherwise **reject P1** → pivot to "feed-forward GNN + refinement" or stop |
| **G2** | attractor (P2) | relative degradation < 5% between T=8 and T=32; median stability decreasing | work on L_stab / A7; if still unstable, the self-organisation hypothesis is weakened but not dead — document it |
| **G3** | repair (P3) | ≥ 80% of the induced error in Ω resolved within ≤ 8 steps; collateral damage < 10% relative | check the do-no-harm fraction and the pool recorruption before concluding |
| **G4** | usefulness | deformation error < B1 **and** < B5 (hand-coded flocking) starting from C3 | beaten by a heuristic or by three fixed rules → learning the cell is not justified, insufficient interest for V1 |

**Decision rule:** G0–G4 must be settled in **≤ 5 experiment-days** (see §8). G1
is eliminatory. G2–G3 are the heart of the thesis. G4 conditions the move to
V0.5/V1.

---

## 7. Full roadmap

### 7.1 V0 (weeks 1–3) — settle P1–P3

1. **W1:** D0+D1 data pipeline, G0 oracle, compiled training loop, G0.5, first
   C0–C2 run.
2. **W2:** vital ablations A1/A2/A4/A6, gates G1–G3, per-iteration curves on the
   probe set.
3. **W3:** documented verdict; if green: C3 (naive init), G4, remaining
   ablations.

### 7.2 V0.5 (weeks 4–6) — harden the result

* C4 (near-uniform init): the regime closest to genuine "self-organisation".
* L_support / sparsemax if support is the weak point; precomputed long edges if
  A5 shows a limiting receptive field.
* Robustness study: noise on the features, different decimations, multiple seeds
  (3 minimum per config).
* "Retro-flocking" analysis (interpretability): correlate the learned deltas Δz
  with the outputs of B5's three fixed rules — does the learned cell decompose
  into separation / alignment / cohesion, or has it discovered a behaviour
  Reynolds' taxonomy does not cover? An interesting result either way.
* Short internal technical report: per-iteration figures, ablation table, known
  limitations.

### 7.3 V1 (detailed projection, ~3 months half-time)

**V1 scope:** FlockRig-A + resolution scale-up + weak coupling + first
comparative benchmark. Still M4 Pro as the dev machine; overnight runs of 8–12 h;
occasional GPU rental only if a compute wall is demonstrated (the code stays
portable if the backend is isolated).

**O1 — FlockRig-A: joint repositioning, hierarchy and cardinality known.**
Joints become nodes in their own right: hidden state `h_j` (32–64), **bipartite**
vertex→joint messages (aggregation over each joint's candidate vertices — the
inverse of the V0 candidate table) and joint→joint messages along the hierarchy
edges. The position update must be **equivariant** (unlike skinning, we are
predicting geometry): an EGNN-style scheme,
`Δx_j = Σ_i φ(h_j, h_i, ‖x_i−x_j‖) · (x_i−x_j)/‖·‖`, translation-invariant and
rotation-equivariant by construction. This is, word for word, a Reynolds-style
steering rule — a weighted sum of directions towards neighbours — but learned:
separation between joints (anti-redundancy) and cohesion towards the medial axis
of the support region must emerge from it, which makes FlockRig's joints the
system's true boids (§0.2). Inputs: perturbed joints (position noise, scale);
supervision: GT positions + deformation error with the GT skinning frozen. Same
principles as V0: damped deltas, pool, random T, analogous gates G1'–G3'.

**O2 — Scale up to 4–8k vertices.**
Precomputed QEM coarsening at 3 levels (e.g. 4096 → 1024 → 256), with
prolongation/restriction matrices stored. A "V-cycle" scheme: iterations at the
coarse level, prolongation, fine refinement — the same cell at every level
(inter-level weight sharing: an ablation to settle). The pool stores multi-level
states. Criterion: property A1 (recurrence is useful) must **hold at scale**,
otherwise the V0 result was an artefact of the small size.

**O3 — Weak Rig ↔ Skin coupling.**
Alternating phases: T_r steps with rig active / skin frozen, then the reverse,
before any joint dynamics. Simultaneous coupling is an instability multiplier —
approach it only with two individually stable subsystems. Key metric: does the
full loop (perturbed joints → repaired rig → re-converged skin) beat the frozen
sequential pipeline?

**O4 — Benchmark and positioning.**
Full RigNet dataset, standard splits. Rig metrics: J2J, J2B, B2B, joint
precision/recall, tree validity (trivially satisfied in V1 since the hierarchy is
given — report it anyway, to prepare for V2). Skin metrics: L1, deformation
error, bleeding. Baselines: geometric heuristics + RigNet; SkinTokens, UniRig,
Puppeteer cited as state-of-the-art reference **with no claim of parity in V1** —
the V1 argument is not SOTA but the quality/parameter ratio and the unique
properties (repair, incremental convergence).

**V1 exit criteria:**

1. A1 holds at 4–8k vertices (recurrence still serves);
2. repair demonstrated on **both** the rig and the skin;
3. J2J/B2B within a factor of ~1.5 of published RigNet figures, at < 5 M
   parameters;
4. **interactive demo**: convergence visible in real time in Blender — the artist
   corrects a weight or moves a joint, and the system re-converges locally. This
   is the differentiating product argument (editability) that no large one-shot
   model offers, and the deliverable that makes V1 communicable (technical report
   + video).

**Design note prepared in V1 for V2 — adding/removing joints:** prefer
**over-provisioning** (J_max joints, an existence gate σ(e_j) updated by the
dynamics, a sparsity penalty) to literal topological growth. The existence
problem becomes differentiable, compatible with the fixed shapes MLX demands, and
the "birth/death" of joints can be read off the trajectory of e_j. This is the
realistic path towards steps 6–8 of the initial brief.

### 7.4 V2+ (horizon, not planned)

Joint existence gates (see the note above) → hierarchy prediction (parent pointer
+ tree constraint via MST over the scores, validity measured) → free topology →
high resolution (50k+) via a cascade of levels → and, if the attractor is well
characterised, **distillation into a single pass** for real-time inference — an
acknowledged irony: the recurrence then serves as learning scaffolding.

---

## 8. Experimental protocol

* **Fixed probe set:** 8 meshes (2 easy procedural, 2 hard procedural with close
  limbs, 4 Mixamo), evaluated identically on every run; all metrics traced per
  iteration.
* **Discipline:** one change per run; controlled seeds (3 per config for any
  conclusion); naming `v0_<date>_<change>`; config + metrics serialised with
  every checkpoint.
* **Decision calendar:** D1 data + G0; D2 loop + G0.5 (overfit 1 mesh); D3 first
  real run C0–C2 + curves; D4 A1/A2/A4/A6; D5 verdict G1–G3 written down in black
  and white.
* **Anti-drift rule:** any fix following a gate failure has an explicit budget (2
  days for G1). The goal of V0 is not to make the system work at any cost, it is
  to **know quickly** whether the hypothesis holds.
