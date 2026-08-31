# Glossary — the ubiquitous language

The bridge between [the spec](spec.md) and the code. Every term used in an
identifier appears here with its symbol, its home in the codebase, and the spec
section that defines it. If a term is not in this table, it does not belong in an
identifier.

## The dynamics

| Term | Symbol | Code | Spec |
|---|---|---|---|
| **cell** | — | `backends/mlx/cell.py` | §2.3 |
| | The local rule. One iteration. Shared across all iterations — that sharing is the claim. | | |
| **rollout** | T | `backends/mlx/rollout.py` | §2.5 |
| | Applying the cell T times. Unrolled, never scanned: shapes must stay constant. | | |
| **hidden state** | `h_v` | `CellState.hidden` | §2.1 |
| **logits** | `z_v` | `CellState.logits` | §2.1 |
| | Pre-softmax weights, initialised `log(w_init + ε)`. | | |
| **delta** | `Δz` | `CellConfig.delta_scale` | §2.3 |
| | The damped update `z ← z + α·Δz`. Damped so that `Δz → 0` is a natural fixed point; predicting `z` outright would make this a feed-forward GNN in disguise. | | |
| **damping** | α = 0.25 | `CellConfig.delta_scale` | §2.3 |
| **fire rate** | — | `CellConfig.fire_rate` | §2.3, A7 |
| | Probability a vertex updates on a given step. 1.0 disables; 0.5 is the NCA setting. | | |
| **state pool** | P = 1024 | `domain/dynamics/pool.py` | §3.1 |
| | Persistent states fed back detached. The mechanism behind P2 and P3: it exposes long horizons while gradients stay short. | | |
| **age** | — | `CellState.age` | §3.1 |
| | Iterations a pooled state has already lived. Supervision samples T from it. | | |
| **BPTT steps** | k = 4 | `TrainConfig.bptt_steps` | §3.1 |
| **attractor** | — | P2, gate G2 | §0.3 |
| | The fixed point the dynamics should reach and stay at. | | |
| **repair** | — | P3, gate G3 | §0.3 |
| | Resolving a local corruption without collateral damage elsewhere. The second half is the one that is easy to forget. | | |

## Geometry and data

| Term | Symbol | Code | Spec |
|---|---|---|---|
| **candidates** | B = 8 | `CandidateTable` | §2.2 |
| | The bones a vertex may be influenced by. Selected by diffusion distance, never with knowledge of the ground truth. | | |
| **coverage** | — | `skinning.metrics.coverage` | §4.4 |
| | GT weight mass reachable within the candidate set. Gate G0. Any shortfall is a ceiling training cannot cross. | | |
| **diffusion distance** | `d_diff` | `preprocess/diffusion.py` | H1, §2.2 |
| | Geodesic-like distance by the heat method, computed once offline. Injected rather than left to emerge: local rules see Euclidean space, where thigh and thigh are adjacent. | | |
| **one-ring** | K = 8 | `Mesh.neighbours` | §2.2 |
| | Immediate mesh neighbours, topped up by KNN below valence K. | | |
| **bleeding** | — | `skinning.metrics.bleeding_mass` | §4.1 |
| | Weight mass on geodesically distant bones. Defined against diffusion distance, so it stays meaningful when the GT is itself a convention. | | |
| **support** | — | `WeightField.support` | §4.1, H5 |
| | Which bones influence a vertex, at a declared threshold. Softmax emits no exact zeros, so the threshold is a reported parameter, not a hidden constant. | | |
| **pose bank** | P = 16-32 | `PoseBank` | §3.3, §5 |
| | Precomputed bone transforms. Train and test banks are disjoint, and that is validated. | | |
| **corruption** | C0-C4 | `skinning/corruption.py` | §3.2 |
| | The curriculum. Applied *after* features and candidates — the structural half of the anti-leakage rule. | | |
| **do-no-harm** | — | `CLEAN_FRACTION` | §3.2, R3 |
| | The clean fraction of every batch, whose target is to be left untouched. Without it, collapsing to the identity is a winning strategy. | | |

## Evaluation

| Term | Code | Spec |
|---|---|---|
| **probe set** | `evaluation/probe.py` | §8 |
| | 8 fixed meshes, evaluated identically on every run. | |
| **deformation error** | `metrics.deformation_error` | H3, §4.1 |
| | LBS vertex error under test poses. **The reference metric** — it measures what a rig is for. | |
| **weight L1** | `metrics.weight_l1` | H3 |
| | Control metric only. Artist weights are non-unique; L1 punishes correct solutions and rewards memorising conventions. | |
| **stability** | `metrics.stability` | §4.1 |
| | Median `‖w_{t+1} − w_t‖₁` past T=8. The attractor signal. | |
| **gate** | `evaluation/gates.py` | §6 |
| | A threshold fixed before the experiment runs. G1 is eliminatory. | |
| **baseline** | `evaluation/baselines.py` | §4.2 |
| | B0-B5. B3 (unshared GNN) is the control for P1; B5 (hand-coded Reynolds) is the necessity test for the whole framing. | |
| **ablation** | `evaluation/ablations.py` | §4.3 |
| | A1-A9; A1, A2, A4, A6 are the vital subset. | |

## Flocking, and where the analogy stops

The lineage is boids (1987) → cellular automata → Neural CA → Graph NCA. Reynolds'
three rules map onto the system:

| Reynolds | FlockSkin (weights) | FlockRig (joints, V1) | Mechanism |
|---|---|---|---|
| **separation** | sharp boundaries between influence regions; no bleeding across geodesically distant parts | joints repel — anti-redundancy, body coverage | diffusion features + learned inhibitory deltas (Δz < 0) |
| **alignment** | a vertex aligns its weight distribution with its neighbours' | bone axis aligns with limb structure | mesh messages `m_uv` and aggregation |
| **cohesion** | weight mass is drawn to relevant bones | joints are drawn to the medial axis of their support region | the vertex↔bone channel `q_vb`; in V1, literally a learned steering rule (EGNN) |

Three places the analogy does **not** hold, and they matter more than the three
that do:

1. **The agents do not move** (in V0). Boids travel through space; here vertices
   are fixed and it is their *state* that moves, across the weight simplex. V1's
   joints are the system's only real boids.
2. **The target regime is the opposite of Reynolds'.** A murmuration never
   converges — flocking lives in perpetual motion. Success here is convergence to
   a fixed point: the regime of consensus and NCA morphogenesis, not of flight.
3. **Reynolds has no rule for stopping.** Boids have no quiescent behaviour; this
   system must learn one. That absence is what justifies the damped deltas, the
   GRU gate and `L_stab` — and it is what P2 tests.

Which is why hand-coded Reynolds is baseline B5 and not merely an inspiration:
the framing supplies the instrument that could sink it.
