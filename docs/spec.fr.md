# FlockRig / FlockSkin — Plan R&D concret, V0 → V1

**Cible matérielle :** MacBook Pro M4 Pro, 24 Go de mémoire unifiée, MLX/Metal.
**Budget par expérience :** ≈ 1 h d'entraînement utile.
**Statut :** document de travail, version raffinée du brief initial.

---

## 0. Fondement théorique et thèse

### 0.1 Héritage : du flocking de Reynolds à une dynamique apprise

Le projet s'ancre explicitement dans la théorie du **flocking** — le mouvement collectif auto-organisé des nuées d'oiseaux, bancs de poissons et essaims, où la coordination émerge sans chef d'orchestre. Le modèle fondateur de Craig Reynolds (1987, les *boids*) montre que trois règles de pilotage purement locales suffisent : **séparation** (répulsion courte portée — éviter la collision avec les voisins), **alignement** (adopter le cap et la vitesse moyens des voisins), **cohésion** (attraction longue portée vers le centre de masse des voisins). De ces règles émergent des motifs globaux complexes — murmurations d'étourneaux, réaction collective quasi instantanée à un prédateur — que personne n'a programmés. Le même schéma décrit des dynamiques de foules humaines (quelques individus suffisent à infléchir tout le groupe) et alimente depuis quarante ans l'animation, la robotique en essaim et les systèmes multi-agents.

Le pari de FlockRig/FlockSkin est que le rigging et le skinning appartiennent à cette classe de problèmes : un ordre global (un champ de poids cohérent, un squelette bien placé) peut être produit par des règles locales simples, itérées, sans coordinateur central — avec deux différences par rapport à Reynolds : les règles sont **apprises** plutôt que codées à la main, et les « agents » sont les vertices et les joints. La filiation technique est directe : boids (1987) → automates cellulaires → Neural Cellular Automata → Graph NCA, la famille dans laquelle ce projet s'inscrit.

Correspondance entre les trois règles de Reynolds et les mécanismes du système :

| Règle de Reynolds | Analogue FlockSkin (poids) | Analogue FlockRig (joints, V1) | Mécanisme dans l'architecture |
|---|---|---|---|
| **Séparation** | frontières nettes entre régions d'influence ; anti-*bleeding* entre parties géodésiquement distinctes | les joints se repoussent (anti-redondance, couverture du corps) | features de diffusion (§ 2.2) + deltas inhibiteurs appris (Δz < 0) |
| **Alignement** | un vertex aligne sa distribution de poids sur celles de ses voisins → régularité du champ de poids | l'axe des bones s'aligne sur la structure du membre | messages mesh m_uv + agrégation (§ 2.3) |
| **Cohésion** | la masse de poids est attirée vers les bones pertinents ; les vertices d'un même membre convergent vers un support commun | les joints sont attirés vers l'axe médian de leur région de support | canal vertex↔bone q_vb ; en V1, mise à jour EGNN — littéralement une règle de pilotage apprise (§ 7.3) |

Deux propriétés du flocking sont exactement celles que le projet cherche à reproduire : l'**émergence** (l'ordre global n'est écrit nulle part dans les règles) et la **réaction aux perturbations** — la nuée qui se disperse puis se reforme après l'attaque d'un prédateur est l'analogue direct de la propriété de réparation P3.

### 0.2 Limites de l'analogie — et la quatrième règle

L'analogie est un guide de conception, pas une identité. Trois écarts doivent rester en tête :

* **Les agents ne bougent pas (en V0).** Les boids se déplacent dans l'espace ; ici les vertices sont fixes sur le mesh et c'est leur *état* qui se déplace — sur le simplexe des poids pour FlockSkin. Les joints de FlockRig (V1), eux, se déplacent réellement dans R³ : ce sont les vrais boids du système.
* **Le régime visé est l'inverse du régime de Reynolds.** Une murmuration ne converge jamais : le flocking vit en mouvement perpétuel. Ici, le succès est la **convergence vers un point fixe** — le régime du consensus et de la morphogenèse NCA (croître, puis se stabiliser), pas celui du vol.
* **Il manque une règle chez Reynolds : l'arrêt.** Les boids n'ont aucun comportement de quiescence ; notre système doit l'apprendre. C'est précisément ce qui justifie les choix de stabilisation de l'architecture — deltas amortis, porte GRU, perte de stabilité (§ 2.3, § 3.3) — et ce que teste la propriété P2.

Conséquence pratique de cet ancrage : les règles de Reynolds **codées à la main** deviennent une baseline à part entière (B5, § 4.2). Si trois règles fixes bien réglées suffisent, apprendre la cellule ne se justifie pas — l'ancrage théorique fournit ainsi son propre test de nécessité.

### 0.3 Thèse et critère de falsification

La thèse n'est **pas** « un petit réseau peut skinner » — un GNN feed-forward le fait déjà. La thèse est :

> Une dynamique locale récurrente à poids partagés — héritière apprise des règles de flocking, de type Graph Neural Cellular Automaton — converge vers un **attracteur** qui est une solution de skinning valide, **s'y stabilise**, et la **répare** après perturbation locale.

Trois propriétés falsifiables en découlent :

* **P1 — Récurrence utile :** T itérations de la même cellule battent une passe unique, et égalent ou battent un GNN feed-forward de profondeur équivalente à poids non partagés (donc ~T× plus gros).
* **P2 — Attracteur :** les itérations au-delà de la convergence ne dégradent pas le résultat.
* **P3 — Réparation :** une corruption locale est résorbée sans dommage collatéral significatif sur les régions correctes.

Si P1 échoue, le projet se réduit à « encore un GNN » et doit pivoter ou s'arrêter. Tout le plan V0 est construit pour trancher P1–P3 le plus vite possible.

---

## 1. Analyse critique du brief initial

### 1.1 Hypothèses fragiles ou incorrectes

**H1 — « La séparation anatomique émergera des règles locales. »** Fragile, probablement fausse telle quelle. Les règles locales voient des distances euclidiennes ; deux régions proches en euclidien mais lointaines en géodésique (cuisse contre cuisse, doigts, bras contre torse) produiront du *bleeding* structurel si les bones candidats sont sélectionnés en euclidien : le mauvais bone reste candidat avec un prior fort et la dynamique locale n'a aucun signal pour l'exclure — en vocabulaire de flocking, la règle de *séparation* ne peut pas opérer sur une métrique aveugle à la topologie du mesh. **Correction :** la géodésie n'a pas à émerger, elle doit être **injectée hors-ligne** — distances de diffusion vertex↔bone précalculées (méthode de la chaleur, une fois, au prétraitement, ce qui respecte la contrainte « aucune géodésique recalculée pendant l'entraînement »), et sélection des candidats sur cette base. L'émergence testée en V0 porte sur la *dynamique de raffinement*, pas sur la géométrie intrinsèque du mesh.

**H2 — « 4–8 étapes suffisent. »** Le champ récepteur d'un GNCA 1-ring après T étapes est de T anneaux ; un membre en couvre 15–30 à 1 k vertices. Ce n'est viable que parce que le canal vertex↔bone fournit un raccourci longue portée (chaque vertex voit directement les bones pertinents). Si les ablations montrent que le champ récepteur limite, la parade V0 est d'ajouter quelques **arêtes longues précalculées** (échantillonnage aléatoire ou par niveaux de distance de diffusion) — pas de sortir le multi-échelle.

**H3 — « L1 vers le GT mesure la qualité. »** Les poids d'artistes sont non uniques : plusieurs champs de poids donnent des déformations quasi identiques. L1 seul punit des solutions correctes et récompense la mémorisation de conventions. La perte et la métrique de référence doivent être **l'erreur de déformation LBS sous poses échantillonnées** — différentiable, quasi gratuite avec des transformations précalculées, et alignée sur ce qui compte réellement.

**H4 — « Il faut apprendre à s'arrêter. »** Non nécessaire en V0. Un T fixe à l'évaluation plus une métrique de stabilité (‖w_{t+1}−w_t‖) suffisent à tester l'existence de l'attracteur. L'arrêt appris (type ACT) est un raffinement V2, coûteux à déboguer et sans valeur de falsification.

**H5 — Softmax et sparsité sont en tension.** Un softmax ne produit jamais de zéros exacts : la sparsité mesurée dépendra d'un seuil arbitraire. En V0, assumer softmax + seuil (reporté dans les métriques) ; en V0.5, tester sparsemax/entmax si le *support* est le point faible.

**H6 — « 1–2 M de paramètres. »** C'est une borne haute, pas une cible. Avec H=32 et des MLP de largeur 96–128, la cellule pèse **~0,1–0,3 M de paramètres**. C'est une bonne nouvelle : l'argument d'efficacité face aux gros modèles autorégressifs devient plus fort, et on garde de la marge pour élargir si sous-apprentissage.

**H7 — Comparaison précoce à SkinTokens/UniRig/Puppeteer.** Sans objet avant V1 : échelles, datasets et objectifs incomparables en V0. La V0 se compare à des heuristiques géométriques et à des GNN feed-forward à budget égal.

### 1.2 Risques techniques, classés par gravité

| # | Risque | Symptôme | Mitigation |
|---|--------|----------|------------|
| R1 | **Couverture des candidats insuffisante** : la masse GT vit hors des B bones candidats → plafond de performance indépassable | L1 stagne à un plancher élevé | Diagnostic **oracle jour 0** (§ 4.4) ; corriger la sélection des candidats avant tout entraînement |
| R2 | Dynamique divergente ou oscillante (classique des NCA sans stabilisation) | métriques qui se dégradent après T_conv | pool d'états, mises à jour en **deltas amortis**, perte de stabilité, supervision à T aléatoire |
| R3 | Collapse vers l'identité (le modèle apprend à ne rien faire si l'init est proche du GT) | réparation nulle, L1 ≈ L1(init) | curriculum de corruption varié **incluant des états déjà corrects** (objectif « do no harm ») |
| R4 | Supervision intermédiaire → dynamique triviale (tout le travail au premier pas) | courbe par itération plate après T=1 | superviser à un nombre d'étapes **aléatoire** (style NCA), pas à chaque pas |
| R5 | Débit tué par l'overhead Python/lancements de kernels (petits tenseurs) | GPU sous-utilisé, < 2 pas/s | `mx.compile` sur le rollout complet, formes fixes, gather uniquement, pas de `.item()` dans la boucle |
| R6 | GT hétérogène (conventions d'artistes) qui brouille le signal | perte bruitée, pas de convergence | démarrer procédural (GT analytique) puis Mixamo (conventions homogènes) |
| R7 | Fuite d'information GT dans les features ou les candidats | résultats trop beaux, non reproductibles | checklist : candidats et features calculés **sans** les poids GT ; corruption appliquée après |

### 1.3 À couper de la V0 (complexité inutile)

* Équivariance stricte → remplacée par des **features invariantes** (distances, produits scalaires, abscisses le long des bones) + augmentation par rotations aléatoires. L'équivariance de type EGNN devient nécessaire seulement quand on prédit des positions (FlockRig, V1).
* Multi-échelle du mesh → inutile à 1 k vertices ; c'est un outil de montée en résolution (V1).
* Mémoire cachée par joint → V0 n'a pas de nœuds joints actifs ; état par vertex uniquement.
* Arrêt appris → cf. H4.
* Sparsemax / entmax → V0.5, seulement si le support est le point faible.
* Toute forme d'attention globale, de RL, de croissance topologique → conformes au brief, confirmés hors périmètre.

---

## 2. Architecture minimale V0 — FlockSkin seul, squelette ground truth

### 2.1 État du système

Par vertex v :

* `h_v ∈ R^32` — état caché, initialisé à zéro ;
* `z_v ∈ R^B` — logits sur les B=8 bones candidats, initialisés depuis le skinning d'entrée (naïf ou corrompu) : `z_init = log(w_init + ε)` ;
* `w_v = softmax(z_v / τ)` — poids courants, τ ≈ 1 (température fixe en V0).

Tout le reste (graphe, features) est statique et précalculé.

### 2.2 Features précalculées (offline, CPU, quelques minutes pour tout le dataset)

| Portée | Features | Dim ≈ |
|--------|----------|-------|
| Vertex | invariants de la normale et courbure approx. (valeurs propres locales du voisinage), aire locale | 4–6 |
| Arête (u,v) | longueur, dot(n_u, n_v), dot(dir_uv, n_v), dot(dir_uv, n_u) | 4 |
| Paire (v,b) | distance point-segment euclidienne, **distance de diffusion précalculée**, abscisse t∈[0,1] le long du bone (clampée), angle normale/axe du bone, longueur du bone, profondeur hiérarchique relative du bone vs bone le plus proche | 6–8 |

Graphe : one-ring plafonné à K=8 voisins (complété par KNN si valence < 8), indices `[V, K]` avec masque. Candidats : top-B bones par distance de diffusion, indices `[V, B]` avec masque. **Formes fixes partout** : V=1024 (padding + masque), K=8, B=8.

### 2.3 La cellule (une itération, poids partagés entre toutes les itérations)

```
# messages mesh (gather sur [V, K])
m_uv   = MLP_msg([h_u, h_v, e_uv])            # largeur 96, sortie 32
m_v    = concat(mean_K(m_uv), max_K(m_uv))     # agrégation double, 64

# messages bones (sur [V, B])
q_vb   = MLP_bone([h_v, f_vb, w_vb])           # largeur 96, sortie 32
q_v    = mean_B(q_vb)                          # 32

# mise à jour de l'état — porte type GRU (stabilité)
h_v    ← GRUCell(h_v, [m_v, q_v])              # état 32

# mise à jour des poids — DELTAS amortis sur les logits
Δz_vb  = MLP_out([h_v, f_vb, w_vb])            # scalaire par paire (v,b)
z_v    ← z_v + α · Δz_v                        # α = 0.25 fixe en V0
w_v    = softmax(z_v / τ)
```

Deux choix structurants, à défendre par ablation :

* **Deltas, pas de prédiction absolue.** Prédire `z` directement transforme le système en GNN feed-forward déguisé (la dernière itération écrase tout). Les deltas amortis rendent le point fixe naturel (`Δz → 0` à convergence) et sont la forme canonique d'un système dynamique.
* **Porte GRU sur l'état.** Les NCA nus divergent facilement ; la porte donne au système un moyen appris de ne *pas* changer.

Option à garder sous le coude (ablation A7) : mise à jour **stochastique** par vertex (*fire rate* ~0,5, comme dans les NCA de Mordvintsev et al.), connue pour améliorer la robustesse des attracteurs.

### 2.4 Taille du modèle

MLP_msg ≈ 10 k, MLP_bone ≈ 10 k, GRUCell(96→32) ≈ 12 k, MLP_out ≈ 4 k → **~40–60 k paramètres** en configuration de base ; ~0,3 M si l'on élargit à 128–192. Très en dessous de la borne du brief — c'est voulu.

### 2.5 Boucle compilée (squelette MLX)

```python
def rollout(params, state, static, k):        # k étapes, formes fixes
    h, z = state
    for _ in range(k):                        # déroulé, k petit (2–4)
        h, z = cell(params, h, z, static)
    return h, z

step = mx.compile(train_step)                 # forward k étapes + loss + grads
# mx.eval() uniquement en fin de pas d'optimiseur ; logging tous les N pas
```

---

## 3. Entraînement

### 3.1 State pool + BPTT tronquée (mécanique précise)

Pool de P=1024 entrées ; chaque entrée = (id du mesh, h, z, méta : nb d'étapes déjà vécues, type de corruption).

À chaque pas d'optimiseur :

1. échantillonner 32 entrées du pool ;
2. **ré-initialiser** ~1/8 du batch avec une corruption fraîche (curriculum § 3.2) et ~1/16 avec un état GT propre (do-no-harm) ;
3. **re-corrompre localement** ~1/8 des états déjà convergés (entraînement à la réparation) ;
4. dérouler k=4 étapes avec gradients (BPTT tronquée) ;
5. loss, backward, update ;
6. réécrire les états **détachés** dans le pool.

Effet : le modèle voit des états à tous les âges (0 à ~100 étapes vécues) et apprend des dynamiques longues en ne rétropropageant que sur 4 pas. C'est le mécanisme central pour P2 et P3.

### 3.2 Curriculum de corruption

| Niveau | Corruption de l'entrée | Phase |
|--------|------------------------|-------|
| C0 | bruit gaussien léger sur les logits GT (global) | V0 |
| C1 | corruption locale : patch BFS de 5–15 % des vertices, poids randomisés/permutés | V0 |
| C2 | transfert de masse d'un bone vers un voisin hiérarchique sur un patch | V0 |
| C3 | init géométrique naïve (inverse-distance² sur candidats) | V0.5 |
| C4 | init quasi uniforme sur candidats + bruit | V0.5 |

Mélange par batch, proportions déplacées progressivement vers les niveaux durs. Le batch contient **toujours** une fraction d'états propres (cible : les laisser intacts).

### 3.3 Pertes

| Perte | Définition | Poids | Notes |
|-------|------------|-------|-------|
| L_w | L1(w_T, w_GT), T échantillonné ~ U{4…12} depuis l'état du pool, supervision **au dernier pas du rollout seulement** | 1.0 | supervision à T aléatoire = façonnage d'attracteur, anti-R4 |
| L_deform | Σ poses ‖LBS(V, w_T) − LBS(V, w_GT)‖₂ ; 2–4 poses tirées d'une banque de 16–32 poses précalculées par mesh (FK offline) | 1.0 après ~1 k pas de warmup | la perte « qui compte » (H3) ; un einsum, quasi gratuit |
| L_stab | ‖w_{t+1} − w_t‖₁ sur 1–2 étapes *overflow* au-delà du pas supervisé | 0.05 | impose Δ→0 au point fixe |
| L_support (opt.) | BCE d'une tête de support σ(s_vb) vs support GT | 0.1 | V0.5, seulement si Dice faible |

Pas de régularisation laplacienne en V0 : la diffusion par messages doit produire la régularité ; si elle ne le fait pas, c'est une information, pas un défaut à masquer.

### 3.4 Hyperparamètres et budget chiffré (M4 Pro, 24 Go)

| Paramètre | Valeur V0 |
|-----------|-----------|
| V / K / B / H | 1024 / 8 / 8 / 32 |
| Largeur MLP | 96 (fallback 128–192) |
| Batch | 32 meshes |
| k (BPTT) | 4 |
| T entraînement | ~ U{4…12} (via âge du pool) |
| Pool | 1024 états |
| Optimiseur | AdamW, lr 3e-4, cosine, warmup 500, clip 1.0 |
| Précision | fp32 d'abord ; bf16 seulement si le profilage montre un mur de débit |
| Pas visés | 10–15 k pas d'optimiseur en 40–50 min |

Ordre de grandeur du coût : ~0,4 GFLOP forward par mesh et par étape à cette taille → ~0,2 TFLOP par pas d'optimiseur (batch 32, k=4, backward compris). Même à 1 TFLOPS effectif (hypothèse pessimiste pour des petits kernels sur M4 Pro), on tient **5–15 pas/s**, soit 15–50 k pas/h. Le calcul n'est pas le goulot ; **l'overhead Python l'est** — d'où § 3.5. La mémoire est un non-sujet : dataset complet + pool + modèle < 1 Go.

### 3.5 Optimisations spécifiques Apple Silicon / MLX

* `mx.compile` sur le pas d'entraînement complet (rollout k étapes + loss + grads) ; **formes fixes obligatoires** pour éviter les recompilations — d'où le padding V=1024 avec masques.
* **Gather uniquement, jamais de scatter** : voisinages `[V,K]` et candidats `[V,B]` en indices fixes → l'agrégation est un `take` + réduction, le schéma le plus favorable à MLX/Metal.
* Évaluation paresseuse : un seul `mx.eval` par pas d'optimiseur ; aucun `.item()` / print dans la boucle chaude (synchronisation GPU) ; logging par paquets tous les 50–100 pas.
* Mémoire unifiée : dataset entier résident en RAM comme tenseurs MLX, zéro pipeline de chargement, zéro copie hôte↔device.
* LBS vectorisé en un einsum sur des transformations `[poses, J, 3, 4]` précalculées.
* GRUCell écrite à la main (quelques lignes) plutôt que de dépendre des couches récurrentes du framework — contrôle total de ce qui est compilé.
* Profiler tôt : si < 5 pas/s, chercher d'abord une recompilation silencieuse (forme variable qui traîne) avant d'optimiser quoi que ce soit d'autre.

---

## 4. Évaluation

### 4.1 Métriques (toutes tracées **en fonction du nombre d'itérations** : T = 0, 1, 2, 4, 8, 16, 32)

* **L1 poids** vs GT (métrique de contrôle, pas de référence — cf. H3).
* **Erreur de déformation** : L2 des vertices sous 8 poses de test fixes, normalisée par la hauteur du mesh. **Métrique de référence.**
* **Support** : precision / recall / Dice avec seuil déclaré (w > 1e-3), nombre moyen d'influences par vertex vs GT.
* **Bleeding** : masse de poids portée par des bones dont la distance de diffusion au vertex dépasse un seuil d_max — définition opérationnelle, indépendante du support GT.
* **Stabilité** : médiane de ‖w_{t+1} − w_t‖₁ pour t > 8 ; variance des poids sur les étapes tardives (détection d'oscillation).
* **Réparation** : après corruption d'une région Ω à l'état convergé, (i) fraction de l'erreur induite dans Ω résorbée après 8 étapes ; (ii) **dommage collatéral** = ΔL1 sur M∖Ω.

### 4.2 Baselines

| # | Baseline | Ce qu'elle teste |
|---|----------|------------------|
| B0 | l'initialisation elle-même (aucun raffinement) | plancher absolu ; ne pas la battre = échec immédiat |
| B1 | inverse-distance² normalisée sur les candidats | l'heuristique géométrique standard |
| B2 | MLP par vertex (features → w), sans messages, budget égal | est-ce que les voisins servent à quelque chose ? |
| B3 | GNN feed-forward, 8 couches **non partagées**, même largeur (≈ 8× les paramètres) | **la baseline scientifique critique pour P1** |
| B4 | la cellule elle-même à T=1 | la récurrence vs une passe |
| B5 | **flocking codé à la main** : itération des trois règles fixes de Reynolds transposées — alignement (diffusion des poids entre voisins), cohésion (attraction vers les bones proches en distance de diffusion), séparation (atténuation géodésique) — pas α réglé par grille, zéro apprentissage | les règles de Reynolds suffisent-elles telles quelles, ou faut-il les apprendre ? le test de nécessité de l'ancrage théorique (§ 0.2) |

### 4.3 Ablations indispensables

| # | Ablation | Question tranchée |
|---|----------|-------------------|
| A1 | T ∈ {1, 2, 4, 8, 16, 32} à l'éval | la récurrence apporte-t-elle un gain monotone ? (P1, P2) |
| A2 | poids partagés vs non partagés | l'itération d'une même règle vs profondeur brute |
| A3 | avec / sans state pool | le pool est-il ce qui crée l'attracteur ? |
| A4 | Δz vs prédiction absolue des logits | le cœur du design § 2.3 |
| A5 | canal bone on/off (diffusion mesh pure) | d'où vient l'information longue portée ? (H2) |
| A6 | features de diffusion on/off (candidats euclidiens) | test direct de H1 (séparation anatomique) |
| A7 | mise à jour stochastique (fire rate 0,5) on/off | robustesse de l'attracteur |
| A8 | L_deform on/off | H3 |
| A9 | H ∈ {16, 32, 64} | marge de capacité |

**Minimum vital pour trancher (jours 3–5) : A1, A2, A4, A6.** Le reste peut suivre.

### 4.4 Diagnostic préalable obligatoire : oracle de couverture des candidats

Avant tout entraînement, calculer par vertex la masse de poids GT portée par les B candidats : `coverage(v) = Σ_{b∈cand(v)} w_GT(v,b)`. Reporter moyenne et 5ᵉ percentile par mesh. **Si moyenne < 0,98 ou p5 < 0,90, corriger la sélection des candidats avant d'entraîner quoi que ce soit** : ce plafond est indépassable par le modèle (R1) et contaminerait toutes les conclusions. Reporter aussi l'oracle L1 (meilleurs poids possibles restreints aux candidats) comme plancher théorique sur chaque courbe.

---

## 5. Dataset minimal

| Étage | Contenu | Rôle | Taille |
|-------|---------|------|--------|
| D0 — procédural | chaînes et arbres de capsules (3–8 bones), maillage tube, GT **analytique** `w ∝ softmax(−d_diff/σ)` avec difficulté contrôlée (σ, angles, proximités volontaires entre membres) | falsifiabilité parfaite : GT propre, tests unitaires de la dynamique, aucun téléchargement | 200–500 générés |
| D1 — Mixamo | 50–100 personnages, décimés à ~1 k vertices, squelette homogène | premier réel, conventions cohérentes | ~80 meshes |
| D2 — RigNet dataset | ModelsResource, 2 703 modèles, splits standards | V1 uniquement (benchmark) | — |

Prétraitement (une fois, CPU, libigl/potpourri3d ou équivalent) : décimation QEM → 1024, normalisation (hauteur unité, centrage), one-ring/KNN, normales, distances de diffusion vertex↔bone, candidats top-8, reprojection + renormalisation des poids GT sur le mesh décimé, banque de 16–32 poses par mesh (perturbations articulaires ±25° + FK). Sérialisation en `.npz` à formes fixes.

Splits : 80/20 par mesh, plus **une catégorie tenue à l'écart** (ex. quadrupèdes de D0) pour un signal faible de généralisation. En V0 la question est la *dynamique*, pas la généralisation — 100–500 meshes suffisent largement.

Checklist anti-fuite (R7) : candidats et features calculés sans les poids GT ; corruption appliquée après le calcul des features ; poses de test distinctes des poses d'entraînement.

---

## 6. Critères go / no-go quantifiés

| Porte | Critère | Seuil | Si échec |
|-------|---------|-------|----------|
| **G0** | oracle de couverture (§ 4.4) | moy ≥ 0,98 ; p5 ≥ 0,90 | corriger les candidats, ne pas entraîner |
| **G0.5** | sanity : overfit d'un seul mesh | L1 → ~0 en < 10 min | bug de boucle/gradients, ne pas continuer |
| **G1** | récurrence (P1) | L1(T=8) ≤ 0,8 × L1(T=1) **et** modèle récurrent ≥ B3 à ±5 % relatif avec ~8× moins de paramètres | 2 jours max de correctifs (α plus petit, T aléatoire, pool, fire rate) ; sinon **rejet de P1** → pivot « GNN feed-forward + raffinement » ou arrêt |
| **G2** | attracteur (P2) | dégradation relative < 5 % entre T=8 et T=32 ; stabilité médiane décroissante | travailler L_stab / A7 ; si toujours instable, l'hypothèse d'auto-organisation est affaiblie mais pas morte — documenter |
| **G3** | réparation (P3) | ≥ 80 % de l'erreur induite dans Ω résorbée en ≤ 8 étapes ; dommage collatéral < 10 % relatif | vérifier la fraction do-no-harm et la re-corruption du pool avant de conclure |
| **G4** | utilité | erreur de déformation < B1 **et** < B5 (flocking codé à la main) en partant de C3 | battu par une heuristique ou par trois règles fixes → l'apprentissage de la cellule ne se justifie pas, intérêt insuffisant pour V1 |

**Règle de décision :** G0–G4 doivent être tranchés en **≤ 5 jours d'expériences** (cf. § 8). G1 est éliminatoire. G2–G3 sont le cœur de la thèse. G4 conditionne le passage en V0.5/V1.

---

## 7. Roadmap complète

### 7.1 V0 (semaines 1–3) — trancher P1–P3

1. **S1 :** pipeline de données D0+D1, oracle G0, boucle d'entraînement compilée, G0.5, premier run C0–C2.
2. **S2 :** ablations vitales A1/A2/A4/A6, portes G1–G3, courbes par itération sur le probe set.
3. **S3 :** verdict documenté ; si vert : C3 (init naïve), G4, ablations restantes.

### 7.2 V0.5 (semaines 4–6) — durcir le résultat

* C4 (init quasi uniforme) : le régime le plus proche d'une « auto-organisation » réelle.
* L_support / sparsemax si le support est le point faible ; arêtes longues précalculées si A5 montre un champ récepteur limitant.
* Étude de robustesse : bruit sur les features, décimations différentes, seeds multiples (3 minimum par config).
* Analyse « rétro-flocking » (interprétabilité) : corréler les deltas appris Δz avec les sorties des trois règles fixes de B5 — la cellule apprise se décompose-t-elle en séparation / alignement / cohésion, ou a-t-elle découvert un comportement que la taxonomie de Reynolds ne couvre pas ? Résultat intéressant dans les deux cas.
* Mini rapport technique interne : figures par itération, tableau d'ablations, limites connues.

### 7.3 V1 (projection détaillée, ~3 mois à mi-temps)

**Périmètre V1 :** FlockRig-A + montée en résolution + couplage faible + premier benchmark comparatif. Toujours M4 Pro en machine de dev ; runs de nuit 8–12 h ; location GPU ponctuelle seulement si un mur de calcul est démontré (le code reste portable si le backend est isolé).

**O1 — FlockRig-A : repositionnement de joints, hiérarchie et cardinalité connues.**
Les joints deviennent des nœuds à part entière : état caché `h_j` (32–64), messages **bipartites** vertex→joint (agrégation sur les vertices candidats de chaque joint — l'inverse de la table de candidats V0) et joint→joint le long des arêtes de hiérarchie. La mise à jour de position doit être **équivariante** (contrairement au skinning, on prédit de la géométrie) : schéma type EGNN, `Δx_j = Σ_i φ(h_j, h_i, ‖x_i−x_j‖) · (x_i−x_j)/‖·‖`, translation-invariant et rotation-équivariant par construction. C'est, mot pour mot, une règle de pilotage à la Reynolds — une somme pondérée de directions vers les voisins — mais apprise : la séparation entre joints (anti-redondance) et la cohésion vers l'axe médian de la région de support doivent en émerger, ce qui fait des joints de FlockRig les véritables boids du système (§ 0.2). Entrées : joints perturbés (bruit de position, échelle) ; supervision : positions GT + erreur de déformation avec le skinning GT gelé. Mêmes principes qu'en V0 : deltas amortis, pool, T aléatoire, portes G1'–G3' analogues.

**O2 — Montée à 4–8 k vertices.**
Coarsening QEM précalculé à 3 niveaux (ex. 4096 → 1024 → 256), avec matrices de prolongation/restriction stockées. Schéma « en V » : itérations au niveau grossier, prolongation, raffinement fin — la même cellule à chaque niveau (poids partagés inter-niveaux : ablation à trancher). Le pool stocke des états multi-niveaux. Critère : la propriété A1 (la récurrence sert) doit **tenir à l'échelle**, sinon le résultat V0 était un artefact de la petite taille.

**O3 — Couplage faible Rig ↔ Skin.**
Alternance de phases : T_r étapes rig actif / skin gelé, puis l'inverse, avant toute dynamique conjointe. Le couplage simultané est un multiplicateur d'instabilité — ne l'aborder qu'avec deux sous-systèmes individuellement stables. Métrique clé : la boucle complète (joints perturbés → rig réparé → skin re-convergé) bat-elle le pipeline séquentiel figé ?

**O4 — Benchmark et positionnement.**
Dataset RigNet complet, splits standards. Métriques rig : J2J, J2B, B2B, precision/recall des joints, validité de l'arbre (trivialement satisfaite en V1 puisque la hiérarchie est donnée — la reporter quand même pour préparer V2). Métriques skin : L1, erreur de déformation, bleeding. Baselines : heuristiques géométriques + RigNet ; SkinTokens, UniRig, Puppeteer cités comme référence de l'état de l'art **sans prétention de parité en V1** — l'argument V1 n'est pas le SOTA mais le rapport qualité/paramètres et les propriétés uniques (réparation, convergence incrémentale).

**Critères de sortie V1 :**
1. A1 tient à 4–8 k vertices (la récurrence sert encore) ;
2. réparation démontrée sur le rig **et** le skin ;
3. J2J/B2B dans un facteur ~1,5 des chiffres RigNet publiés, à < 5 M de paramètres ;
4. **démo interactive** : convergence visible en temps réel dans Blender — l'artiste corrige un poids ou déplace un joint, le système re-converge localement. C'est l'argument différenciant produit (éditabilité) qu'aucun gros modèle one-shot n'offre, et le livrable qui rend V1 communicable (rapport technique + vidéo).

**Note de design préparée en V1 pour V2 — ajout/suppression de joints :** préférer un **sur-provisionnement** (J_max joints, gate d'existence σ(e_j) mis à jour par la dynamique, pénalité de parcimonie) à une croissance topologique littérale. Le problème d'existence devient différentiable, compatible avec les formes fixes exigées par MLX, et la « naissance/mort » de joints se lit dans la trajectoire de e_j. C'est la voie réaliste vers les étapes 6–8 du brief initial.

### 7.4 V2+ (horizon, non planifié)

Gates d'existence des joints (cf. note ci-dessus) → prédiction de la hiérarchie (pointeur vers le parent + contrainte d'arbre par MST sur les scores, validité mesurée) → topologie libre → haute résolution (50 k+) par cascade de niveaux → et, si l'attracteur est bien caractérisé, **distillation vers une passe unique** pour l'inférence temps réel — ironie assumée : la récurrence sert alors d'échafaudage d'apprentissage.

---

## 8. Protocole expérimental

* **Probe set fixe :** 8 meshes (2 procéduraux faciles, 2 procéduraux durs avec membres proches, 4 Mixamo), évalués à l'identique sur chaque run ; toutes les métriques tracées par itération.
* **Discipline :** un changement par run ; seeds contrôlés (3 par config pour toute conclusion) ; nomenclature `v0_<date>_<change>` ; config + métriques sérialisées avec chaque checkpoint.
* **Calendrier de décision :** J1 données + G0 ; J2 boucle + G0.5 (overfit 1 mesh) ; J3 premier vrai run C0–C2 + courbes ; J4 A1/A2/A4/A6 ; J5 verdict G1–G3 écrit noir sur blanc.
* **Règle anti-dérive :** tout correctif post-échec de porte a un budget explicite (2 jours pour G1). Le but du projet V0 n'est pas de faire marcher le système à tout prix, c'est de **savoir vite** si l'hypothèse tient.
