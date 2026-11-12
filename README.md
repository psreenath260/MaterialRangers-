# MaterialRanger

**From a data-driven materials ranking to the microstructure — refractory linings for BOF steelmaking.**
Machine learning on DFT data ranks the candidate materials. Calibrated 2-D and 3-D microstructure simulations then explain
the criterion that decides the ranking, and ML surrogates make those simulations fast enough to use for screening.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Abaqus](https://img.shields.io/badge/Abaqus-2025-lightgrey)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## The project in four steps

```
1  Ranking              66 refractories · ML trained on DFT data · physics laws · TOPSIS
                        → the choice between MgO–ZrO₂, magnesia and MgO–C depends on how
                          thermal-shock damage is weighted
        ↓
2  2-D RVEs + ML        320 periodic microstructures · thermal-shock cracking and creep
                        ANN (descriptors) + periodic U-Net CNNs (damage map, creep-strain map)
        ↓
3  Calibrated 2-D model 5 parameters fitted on open brick data, validated on bricks not used
                        for fitting, cross-checked against Abaqus · dual-head CNN surrogate
        ↓
4  3-D Abaqus models    cohesive RVE with 13 645 interfaces · 3-D creep RVE ·
                        brick-scale XFEM and fracture mechanics
        ↓
   Paper                Sreenath & Hartmaier, preprint 2026
```

| Folder | Content |
|---|---|
| [`1_ranking/`](1_ranking/) | [`MaterialRanger_HKM_EXECUTED.ipynb`](1_ranking/MaterialRanger_HKM_EXECUTED.ipynb): data, physics engine, ML, multi-objective ranking |
| [`2_rve_2d_crack_creep_ml/`](2_rve_2d_crack_creep_ml/) | [`MaterialRanger_RVE_Crack_Creep_ML.ipynb`](2_rve_2d_crack_creep_ml/MaterialRanger_RVE_Crack_Creep_ML.ipynb): 2-D RVEs, crack and creep solver, Abaqus decks, ANN and CNN surrogates |
| [`3_rve_2d_calibrated/`](3_rve_2d_calibrated/) | [`MaterialRanger_RVE_Calibrated_Kanapy.ipynb`](3_rve_2d_calibrated/MaterialRanger_RVE_Calibrated_Kanapy.ipynb): Kanapy RVEs, calibration and validation against literature, Abaqus comparison, dual-head CNN |
| [`4_rve_3d/`](4_rve_3d/) | 3-D Abaqus models and results ([README](4_rve_3d/README.md)) |
| [`paper/`](paper/) | The paper (PDF) |

All notebooks are executed, with figures inline, so they can be read on GitHub without running anything.

---

## Key results

- **The ranking hinges on thermal shock.** With default weights MgO–ZrO₂ ranks first. MgO–C, the industrial standard,
  ranks first only when the weight of thermal-shock resistance reaches 0.28.
- **The microstructure explains why MgO–C wins.** MgO–C starts cracking first (median 26 K) but keeps 0.87–0.92 of its
  stiffness after a 100 K shock, against 0.55–0.56 for MgO–ZrO₂ and magnesia. It also creeps 5–19 times less. The compliant
  graphite network arrests the cracks it starts.
- **Calibrated and validated.** Five parameters are fitted on one MgO–C brick and one magnesia creep series. Without
  refitting, the model predicts the stiffness and strength ratio of a second MgO–C brick within −16 % and −6 %, and magnesia
  creep at 1500 °C within 0.22 dex (σ ≥ 1.5 MPa).
- **Independent solver check.** Abaqus/Standard on 60 calibrated RVEs gives identical moduli, creep strains within 0.5 % and
  post-shock stiffness within 0.016 up to 100 K.
- **3-D confirms 2-D.** The 3-D cohesive RVE still carries 12 % of the elastic hot-face stress after 400 K (2-D: 3 %), and
  the connected graphite skeleton lowers creep 3.5-fold. Brick-scale XFEM gives a crack onset of 72 K and a modulus of
  rupture of 15.5 MPa (measured 13–16 MPa).
- **Fast surrogates.** The dual-head CNN predicts creep-strain maps with R² 0.98–0.99 and crack maps with IoU 0.45–0.47
  (0.20 for a prior-only baseline). The surrogates reproduce the simulated grade ranking (Spearman 0.90–1.00) in milliseconds.

---

## Step 1 — Data-driven ranking

[`1_ranking/MaterialRanger_HKM_EXECUTED.ipynb`](1_ranking/MaterialRanger_HKM_EXECUTED.ipynb)

An end-to-end pipeline that evaluates and ranks refractory materials for high-temperature steelmaking. It predicts
thermal-shock resistance alongside the other properties that decide a lining (slag corrosion, creep, oxidation, strength,
manufacturability, cost/CO₂) and identifies promising compositions and processing routes. Target processes are BOF
converters, steel ladles, tundishes/SEN and H₂-DRI / EAF routes.

```
Literature + handbooks ─┐
DFT databases (MP · OQMD · AFLOW · Alexandria) ─┤   matminer-style featurizers
                        ─┘
        ↓
  ~66 industrial refractories & ceramic composites
  (+ coatings, + porosity/grade Monte-Carlo sampling)
        ↓
  Physics engine  (Kingery · Hasselman · Ryshkewitch–Spriggs ·
                   Duffy–Ingram · Norton–Bailey / Frost–Ashby · Wagner · Darcy–Washburn)
        ↓
  ML / AI  (group-aware regression & classification · CLR-PCA · GNN ·
            active learning · generative inverse design)
        ↓
  Multi-objective optimisation  (Pareto front + TOPSIS, process-specific weights)
        ↓
  Top screened candidates, with Monte-Carlo confidence intervals
```

**Findings**

- **Data-leakage-aware evaluation.** Every material contributes ~26 porosity/grade siblings, so a plain `KFold` split
  puts near-duplicate rows in train and test and inflates the score. `GroupKFold` / `GroupShuffleSplit` (groups = material)
  gives the honest *extrapolation-to-unseen-chemistry* number; the random split is reported only as the
  *interpolation* regime. `StratifiedGroupKFold` does the same for classification.
- **The pipeline reproduces industrial practice without being told it.** The screened BOF winner is a basic carbon-bonded
  brick and the caster winner is carbon-bonded alumina. This follows from a quantitative **strength paradox**: strong dense
  ceramics resist crack *initiation* but not *propagation*, which is why weaker, porous, non-wetting carbon-bonded bricks
  survive cycled service. Steps 2–4 test this at the microstructure scale.
- **Bonding character is the dominant learned descriptor.** Electronegativity spread / ionicity correlates most strongly
  with the thermal-shock figure of merit (corr(δ_EN, log R′) ≈ −0.70).
- **A graph representation beats a flat descriptor vector on extrapolation.** On unseen chemistries a message-passing GNN
  on composition graphs (R² ≈ 0.92) edges out an equivalent flat MLP (R² ≈ 0.90).
- **Active learning cuts the experimental budget.** A Matérn-5/2 Gaussian process with uncertainty sampling reduces the
  prediction error by ~24 % at the same number of tests.
- **Honest small-data baselines.** At n = 36 "experimental" points a random forest beats every deep transfer-learning variant.

| Phenomenon | Model | Key references |
|---|---|---|
| Thermal shock | Kingery surface-stress; Hasselman R, R′, R″″ | Kingery (1955); Hasselman (1969) |
| Porosity scaling | Ryshkewitch–Spriggs (σ, E); Maxwell–Eucken (k) | Ryshkewitch (1953); Spriggs (1961) |
| Slag corrosion | Duffy–Ingram optical basicity → Δμ dissolution/diffusion | Duffy & Ingram |
| Penetration | Darcy–Washburn with Kozeny–Carman permeability | — |
| Creep | Norton–Bailey; Frost–Ashby diffusional regime | Frost & Ashby (1982) |
| Oxidation | Wagner parabolic kinetics | — |

Every empirical constant is collected in one referenced calibration table, the corrosion/life coefficients are fitted by
non-negative least squares against published campaign lives, and Monte-Carlo uncertainty is propagated through the physics
(90 % confidence intervals). The ML stack covers group-aware regression and classification, CLR-PCA / t-SNE / UMAP,
clustering, a from-scratch neural-network framework, a message-passing GNN, Gaussian-process active learning, transfer
learning, an Ising microstructure generator + CNN, autoencoder / VAE / GAN for inverse design, and Pareto + TOPSIS ranking.

*Committed outputs are from the `QUICK=True` pass. Set `QUICK=False` for the full-size configuration.*

## Step 2 — 2-D RVEs: crack, creep and ML surrogates

[`2_rve_2d_crack_creep_ml/MaterialRanger_RVE_Crack_Creep_ML.ipynb`](2_rve_2d_crack_creep_ml/MaterialRanger_RVE_Crack_Creep_ML.ipynb)

The ranked candidates (MgO–C, MgO–ZrO₂, magnesia, Al₂O₃–C) differ in microstructure (graphite flakes, pores, grain
boundaries, weak interfaces), which composition-based models cannot see. This notebook:

1. generates **320 periodic 2-D RVEs** (one finite element per pixel) by a Latin-hypercube design over phase fractions,
   flake geometry and texture, porosity, grain size and boundary/interface strength;
2. simulates **thermal-shock cracking** (constrained cooling of the hot face) and **Norton creep** (10 h at 1450–1650 °C),
   with a verified Python reference solver and matching **Abaqus input decks** (CPS4, concrete damaged plasticity, *VISCO);
3. trains an **ANN** (descriptors → crack-onset temperature, residual stiffness, creep rate);
4. trains **periodic U-Net CNNs** that predict the **damage map** and the **creep-strain map** directly from the
   microstructure image, then one **shared-encoder dual-head CNN** for both;
5. tests the surrogates on **Kanapy** microstructures of the ranked grades, i.e. on a different generator than the training data.

The physics in the data reproduces the strength paradox: MgO–C cracks early at weak flake interfaces but keeps far more
stiffness than the dense oxides, whose cracks run through the whole cell.

## Step 3 — Calibrated 2-D model

[`3_rve_2d_calibrated/MaterialRanger_RVE_Calibrated_Kanapy.ipynb`](3_rve_2d_calibrated/MaterialRanger_RVE_Calibrated_Kanapy.ipynb)

All microstructures are generated with **Kanapy** ellipsoid packing. Thermal shock is solved by sequentially linear
analysis with an energy-consistent saw-tooth law, and creep by a Norton law integrated at the Gauss points.

- **Calibration** on open literature data: Young's modulus and modulus of rupture of MgO–C with 15 wt% graphite, and
  compressive creep of a fired magnesia brick at 1350 and 1425 °C (boundary stiffness s_b = 0.0104, strength scale
  k_f = 0.80, creep n = 1.75, Q = 274 kJ/mol).
- **Validation** on data not used for calibration: MgO–C with 7 wt% graphite, the magnesia brick's modulus, and creep at 1500 °C.
- **Abaqus cross-check** of the solver on identical meshes.
- **Dual-head U-Net** trained on 320 calibrated RVEs, evaluated on held-out RVEs and on Kanapy realizations of five ranked grades.

Verification (exact solutions, scaling laws, convergence), calibration and validation are kept strictly separate, and
each notebook ends with an automatic summary of its checks.

## Step 4 — 3-D Abaqus models

[`4_rve_3d/`](4_rve_3d/README.md)

Independent check of the 2-D result for MgO–C 20 % C without the plane-stress assumption: a cohesive RVE with zero-thickness
cohesive elements on all 13 645 grain-boundary, graphite and binder interfaces (tension and thermal shock), a 3-D creep RVE,
and brick-scale XFEM bending, hot-face shock and notched-beam (contour integral) models. Details and results are in the
[step-4 README](4_rve_3d/README.md).

## Paper

[`paper/Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf`](paper/Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf)

**P. Sreenath, A. Hartmaier** (ICAMS, Ruhr-Universität Bochum): *From a data-driven materials ranking to microstructure:
thermal-shock damage and creep of BOF refractory linings simulated with calibrated Kanapy RVEs, three-dimensional Abaqus
models and machine-learning surrogates.* Preprint, October 2026, 20 pages.

---

## Repository structure

```
MaterialRanger/
├── 1_ranking/
│   └── MaterialRanger_HKM_EXECUTED.ipynb
├── 2_rve_2d_crack_creep_ml/
│   └── MaterialRanger_RVE_Crack_Creep_ML.ipynb
├── 3_rve_2d_calibrated/
│   └── MaterialRanger_RVE_Calibrated_Kanapy.ipynb
├── 4_rve_3d/
│   └── README.md
├── paper/
│   └── Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/psreenath260/MaterialRangers-.git
cd MaterialRangers-

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

jupyter notebook
```

- **Step 1** runs offline from its curated dataset. For live Materials Project pulls, set `MP_API_KEY` as an environment
  variable (it is never hard-coded).
- **Steps 2 and 3** write their solver modules to `mr_rve_src/` with `%%writefile` and generate all RVEs and results in a
  working folder (`MR_RVE_work/`). A single RVE and load set takes about 1–2 CPU minutes. The Abaqus comparison sections
  need the `*_results.npz` files extracted from the Abaqus runs, which are not part of this repository.

## Limitations

- **Screening, not deployment.** The ranking gives screened candidates. CALPHAD equilibrium checks and thermal-shock coupon
  tests are needed before any industrial claim.
- **Ranking data.** 66 base grades is modest; the ~1,700 Monte-Carlo samples are porosity/grade siblings of those 66,
  which is why the group-aware split matters. The thermal-shock regressor learns *descriptor → physics law → log R′*.
- **Scale.** The 1 mm RVEs resolve the fine matrix (grains, flakes, pores), not the millimetre-sized aggregates of real
  bricks; the brick-scale models therefore use homogenised RVE properties and brick-level fracture parameters.
- **Validation data.** The calibration and validation use published room-temperature stiffness/strength and creep data.
  No quench experiments were available, so thermal-shock results are compared with literature values, not measured directly.
- **3-D models.** One microstructure realization per 3-D model.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Developed by Prajwal Sreenath at ICAMS (Interdisciplinary Centre for Advanced Materials Simulation), Ruhr-Universität
Bochum, together with Prof. Alexander Hartmaier (ICAMS). The project was inspired by a visit to Hüttenwerke Krupp
Mannesmann (HKM), Duisburg. Abaqus simulations were run on the CIP computing facilities of Ruhr-Universität Bochum.
