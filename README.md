# MaterialRanger

**AI-guided refractory design for BOF steelmaking** — composition → physics-informed descriptors → machine learning → thermal-shock & multi-objective ranking → top screened candidates.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

MaterialRanger is an end-to-end pipeline that designs, evaluates, ranks, and optimises refractory materials for high-temperature steelmaking service. It predicts thermal-shock resistance alongside the other properties that actually decide a lining — slag corrosion, creep, oxidation, strength, manufacturability, and cost/CO₂ — and identifies promising compositions and processing routes while minimising experimental cost through AI-guided discovery. Target processes span BOF converters, steel ladles, tundishes/SEN, and H₂-DRI / EAF routes.

> **Scope note.** Everything reported here is a set of **screened candidates**, not a deployment recommendation. The models fuse published physics laws with descriptor-based ML on a small curated dataset. External validation — CALPHAD equilibrium checks and thermal-shock coupon testing — is a required next step before any industrial claim. Limitations are stated explicitly in each section of the notebook.

---

## Pipeline at a glance

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

The full workflow lives in a single executed notebook: [`MaterialRanger_HKM_EXECUTED.ipynb`](MaterialRanger_HKM_EXECUTED.ipynb). All figures and results are rendered inline, so you can read the entire study on GitHub without running anything.

---

## Paper

[`manuscript/Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf`](manuscript/Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf)

**P. Sreenath, A. Hartmaier** (ICAMS, Ruhr-Universität Bochum): *From a data-driven materials ranking to microstructure: thermal-shock damage and creep of BOF refractory linings simulated with calibrated Kanapy RVEs, three-dimensional Abaqus models and machine-learning surrogates.* Preprint, October 2026, 20 pages.

The paper takes the MaterialRanger ranking in this repository a step further. The ranking shows that the choice between MgO–ZrO₂, magnesia and MgO–C depends on the weight given to thermal-shock damage (MgO–C ranks first only when that weight is at least 0.28). The paper then resolves this criterion with microstructure models:
- periodic 2-D Kanapy RVEs (320 microstructures, Latin-hypercube DoE) with sequentially linear analysis for thermal shock and Gauss-point Norton creep;
- five parameters calibrated on open brick data and validated on bricks not used in the calibration;
- an independent Abaqus/Standard solution of 60 calibrated RVEs;
- 3-D cohesive RVEs (13 645 interface elements) and a 3-D creep RVE of MgO–C;
- brick-scale XFEM and contour-integral models;
- descriptor- and image-based ML surrogates (Spearman 0.90–1.00 on the grade ranking).

---

## Headline findings

**Data-leakage-aware evaluation is the methodological core.** Every material contributes ~26 porosity/grade siblings, so a plain `KFold` split places near-duplicate rows (the same brick at P = 0.10 and P = 0.12) in train and test — inflating the score. The pipeline therefore separates two regimes explicitly: **`GroupKFold` / `GroupShuffleSplit`** (groups = material) gives the honest *extrapolation-to-unseen-chemistry* number, while the random split is reported only as the *interpolation / deployment* regime and labelled as such. `StratifiedGroupKFold` does the same for classification. This split discipline is what separates a believable figure of merit from an optimistic one.

**The pipeline reproduces industrial practice without being told it.** The screened BOF winner is a basic carbon-bonded brick and the caster winner is carbon-bonded alumina — matching what plants actually run. This falls out of a quantitative **strength paradox**: strong dense ceramics resist crack *initiation* but not *propagation*, which is why weaker, porous, non-wetting carbon-bonded bricks survive cycled service.

**Bonding character is the dominant learned descriptor.** Electronegativity spread / ionicity correlates most strongly with the continuum figure of merit (e.g. corr(δ_EN, log R′) ≈ −0.70), linking the quantum scale to the thermal-shock index.

**A graph representation beats a flat descriptor vector on extrapolation.** On unseen chemistries, a message-passing GNN on composition graphs (R² ≈ 0.92) edges out an equivalent flat MLP (R² ≈ 0.90); hand-derived gradients agree with finite differences to ~1e-8.

**Active learning cuts the experimental budget.** With a Matérn-5/2 Gaussian process, uncertainty sampling reduces prediction error by ~24 % at the same number of (expensive) tests, with selection constrained to distinct materials.

**Honest small-data baselines.** At n = 36 "experimental" points, a random forest beats every deep-network transfer-learning variant — reported rather than hidden, to avoid the "deep learning everywhere" trap in the small-data limit.

*(Committed outputs are from the `QUICK=True` fast-verification pass. Set `QUICK=False` to reproduce the full-size figures used in the write-up.)*

---

## What's inside the physics engine

| Phenomenon | Model | Key references |
|---|---|---|
| Thermal shock | Kingery surface-stress; Hasselman R, R′, R″″ | Kingery (1955); Hasselman (1969) |
| Porosity scaling | Ryshkewitch–Spriggs (σ, E); Maxwell–Eucken (k) | Ryshkewitch (1953); Spriggs (1961) |
| Slag corrosion | Duffy–Ingram optical basicity → Δμ dissolution/diffusion | Duffy & Ingram |
| Penetration | Darcy–Washburn with Kozeny–Carman permeability | — |
| Creep | Norton–Bailey; Frost–Ashby diffusional regime | Frost & Ashby (1982) |
| Oxidation | Wagner parabolic kinetics | — |

Every empirical constant is collected in one referenced calibration table (with provenance), and the corrosion/life coefficients are **fitted by non-negative least squares** against published campaign lives — not hand-tuned. Monte-Carlo uncertainty is propagated through the physics so each derived quantity is a distribution with a 90 % confidence interval; rankings that cannot survive their own error bars are flagged.

## What's inside the ML / AI stack

Group-aware regression and classification (with SMAPE, ROC/PR-AUC, balanced accuracy) · centred-log-ratio PCA on the Aitchison simplex + t-SNE/UMAP/Isomap · k-Means/GMM/DBSCAN/HDBSCAN/Ward clustering · a small from-scratch neural-network framework (classes for activations/losses/optimisers, dropout, weight decay, gradient clipping, early stopping) · message-passing GNN with numerically verified gradients · Gaussian-process active learning (uncertainty / EI / UCB) · transfer learning with honest baselines · a Binder-cumulant-validated Ising microstructure generator + CNN · autoencoder-vs-PCA, VAE with separated reconstruction/KL terms, and a GAN for compositional inverse design with a diversity audit · Pareto front + TOPSIS multi-objective ranking with process-specific weight vectors.

---

## Repository structure

```
MaterialRanger/
├── MaterialRanger_HKM_EXECUTED.ipynb   # the full pipeline (executed, figures inline)
├── manuscript/
│   └── Sreenath_Hartmaier_2026_BOF_refractories_microstructure.pdf   # the paper (preprint)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Running the notebook additionally generates `results.json`, `materialranger_dataset.csv`, a `figs/` directory, and a `dft_cache.csv` (the last two are git-ignored by default; the first two are tracked if present).

---

## Getting started

```bash
git clone https://github.com/<your-username>/MaterialRanger.git
cd MaterialRanger

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

jupyter notebook MaterialRanger_HKM_EXECUTED.ipynb
```

**Live DFT pulls (optional).** The pipeline runs fully offline from its curated dataset. To enable live ingestion from the Materials Project, set your API key as an environment variable — it is **never** hard-coded:

```bash
export MP_API_KEY="your-key-here"     # get one free at https://materialsproject.org/api
```

**Run modes.** `QUICK=True` (the default in the committed notebook) is a fast verification pass that finishes in minutes. Set `QUICK=False` for the full-size configuration (deeper GAN, longer Ising equilibration, more Monte-Carlo draws and active-learning rounds) used for the reported figures.

---

## Data sources

Curated handbook/datasheet values for ~66 industrial refractories and ceramic composites (each record carries a provenance flag), augmented by DFT properties from the **Materials Project**, **OQMD**, **AFLOW** (elastic / Debye / thermal conductivity), and **Alexandria**, with matminer-style composition featurization.

## Limitations & next steps

- **Small dataset.** 66 base grades is modest; the ~1,700 Monte-Carlo samples are porosity/grade *siblings* of those 66, not independent measurements — which is exactly why the group-aware split matters.
- **Synthetic regression target.** The thermal-shock regressor learns *descriptor → physics-equation → log R′*, so it benchmarks the ML methods and the descriptor set rather than discovering new physics.
- **No external experimental validation yet.** CALPHAD equilibrium validation and thermal-shock coupon testing are the acknowledged prerequisites before any deployment claim.
- **Screening-only coupling models.** The coating spallation and penetration estimates are first-order screens, not design stresses.

---

## License

Released under the [MIT License](LICENSE). If you'd prefer a different license (e.g. Apache-2.0, or all-rights-reserved for a thesis/employer context), swap the `LICENSE` file before making the repository public.

## Acknowledgements

Developed by Prajwal Sreenath at ICAMS (Interdisciplinary Centre for Advanced Materials Simulation), Ruhr-Universität Bochum. The project was inspired by a visit to Hüttenwerke Krupp Mannesmann (HKM), Duisburg. The microstructure work in the manuscript was done with Prof. Alexander Hartmaier (ICAMS). Abaqus simulations were run on the CIP computing facilities of Ruhr-Universität Bochum.
