# Step 4 — Three-dimensional Abaqus models (MgO–C 20 % C)

The 2-D models (steps 2 and 3) found that MgO–C 20 % C is the most damage-tolerant grade. The 3-D models check that result
for this grade. They remove the two main simplifications of the 2-D model: plane stress, and a hot face held in only one
direction. They also use a different crack description (cohesive interfaces and XFEM instead of smeared saw-tooth damage)
and a different solver.

All models are solved with Abaqus/Standard 2025. The input decks are flat `.inp` files written in Python. Each deck is
checked by a static validator and an independent linear re-solution before it is submitted. The full description and all
figures are in the paper ([`../paper/`](../paper/), sections 5, 6.3 and 10).

## Models

| Model | What it is | Load |
|---|---|---|
| **A1** cohesive RVE, tension | Kanapy cube 0.5 mm, 24³ voxels (20.8 µm); 13 271 C3D8 + **13 645 cohesive elements** on all grain-boundary, graphite and binder interfaces | uniaxial tension to past the peak |
| **A2** cohesive RVE, thermal shock | same RVE; hot face held in both in-plane directions | uniform cooling by up to 400 K |
| **C1–C3** creep RVE | 32³ voxels (31 457 C3D8) with a grain-boundary layer; boundary stiffness and creep law recalibrated in 3-D | 5 MPa compression for 10 h at 1350, 1450 and 1550 °C |
| **B1** bending bar | 150 × 25 × 25 mm, XFEM, homogeneous material with the RVE stiffness | 3-point bending, span 125 mm |
| **B2** hot-face slice | 50 × 30 × 10 mm, XFEM | hot-face shock with an erfc temperature profile |
| **D1** notched beam (SENB) | a/W = 0.3, contour integrals | 1 kN |

## Results

- **Tension (A1).** Peak stress 20.0 MPa at 0.168 % strain. Graphite interfaces crack first, then grain boundaries;
  the binder breaks last. The cracks link into one rough band perpendicular to the load.
- **Thermal shock (A2).** The RVE is elastic up to 100 K. At 400 K, 20 % of the interfaces are broken and form a
  **distributed microcrack network**. The RVE still carries **12 % of the elastic hot-face stress**, against 3 % in the 2-D
  model: in 3-D the cracks have to link over a surface, not along a line.
- **Creep (C1–C3).** Creep strain after 10 h: 0.16 % (1350 °C), 0.32 % (1450 °C), 0.58 % (1550 °C). That is **3.5 times
  less** than the 2-D estimate at 1550 °C, because the connected graphite skeleton does not creep and takes over the load.
  Graphite stress rises from 7 to 21 MPa while the MgO stress falls from 7 to 2.5 MPa. As a result the creep rate
  decreases with time (ε ∝ t^0.6) and the apparent activation energy is about 160 kJ/mol, below the 281 kJ/mol of the
  input law, even though the model has no primary-creep term.
- **Brick scale (B1, B2, D1).** Hot-face crack onset at 72 K (Hasselman estimate 66 K). Modulus of rupture 15.5 MPa
  (measured 13–16 MPa). Stress intensity factors within 1 % of ASTM E399.

## Verification

| Check | Reference | Abaqus |
|---|---|---|
| Stiffness of the full RVE (A1) | 13.3 GPa (independent solution) | 13.3 GPa |
| Hot-face elastic stress per K (A2) | 0.159 / 0.241 MPa/K | within 3 % |
| Interface peak tractions (A1) | 38.4 / 19.2 / 12.0 MPa (input) | 35.9 / 18.5 / 12.0 MPa |
| Crack onset at the hot face (B2) | 66 K (Hasselman) | 72 K |
| K_I of the notched beam (D1) | 48.7 MPa√mm (ASTM E399) | 48.3 MPa√mm |

## Files

The Abaqus input decks and the post-processing scripts are not in this repository yet. The result databases (`.odb`)
are too large for GitHub and stay on the compute machines.
