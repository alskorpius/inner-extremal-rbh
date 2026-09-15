# inner-extremal-rbh

Code and data for the paper **"Inner-extremal regular black holes with NEC-satisfying anisotropic sources: existence, extent, and the absence of a classical selection mechanism"**.

A family of regular black holes whose inner horizon is degenerate — a triple root of f(r), so that κ₋ = 0 and the Cauchy-horizon instability is switched off at linear order — exists over a finite range of ℓ/M and satisfies the null energy condition throughout. No classical mechanism selects it: the degeneracy is destroyed by any perturbation according to κ₋ ∝ ε^{2/3}, the back-reaction pushes away from it rather than towards it, and every dynamical process examined here — mass inflation, semiclassical flux, viscosity, a first-order transition, accretion, rotation — either leaves the tuning untouched or destroys it.

## Install and run

```
git clone https://github.com/alskorpius/inner-extremal-rbh
cd inner-extremal-rbh
python -m venv .venv && .venv/Scripts/activate      # POSIX: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13.6, five dependencies, no compiled extensions. Every script is single-threaded and run from the repository root:

```
python src/verification_01/T16_base_audit.py        # the base profile, < 1 min
python src/verification_01/T1_kappa_law.py          # the sensitivity law, ~3 min
python figures/make_figures.py                      # all seven figures, ~15 s
```

The full ordered run, with timings, is in [`docs/reproducibility.md`](docs/reproducibility.md). A complete rebuild from an empty `data/` takes roughly four hours of single-core time; the two expensive steps are `T2_feedback.py` (~50 min) and the frequency-domain QNM checks (~65 min combined).

## Figure and table map

| Item in the paper | Script | Output |
|---|---|---|
| Fig. 1 — the triple-root family | `figures/make_figures.py 1` | `figures/fig1_family.pdf` |
| Fig. 2 — κ₋ against the detuning ε | `figures/make_figures.py 2` | `figures/fig2_kappa_eps.pdf` |
| Fig. 3 — Kantowski-Sachs phase portrait | `figures/make_figures.py 3` | `figures/fig3_ks_phase.pdf` |
| Fig. 4 — mass-inflation iterations | `figures/make_figures.py 4` | `figures/fig4_t2_iterations.pdf` |
| Fig. 5 — viscosity and the first-order transition | `figures/make_figures.py 5` | `figures/fig5_t29_t30.pdf` |
| Fig. 6 — halo against the QNM shift | `figures/make_figures.py 6` | `figures/fig6_halo_qnm.pdf` |
| Fig. 7 — residual κ₋ for ℓ² = λ²M² + ℓ₀² | `figures/make_figures.py 7` | `figures/fig7_residual_kappa.pdf` |
| Existence of the family, λ ∈ [~0.02, 1.066] | `src/verification_01/T15_lambda_bound.py`, `T15_qnm.py` | `data/verification_01/T15_lambda_bound.json`, `T15_qnm.json` |
| Base profile constants | `src/verification_01/T16_base_audit.py` | `data/verification_01/T16_base_audit.json` |
| Sensitivity law, exponent p = 0.660–0.674 | `src/verification_01/T1_kappa_law.py` | `data/verification_01/T1_kappa_law.json` |
| Back-reaction (mass inflation) | `src/verification_01/T2_feedback.py` | `data/verification_01/T2_feedback.json` |
| Semiclassics (Polyakov, Unruh) | `src/verification_01/T14_polyakov.py` | `data/verification_01/T14_polyakov.json` |
| Extremal end and M_ext | `src/verification_01/T21_extremal.py` | `data/verification_01/T21_extremal.json` |
| Kantowski-Sachs minisuperspace | `src/verification_02/T26_ks.py` | `data/verification_02/T26_ks.json` |
| Viscosity scan | `src/verification_03/T29_viscosity.py` (+ refinements) | `data/verification_03/T29_viscosity.json` |
| First-order transition | `src/verification_03/T30_phase_transition.py` | `data/verification_03/T30_phase_transition.json` |
| Holographic estimate | `src/holography/O1_holography.py` | `data/holography/O1_holography.json` |
| Halo and QNM shift | `src/verification_01/T18_halo.py`, `T18_trigger.py` | `data/verification_01/T18_halo.json` |
| QNM band over the family | `src/polar_qnm/qnm_band.py` | `data/polar_qnm/qnm_band.json` |
| Frequency-domain QNM check | `src/polar_qnm/axial_fd.py`, `axial_fd_fix.py` | `logs/polar_qnm/axial_fd*_stdout.txt`, `data/polar_qnm/axial_fd_fix.json` |
| Thin walls and the gravastar regime | `src/audit_bhp/audit_bhp.py` | `logs/audit_bhp/audit_log.txt` |
| Rotation (caveat) | `src/verification_03/T31_*.py`, `src/verification_02/T23_rotation.py` | `data/verification_03/T31_*.json` |
| Family constants to four digits | `src/verification_02/T25_constants.py` | `data/verification_02/T25_constants.json` |

The complete map, number by number, is [`docs/reproducibility.md`](docs/reproducibility.md).

## The base profile as data

`data/base_profile_m_of_r.csv` is the tabulated base member of the family — m(r), ρ(r), f(r), σ(r) on 4000 points in units G = c = M = 1, with ℓ/M = 0.27102, R₋ = 0.67744, R₊ = 1.98465. It is written by `src/verification_01/T16_base_audit.py` (as `data/verification_01/T16_base_profile_m_of_r.csv`, copied to the top of `data/` for convenience) and is meant to be usable on its own, without running anything in this repository.

## Layout

```
src/verification_01/   T1, T2, T14, T15, T16, T18, T21
src/verification_02/   T23, T25, T26
src/verification_03/   T29, T30, T31
src/stability/         the triple-root family (core module)
src/polar_qnm/         quasinormal modes, time and frequency domain
src/holography/        the holographic estimate
src/ori_triple_root/   the Ori two-flux model
src/audit_bhp/         thin shells, stabiliser, towers
src/accretion_tracking/, src/defect_field/, src/ned_lagrangian/,
src/inhomogeneous_collapse/, src/mechanism_ellM/, src/approach_map/,
src/ks_curvature_transition/, src/rotating_eikonal/, src/baseline/
data/                  results as JSON and CSV
figures/               figure script and its PDF/PNG output
logs/                  raw stdout logs of the runs
docs/reproducibility.md  number in the paper -> script -> result file
docs/limitations.md      scope and limits
```

## Limitations

The family is constructed rather than derived from a field theory: the profile is a specific σ(r) parameterisation with the triple root tuned numerically, and its constants are reliable to four digits. The semiclassical treatment is the 2D Polyakov approximation. Viscosity is treated in a linearised settling model, mass inflation through the Ori two-flux model, and thin walls through the standard junction conditions. Rotation is a caveat, not a result: the degenerate inner horizon was not constructed for a rotating regular metric, only bounded (recoverable only for spin a ≲ 0.1–0.15). One quantity is stochastic — the random search that locates λ_max, seeded at 7 and reproducible to about three digits across seeds.

The paper's central claim is negative, and it is a claim about the *classical* theory: no classical mechanism that selects the degenerate inner horizon was found among those examined, which is not a proof that none exists. Full discussion in [`docs/limitations.md`](docs/limitations.md).

## Related

The crossing of this inner horizon by an infalling body — the cost and the traversable window — is treated in the companion note: https://github.com/alskorpius/cauchy-horizon-crossing

## Citation

Oleh Popenkov, *Inner-extremal regular black holes with NEC-satisfying anisotropic sources: existence, extent, and the absence of a classical selection mechanism* (2026).
Code and data: https://github.com/alskorpius/inner-extremal-rbh — see [`CITATION.cff`](CITATION.cff).

ORCID: [0009-0008-9894-2982](https://orcid.org/0009-0008-9894-2982)

## License

Code in `src/` and `figures/`: MIT. Data in `data/`, raw logs in `logs/`, and the generated figure images: CC-BY-4.0. See [`LICENSE`](LICENSE).

## How this work was produced

The calculations in this repository were carried out with automated agents under the author's direction, with an independent cross-audit performed between two separate research projects. The author is responsible for the results.
