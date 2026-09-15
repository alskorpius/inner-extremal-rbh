# Reproducibility map

Every number in the paper maps to one command and one result file. All commands are run from the repository root with the interpreter of the virtual environment created from `requirements.txt`. Units are G = c = 1 and M = 1 unless stated otherwise.

## Environment

Python 3.13.6; NumPy 2.3.5; SciPy 1.17.0; SymPy 1.14.0; mpmath 1.3.0; Matplotlib 3.10.8. Every script is single-threaded. Timings below are for a 2024 desktop and are indicative.

All results are deterministic except the random search in `T15_lambda_bound.py`; see [Determinism](#determinism).

## Number → script → result file

| Number or statement in the paper | Command | Result file | Section |
|---|---|---|---|
| Triple-root family exists for λ ∈ [~0.02, 1.066]; maximum λ_max = 1.066 at u₁ = 2.05, u₂ = 2.99, u₃ = 3.99, s_end = 32.3, σ_min = 0.464; the boundary is extremality R₋ → R₊ | `python src/verification_01/T15_lambda_bound.py` then `python src/verification_01/T15_qnm.py` | `data/verification_01/T15_lambda_bound.json` (`best`, `lam_max`, `breaks`), `data/verification_01/T15_qnm.json` (`opt30`, `base_opt`, `points`), `logs/verification_01/T15_log.txt` | Existence |
| Identity λ = (R₋/M)·√(3ρ(R₋)/ρ_c), checked on every profile found | `python src/verification_01/T15_lambda_bound.py` | `data/verification_01/T15_lambda_bound.json`, field `rows[*].identity_lam` against `lam` | Existence |
| Base profile: R₋ = 0.6774, κ₋ = −5.4·10⁻⁸, f″(R₋) = 2.5·10⁻³, σ(R₋) = 2.0011, 8πR₋²ρ = 1.000000, R₊ = 1.9846, κ₊ = 0.2211, T_H/T_Schw = 0.884, halo 1 − m(R₊)/M = 0.77 %, ρ′ ≤ 0 everywhere, min(ρ + p_⊥) ≥ 0 | `python src/verification_01/T16_base_audit.py` | `data/verification_01/T16_base_audit.json`, `data/verification_01/T16_base_profile_m_of_r.csv`, `logs/verification_01/T16_log.txt` | Existence / Setup |
| Tabulated base profile m(r), ρ(r), f(r), σ(r) on 4000 points, for external use | `python src/verification_01/T16_base_audit.py` | `data/verification_01/T16_base_profile_m_of_r.csv` (copy at `data/base_profile_m_of_r.csv`) | Setup |
| Profile construction (σ slope: sigmoids plus a Gaussian dip; triple root tuned through s₁); base parameters u₁ = 1, u₂ = 4, u₃ = 8, w₁ = 0.25, w₂ = 0.35, w₃ = 0.1, s_end = 24, σ_min = 1 | `python src/stability/triple_root_monotone.py` | `data/stability/` | Setup |
| Sensitivity law κ₋ ∝ ε^{2/3}: exponent p = 0.660–0.674 over 4 profiles, 5 perturbation types, both signs of ε; coefficient (3/2)\|a\|^{1/3}\|g\|^{2/3} to 5–10 % for \|ε\| ≤ 10⁻⁴; a = f‴(R₋)/6 = −2.16 (base), −19.6 (λ = 0.13), −4.42 (λ = 0.21), −1.59 (λ = 0.30); one simple root at R₋ for both signs of ε; mass added only outside R₋ leaves κ₋ unchanged | `python src/verification_01/T1_kappa_law.py` | `data/verification_01/T1_kappa_law.json` (`results[*].kinds[a,b,g,v,v2]` → `rows`, `fits`, `C_pred`), `logs/verification_01/T1_log.txt` | Sensitivity |
| Required tuning accuracy \|ε\| ≤ 3·10⁻³² (10 M☉) and ~10⁻²⁴ (10⁶ M☉) for N_e = \|κ₋\|τ ≤ 10 over τ = 10¹⁰ yr; a 0.1 % error gives κ₋ = −0.02 on the base profile and −0.04 on an early profile with R₋ ≈ 0.3 M | `python src/verification_01/T1_kappa_law.py` | `data/verification_01/T1_kappa_law.json`, blocks `precision[10.0]`, `precision[1000000.0]` | Sensitivity |
| Residual κ₋ for ℓ² = λ²M² + ℓ₀²: \|κ₋\| ≈ 5.3 (ℓ₀/M)^{4/3} analytically; numerically slope 1.32 for ℓ₀/M ≤ 10⁻² with coefficient 4.8 | `python figures/make_figures.py 7` | `figures/figures_log.txt`, `figures/fig7_residual_kappa.pdf` | Absence (ratchet) |
| Mass inflation as back-reaction is positive for both signs of ε: κ₋ goes −10⁻³ → −1.47 (heavier) and −1.04 (lighter) over N_e = 12; dκ₋/dN_e on the first step −6·10⁻³…−0.6; smoothing in ℓ does not change the sign; R₋ moves inward, dR₋/dN_e = −0.002…−0.05 M | `python src/verification_01/T2_feedback.py` | `data/verification_01/T2_feedback.json` (`results[*].hist`, `summary`), `logs/verification_01/T2_log.txt` | Absence |
| Classical robustness of the exact triple root: mass excess behind the shell ≤ 1.3·10⁻⁴ with no growth (Price tail to δm/m₀ = 0.1, p = 11, 12); RN and Hayward controls give an exponential at rate \|κ₀\| − (p + 1)/v | `python src/ori_triple_root/ori_model.py` | `data/ori_triple_root/`, `logs/ori_triple_root/run_log.txt` | Existence / Absence |
| Semiclassics (2D Polyakov, Unruh): T_uu(R₋) = (κ₊² − κ₋²)/(48π) = +3.24·10⁻⁴, T_vv(R₋) = 0; luminosity L = κ₊²/(48π) → 1.4·10⁻²⁹ W (10 M☉) against the required 2.3·10³¹ W, ratio 6·10⁻⁶¹; ρ_ff ∝ (r − R₋)⁻⁶ at the triple root and ⁻² at a simple root; the layer where ρ_ff = ρ_c is 2.7·10⁻¹⁰ m (10 M☉), 1.2·10⁻¹⁰ m (3 M☉), 1.5·10⁻⁶ m (Sgr A*); the coefficient vanishes at \|κ₋\| = κ₊, i.e. ε ≈ 2.8·10⁻² | `python src/verification_01/T14_polyakov.py` | `data/verification_01/T14_polyakov.json`, `logs/verification_01/T14_log.txt` | Absence |
| No protecting quantity: the conditions m(R₋) = R₋/2, m′ = 1/2, m″ = 0 are algebraic; RN and NED give a charge relation; a string cloud has no degenerate root; a monopole gives a theory constant. Illustration M/M\* = 0.97 / 1.03 → κ₋ = −0.10 / −0.18 | `python src/ned_lagrangian/ned_reconstruct.py` | `data/ned_lagrangian/`, `logs/ned_lagrangian/run_log.txt` | Absence |
| Extremal end: λ = 1.000 / 1.018 / 1.035 / 1.066 → R₋/M = 1.408 / 1.439 / 1.473 / 1.558, R₊/M = 1.836 / 1.810 / 1.773 / 1.619, κ₊ = 5.7·10⁻² / 4.0·10⁻² / 2.3·10⁻² / 2.1·10⁻⁴, halo 8.2 / 9.5 / 11.4 / 19.1 %; κ₊ ∝ (R₊ − R₋)^{2.88}; ±δM gives two horizons (κ₋ = −2.5·10⁻³…−7.7·10⁻²) or a horizonless object; M_ext(ℓ₀ = 10.7 km) = 6.80 M☉ | `python src/verification_01/T21_extremal.py` | `data/verification_01/T21_extremal.json`, `logs/verification_01/T21_log.txt` | Absence |
| Kantowski-Sachs minisuperspace: identities (C), (X), (Θ) hold on the base profile to 10⁻¹⁵ / 10⁻¹²; τ(R₋ + x) ∝ x^{−1/2} with fitted slope −0.479; detuning ε = ±10⁻⁴…±10⁻¹ crosses R₋ within τ = 4–17 M, with τ ∝ \|ε\|^{−1/6}; Nariai is a saddle with eigenvalue +1/b\* for any V; in k-essence only C = 0 (Λ) stays regular | `python src/verification_02/T26_ks.py` | `data/verification_02/T26_ks.json` (`part1`, `part2`, `tau`, `scalar`, `kessence`), `logs/verification_02/T26_log.txt` | Absence / App. A |
| Viscosity: a sink requires ζ < 0 (linearisation s² − 8πζs + 8πρw = 0); for ζ > 0 with ζ = ζ₀ρⁿ, ζ₀ ∈ [10⁻³, 3], n ∈ {0, ½, 1}, in Eckart and Israel-Stewart (τ_IS = 0.1, 1): 0 degenerate exits out of ~300 (variant I) and 0 out of 36 (variant II); anti-dissipative window n = 0, ζ₀ ∈ [−0.3, −0.07] settles at b\* = 1.44–1.72 M with NEC min = +0.55…+0.76 and a static negative-mass interior behind the settling | `python src/verification_03/T29_viscosity.py`, then `T29_refine.py`, `T29_refine2.py abd`, `T29_refine2.py c`, `T29_interior.py` | `data/verification_03/T29_viscosity.json` (`scan_I`, `t1`, `IS`, `II`), `T29_refine2_abd.json`, `T29_refine2_c.json`, `T29_interior.json`, `logs/verification_03/` | Absence |
| First-order transition: the outcome is set by m₀ = m_c − (4π/3)ρ_vac b_c³ (a light fluid gives a singularity for any ΔE and τ_rel; a heavy one gives a simple horizon with negative mass); ΔE\* = −0.453 ε_c at ℓ = 0.27, ρ₀ = 10⁻² ε_c with κ = −8·10⁻⁵; shifting ΔE by 0.1 % gives κ = −0.59 or a singularity; the scenario EOS gives κ = −1.5·10⁻⁴ only at b_c = u₃R₁ = 1.864, and ±0.5 % raises \|κ\| to ≥ 0.08 | `python src/verification_03/T30_phase_transition.py` | `data/verification_03/T30_phase_transition.json` (`rows`, `dE_star`, `scen`, `scen_star`), `logs/verification_03/T30_log.txt` | Absence |
| Holographic estimate (O1) on the base profile | `python src/holography/O1_holography.py` | `data/holography/O1_holography.json`, `logs/holography/O1_log.txt` | Absence |
| Thin walls: no static equilibrium for ℓ < 2M (f_in > 0 and f_out > 0 are incompatible); in the gravastar regime the equilibria R\* = 2.99–5.81 M are unstable (V″ = −0.047…−0.567) and stabilise only for Γ > Γ_c = 6.65 / 0.87 / 0.36 at R\* = 2.79 / 3.71 / 5.38 M | `python src/audit_bhp/audit_bhp.py` | `data/audit_bhp/`, `logs/audit_bhp/audit_log.txt` | App. B |
| Halo and QNM: the shift is ≈ −0.2·(m_halo/M); hard cutoffs R_c = 1.2 / 1.4 / 1.6 M give halo 0 and shift 0 (+7·10⁻¹⁰); 46 smooth configurations with halo 0.77 % give < 10⁻⁵; nearly compact profiles give 0.043 % → −9.6·10⁻⁵, 0.012 % → −2.7·10⁻⁵, 0.003 % → −7.5·10⁻⁶ | `python src/verification_01/T18_halo.py`, `python src/verification_01/T18_trigger.py` | `data/verification_01/T18_halo.json` (`A`, `B`, `C`, `D`), `T18_trigger.json`, `logs/verification_01/T18_log.txt` | Observables |
| QNM band over 8 profiles (axial and polar, l = 2, 3, time domain plus matrix pencil; Leaver control to 10⁻⁵ / 10⁻⁴): frequency −0.95…+0.51 %, damping −4.2…+0.14 % (l = 2, axial) | `python src/polar_qnm/qnm_band.py` | `data/polar_qnm/qnm_band.json`, `logs/polar_qnm/qnm_band_log.txt` | Observables |
| Independent frequency-domain check of the band (Wronskian, complex contour; Leaver to 10⁻⁶): discrepancy with the time domain ≤ 7·10⁻⁵ in frequency and ≤ 4.2·10⁻⁴ in damping, with matching signs | `python src/polar_qnm/axial_fd.py`, then `python src/polar_qnm/axial_fd_fix.py u3=20 s_end=8 "base (scenario)" u2=3 u3=12 s_end=48 sigma_min=0.5 w2=0.6` | `logs/polar_qnm/axial_fd_stdout.txt`, `logs/polar_qnm/axial_fd_fix_stdout.txt`, `data/polar_qnm/axial_fd_fix.json` | Observables / App. D |
| Cost of tracking ℓ ∝ M under accretion: m_M < 0 for R < 1.49 M; ingoing negative-energy flux up to 0.20 Ṁ; relative density (m_M/m_R)Ṁ ≥ −0.28 Ṁ, i.e. 3·10⁻²¹ ρ_c at the Eddington rate for 10 M☉; without tracking, M → 1.01 M gives κ₋ = −0.079 | `python src/accretion_tracking/accretion_tracking.py`, `python src/accretion_tracking/trapped_mass_monotonic.py` | `data/accretion_tracking/`, `logs/accretion_tracking/run_log.txt` | Absence (ratchet) / Setup |
| Rotation (caveat only): the Franzin metric is a conformally regular class with m(0) ≠ 0; in the GG class with m ∝ r³, κ₋ ∝ a^{4/3} (−0.093 at a = 0.1) and the base profile is horizonless for a ≳ 0.58; degeneracy is recoverable only for a ≲ 0.1–0.15; the ring has R = 12/ℓ² on the equator and 0 off it (a directional singularity); frame dragging Ω₋/Ω₊ = 8.45 / 2.29 at a = 0.6 / 0.9 with e = 0.5, and T^r_t ≡ 0 | `python src/verification_03/T31_rotating_core.py`, `T31_retune.py`, `T31_amax.py`, `T31_ring_numeric.py`, `python src/verification_02/T23_rotation.py` | `data/verification_03/T31_*.json`, `data/verification_02/T23_rotation.json`, `logs/verification_03/`, `logs/verification_02/` | App. C |
| Family constants are reliable to 4 digits (λ = 0.271016, R₋ = 0.677630, R₊ = 1.984648, κ₊ = 0.221082); no closed forms (the control on 10 random numbers is not passed) | `python src/verification_02/T25_constants.py` | `data/verification_02/T25_constants.json`, `logs/verification_02/T25_log.txt` | App. D |
| Field defect: a smooth field has a degenerate horizon only at the scale of the theory constants (x_h² = 1/(1 − 1/(2α))); a triple root is impossible for the Mexican hat; the subcritical σ family runs 0 → 2 from below; supercritical horizons give κ_h = −0.06…−0.24 per δ_c | `python src/defect_field/defect_monopole.py`, `python src/defect_field/near_critical.py` | `data/defect_field/`, `logs/defect_field/defect_monopole_log.txt` | Absence (mechanisms for ℓ ∝ M) |
| NED realisation: L(F) is reconstructed, stability holds outside the horizon (det S > 0, l = 2–4), inside κ_MS < 0 on [0.90, 1.63] and in the core, with amplification ln‖T‖ = 2.4ℓ (Hadamard ill-posedness) | `python src/ned_lagrangian/ned_reconstruct.py`, `python src/polar_qnm/interior_layer.py` | `data/ned_lagrangian/`, `data/polar_qnm/interior_layer.json` | Discussion |
| Inhomogeneous collapse: all 160 shells bounce at K = (0.98 ± 0.01)·24/ℓ⁴ for ε = ±0.01…±0.3; caustics form after the bounce; the weak solution is not unique | `python src/inhomogeneous_collapse/ltb_bounce.py`, `postprocess.py`, `variant_A.py`, `weak_solution.py`, `weak_solution_shells.py` | `data/inhomogeneous_collapse/`, `logs/inhomogeneous_collapse/` | Consequence / Discussion |
| Figures 1–7 | `python figures/make_figures.py` (or with figure numbers) | `figures/fig1_family.pdf` … `figures/fig7_residual_kappa.pdf` (plus `.png`), `figures/figures_log.txt` | — |

Numbers that the working notes attribute to the companion paper (e-folding 18–41 μs for n = 6 profiles; echo exclusion; the ET/CE stack of 25–400 events) are cited in the paper from the companion preprint and are not reproduced here.

## Full run

```
# existence and the base profile
python src/verification_01/T15_lambda_bound.py      # ~15 min (39-point scan + 300 random + Nelder-Mead; seed = 7)
python src/verification_01/T15_qnm.py               # ~10 min
python src/verification_01/T16_base_audit.py        # < 1 min
# sensitivity and the absence of a mechanism
python src/verification_01/T1_kappa_law.py          # ~3 min
python src/verification_01/T2_feedback.py           # ~50 min
python src/verification_01/T14_polyakov.py          # ~5 min
python src/verification_01/T21_extremal.py          # ~6 min
python src/verification_02/T26_ks.py                # ~8 min
python src/verification_03/T29_viscosity.py         # ~20 min, then T29_refine.py, T29_refine2.py abd, T29_refine2.py c, T29_interior.py
python src/verification_03/T30_phase_transition.py  # ~10 min
python src/ori_triple_root/ori_model.py             # ~10 min
python src/accretion_tracking/accretion_tracking.py # ~2 min
python src/holography/O1_holography.py              # < 1 min (reads the T16 CSV)
# observables
python src/polar_qnm/qnm_band.py                    # ~1.5 min
python src/polar_qnm/axial_fd.py                    # ~40 min
python src/polar_qnm/axial_fd_fix.py u3=20 s_end=8 "base (scenario)" u2=3 u3=12 s_end=48 sigma_min=0.5 w2=0.6   # ~25 min
python src/verification_01/T18_halo.py              # ~8 min
python src/verification_01/T18_trigger.py           # ~2 min
# appendices
python src/audit_bhp/audit_bhp.py                   # ~5 min
python src/verification_03/T31_rotating_core.py     # ~5 min; T31_retune.py ~15 min; T31_amax.py ~5 min; T31_ring_numeric.py ~3 min
python src/verification_02/T23_rotation.py          # ~2 min
python src/verification_02/T25_constants.py         # ~2 min
python src/defect_field/defect_monopole.py          # ~5 min
python src/ned_lagrangian/ned_reconstruct.py        # ~2 min
python src/polar_qnm/interior_layer.py              # ~3 min
# figures
python figures/make_figures.py                      # ~15 s
```

Order matters in three places. The figures read JSON from `data/`, so they come last. `axial_fd_fix.py` reads `data/polar_qnm/qnm_band.json`. `T21_extremal.py` reads `data/verification_01/T2_feedback.json`, `T25_constants.py` reads the `T15_*.json` files, `T18_halo.py` reads `data/polar_qnm/qnm_band.json` and `T15_qnm.json`, and `O1_holography.py` reads `data/verification_01/T16_base_profile_m_of_r.csv`.

Scripts locate their neighbours through paths relative to their own file, so they can be started from any working directory, but the commands above assume the repository root.

## Determinism

Deterministic, with results depending on the NumPy and SciPy versions only through integrator tolerances: T1, T2, T14, T16, T18, T21, T23, T25, T26, T29, T30, T31, `qnm_band`, `axial_fd*`, `ori_model`, `accretion_tracking`, `audit_bhp`, `defect_monopole`, `ned_reconstruct`, `interior_layer`, `O1_holography`, `make_figures`.

`T15_lambda_bound.py` is the one exception: it performs a 300-point random search with `np.random.default_rng(7)`, which is reproducible at that seed, and starts Nelder-Mead from the best random point. The value λ_max = 1.066 comes from refining that optimum in `T15_qnm.py`, which is deterministic given the same start. At a different seed the bound reproduces to about three digits, which is the limit of the parameterisation rather than of the number.

Grid sensitivity: the base-profile constants are reliable to four digits on the 60001-point LU grid used in T25. Time-domain QNM damping carries a systematic of about 3·10⁻⁴ against the Leaver control; the frequency-domain method reaches 10⁻⁶.

## Results that live only in logs

`audit_bhp.py` writes its thin-shell and stabiliser tables to `logs/audit_bhp/audit_log.txt` and `stabilizer_check.txt` rather than to JSON, and the frequency-domain QNM comparison of `axial_fd.py` and `axial_fd_fix.py` is printed to stdout, captured in `logs/polar_qnm/axial_fd_stdout.txt` and `axial_fd_fix_stdout.txt`. `make_figures.py` parses the latter for figure 6. The numeric summary of the frequency-domain run is additionally written to `data/polar_qnm/axial_fd_fix.json`.

## Known technical negatives

These are recorded in the working notes, are not used in the paper, and are listed here so that a reader who runs everything is not surprised: the gas-hydrodynamics run `settling_hydro/ms_hydro.py` breaks down at t ≈ 97.9 and is not part of this repository; the symbolic Kretschmann computation for the full m(r) in T31 did not terminate and was replaced by a polynomial core; in T29 with n = 0 and ζ₀ > 0 some runs leave the physical domain near R₊ through the Eckart singularity.
