"""Stability diagnostics of the transition layer: effective "sound speeds" of the anisotropic vacuum-like medium.

For the class p_R = -rho the radial c_R^2 = dp_R/drho = -1 identically (as for Lambda: there are no radial sound
modes -- this is not an instability but the absence of a degree of freedom, if the medium is genuinely vacuum-like).
The transverse c_perp^2 = dp_perp/drho along the profile: where < 0 -- treating the medium as a fluid, the layer is
gradient-unstable to transverse perturbations; where > 1 -- superluminal sound. This is a diagnostic, not a rigorous
perturbation analysis: the medium has no Lagrangian (see eos_condensate, sec. 3, and the literature review).
Additionally -- Herrera's "cracking" criterion for anisotropic configurations: the sign of the derivative of the
anisotropy Delta = p_perp - p_R with respect to rho; a sign change of d(Delta)/d(rho) inside the layer signals
possible cracking (delamination).
Run: python src/stability/layer_sound.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach_map"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ks_curvature_transition"))
import approach_map as am  # noqa: E402
from ks import KSFamily  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name


def diagnose(fam, R_lo, R_hi, label):
    R = np.linspace(R_lo, R_hi, 6001)
    T = am.tensors(fam, R)
    rho = np.asarray(T["rho8pi"]) / (8 * np.pi)
    pR = np.asarray(T["pr8pi"]) / (8 * np.pi)
    pt = np.asarray(T["pt8pi"]) / (8 * np.pi)
    drho = np.gradient(rho, R)
    ok = np.abs(drho) > 1e-12 * np.max(np.abs(drho))
    cs2_t = np.full_like(R, np.nan)
    cs2_t[ok] = np.gradient(pt, R)[ok] / drho[ok]
    Delta = pt - pR
    dDelta = np.full_like(R, np.nan)
    dDelta[ok] = np.gradient(Delta, R)[ok] / drho[ok]
    neg = R[np.nan_to_num(cs2_t, nan=0.0) < -1e-9]
    sup = R[np.nan_to_num(cs2_t, nan=0.0) > 1 + 1e-9]
    out = dict(label=label, cs2_perp_min=float(np.nanmin(cs2_t)), cs2_perp_max=float(np.nanmax(cs2_t)),
               region_cs2_neg=[float(neg.min()), float(neg.max())] if neg.size else None,
               region_cs2_gt1=[float(sup.min()), float(sup.max())] if sup.size else None,
               dDelta_drho_sign_changes=int(np.sum(np.diff(np.sign(np.nan_to_num(dDelta))) != 0)),
               cs2_radial=-1.0)
    print(f"  {label}: c_perp^2 in [{out['cs2_perp_min']:.3g}, {out['cs2_perp_max']:.3g}]; c_perp^2<0 on R∈{out['region_cs2_neg']}; "
          f"c_perp^2>1 on R∈{out['region_cs2_gt1']}; sign changes of d(Delta)/d(rho): {out['dDelta_drho_sign_changes']}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Curvature transition layer (M=1): transverse effective sound speed along the profile")
    for Bc in (0.5, 0.2):
        for d in (0.3, 0.1, 0.03):
            B_c = Bc * 2.0
            fam = KSFamily(1.0, B_c, d * B_c)
            rows.append(diagnose(fam, max(1e-3, B_c - 8 * d * B_c), B_c + 8 * d * B_c, f"B_c/2M={Bc}, δ/B_c={d}"))
    print("\nControls: Hayward ell=2/3, Dymnikova r*=0.96 (whole profile 0.001..3)")
    for fam, lab in ((am.Family("h", "Hayward", am.F_hayward, dict(m=1.0, ell=2 / 3)), "Hayward ell=2/3"),
                     (am.Family("d", "Dymnikova", am.F_dymnikova, dict(m=1.0, rstar=0.9615)), "Dymnikova r*=0.96")):
        rows.append(diagnose(fam, 1e-3, 3.0, lab))
    (OUT / "layer_sound.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", OUT)


if __name__ == "__main__":
    main()
