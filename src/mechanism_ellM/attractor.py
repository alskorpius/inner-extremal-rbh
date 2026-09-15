"""Dynamical attractor toward extremality: evaporation, accretion, and detuning of the tuned triple root.

(1) Hayward and NPQT-4D with fixed fundamental ell: quasi-static evolution M(v) under evaporation
    dM/dv = -C T_H^4 A_H ∝ -kappa_+^4 R_+^2 (Stefan-Boltzmann in terms of temperature and area of the regular BH's outer horizon);
    the trajectory kappa_-(M) and the fraction of the lifetime spent at |kappa_- ell| < 0.1 (close to extremality).
(2) Accretion: M grows => kappa_- -> -1/ell (away from extremality).
(3) Branch A (triple root) at fixed physical profile scale R1 and varying mass M:
    the level h(u_s) = 1 is detuned; we compute kappa_- for M/M0 = 0.9 ... 1.1 -- is there recovery.
Run: python src/mechanism_ellM/attractor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
for p in (HERE.parents[0] / "approach_map", HERE.parents[0] / "stability"):
    sys.path.insert(0, str(p))
import approach_map as am  # noqa: E402
import triple_root_monotone as t  # noqa: E402

OUT = HERE.parents[1] / "data" / HERE.name


def horizons_hayward(M, ell):
    fam = am.Family("h", "h", am.F_hayward, dict(m=M, ell=ell), x_min=1e-4, x_max=max(12.0, 3 * M))
    return am.horizons(fam)


def evaporation_track(ell=1.0, M0=10.0, C=1.0, n=400):
    """quasi-static: dM/dv = -C kappa_+^4 R_+^2 (dimensionless normalization); M from M0 to M_cr = 3 sqrt3/4 ell."""
    M_cr = 3 * np.sqrt(3) / 4 * ell
    Ms = np.linspace(M0, M_cr * 1.0005, n)
    rows = []
    for M in Ms:
        hs = horizons_hayward(M, ell)
        if len(hs) < 2:
            continue
        rows.append(dict(M=M, R_plus=hs[-1]["R"], R_minus=hs[0]["R"], kappa_plus=hs[-1]["kappa"], kappa_minus=hs[0]["kappa"]))
    M = np.array([r["M"] for r in rows]); kp = np.array([r["kappa_plus"] for r in rows]); km = np.array([r["kappa_minus"] for r in rows]); Rp = np.array([r["R_plus"] for r in rows])
    rate = C * kp**4 * Rp**2                      # dM/dv magnitude
    # time: dv = dM/rate; integrate from M0 downward
    dv = np.abs(np.gradient(M)) / rate
    v = np.cumsum(dv)
    total = v[-1]
    near = np.abs(km * ell) < 0.1
    frac_near = float(np.sum(dv[near]) / total)
    near5 = np.abs(km * ell) < 0.5
    frac_near5 = float(np.sum(dv[near5]) / total)
    return dict(ell=ell, M0=M0, M_cr=M_cr, total_time=float(total), frac_time_kappa_lt_0p1=frac_near, frac_time_kappa_lt_0p5=frac_near5,
                kappa_minus_at_M0=float(km[0]), kappa_minus_at_end=float(km[-1]), M_at_kappa_lt_0p1=float(M[near].max()) if near.any() else None,
                track=[dict(M=float(m), kappa_minus_ell=float(k * ell), kappa_plus=float(p)) for m, k, p in zip(M[::40], km[::40], kp[::40])])


def branchA_detune():
    base = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    s1s, u_s, info = t.find_merge_first(sigma_min=1.0, base=base)
    kw = info["kw"]; d = s1s - 1.0
    sig, s, mI, H, Hp = t.profile(d, **kw)
    scale0 = 1.0 / float(np.interp(u_s, t.U, H))
    fam0 = t.MonotoneFamily(d, scale0, kw=kw)
    R1 = fam0.R1
    rows = []
    for ratio in (0.90, 0.95, 0.99, 1.0, 1.01, 1.05, 1.10):
        # fix the profile's physical scale R1 and rho_c, vary M: profile mass M' = ratio*M0 means scale' = scale0*ratio
        # (h = scale*H at fixed shape and R1: h ∝ rho_c R1^2 ∝ M/R1... from R1 = M/(scale Itot): at fixed R1, scale ∝ M)
        fam = t.MonotoneFamily(d, scale0 * ratio, kw=kw)
        # MonotoneFamily recomputes R1 from M=1 and scale; to hold R1 fixed, we set M=ratio: R1 = M/(scale*Itot) = ratio/(scale0*ratio*Itot) = R1_0
        fam = t.MonotoneFamily(d, scale0 * ratio, M=ratio, kw=kw)
        hs = am.horizons(fam)
        rows.append(dict(M_over_M0=ratio, R1=fam.R1, horizons=[(round(h["R"], 5), round(h["kappa"], 6)) for h in hs]))
        print(f"  M/M0={ratio:5g} (R1 fixed = {fam.R1:.4f}): horizons (R, kappa) = {rows[-1]['horizons']}")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("(1) Hayward evaporation at fixed ell = 1: trajectory kappa_-(M) from M0 = 10 to M_cr = 1.299")
    ev = evaporation_track()
    for r in ev["track"]:
        print(f"    M={r['M']:7.3f}: kappa_- ell = {r['kappa_minus_ell']:+.4f}, kappa_+ = {r['kappa_plus']:.4f}")
    print(f"  fraction of lifetime at |kappa_- ell| < 0.1: {ev['frac_time_kappa_lt_0p1']:.2e}; at < 0.5: {ev['frac_time_kappa_lt_0p5']:.2e}; "
          f"|kappa_- ell| < 0.1 only for M < {ev['M_at_kappa_lt_0p1']}")
    print("  Conclusion: kappa_- ~ -1/ell for almost the entire lifetime; extremality occurs only in the final Planckian stretch (M -> M_cr).")
    print("\n(2) Accretion: kappa_- as M grows (ell = 1):")
    acc = []
    for M in (1.3, 1.5, 2, 5, 10, 100):
        hs = horizons_hayward(M, 1.0); acc.append(dict(M=M, kappa_minus=hs[0]["kappa"] if len(hs) >= 2 else None))
        print(f"    M={M:5g}: kappa_- = {acc[-1]['kappa_minus']}")
    print("  Conclusion: as mass grows, kappa_- -> -1/ell -- away from extremality.")
    print("\n(3) Branch A: detuning of the triple root under varying M at fixed profile scale R1")
    det = branchA_detune()
    print("  Conclusion: for any deviation of M from M0 the triple root splits (kappa_- != 0 or an extra pair of horizons); there is no restoring dynamics within the static family -- evolution of the profile itself is required.")
    (OUT / "attractor.json").write_text(json.dumps(dict(evaporation=ev, accretion=acc, detune=det), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
