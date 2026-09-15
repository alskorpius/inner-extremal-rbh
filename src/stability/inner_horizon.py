"""Stability of the inner horizon: mass inflation and the search for kappa_- = 0 with a monotone profile.

Class g_tt g_rr = -1: f = 1 - 2 m(R)/R, m' = 4 pi R^2 rho, p_R = -rho, p_perp = -rho - R rho'/2.
Theorem (Dymnikova): WEC holds if and only if rho' <= 0 (rho non-increasing outward).
Inner horizon R_-: f(R_-) = 0, kappa_- = f'(R_-)/2. Standard estimate for mass inflation:
the mass perturbation grows as exp(kappa_- v) (Poisson-Israel, Ori); e-folding over dv = 1/|kappa_-|.
Carballo-Rubio et al. (2205.13556): no exponential growth at kappa_- = 0 (triple root of f).

Question: is a triple root (f = f' = f'' = 0 at R_-) reachable with a MONOTONE rho(R), i.e. without violating WEC/NEC?
Analytics: at R_- the conditions give rho(R_-) = 1/(8 pi R_-^2), rho'(R_-) = -2 rho/R_- < 0 -- locally consistent
with monotonicity. Globally we search numerically within the family
    rho(R) = rho_c / [(1 + (R/R1)^a) (1 + (R/R2)^b)],   a = 2 (halo ~R^-2), b = 6 (cutoff),
h = 2m/R = 8 pi rho_c R1^2 H(u; R2/R1), u = R/R1: we need a stationary inflection point H' = H'' = 0 at u_s,
then the scale rho_c R1^2 = 1/(8 pi H(u_s)) gives h(u_s) = 1 -- a triple root. The mass M fixes R1.
Run: python src/stability/inner_horizon.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve, brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach_map"))
import approach_map as am  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
G_SI, c_SI, M_sun = 6.67430e-11, 299792458.0, 1.98847e30


def shape(u, r21, a=2.0, b=6.0):
    """Dimensionless profile s(u) = 1/((1+u^a)(1+(u/r21)^b)) and H(u) = (2/u) int_0^u s u'^2 du' (units rho_c=R1=1, without 8pi)."""
    return 1.0 / ((1 + u**a) * (1 + (u / r21) ** b))


def H_and_derivs(r21, a=2.0, b=6.0, umax=None, n=200001):
    umax = umax or 30 * r21
    u = np.r_[np.geomspace(1e-6, 1e-2, 2000), np.linspace(1e-2, umax, n)]
    s = shape(u, r21, a, b)
    mI = cumulative_trapezoid(s * u**2, u, initial=0.0)   # int s u^2
    H = 2 * mI / u
    Hp = np.gradient(H, u)
    Hpp = np.gradient(Hp, u)
    return u, s, mI, H, Hp, Hpp


def find_stationary_inflection(a=2.0, b=6.0):
    """Find r21 and u_s such that H'(u_s) = 0 and H''(u_s) = 0 (via grid + refinement)."""
    best = None
    for r21 in np.geomspace(0.5, 50, 400):
        u, s, mI, H, Hp, Hpp = H_and_derivs(r21, a, b)
        # look for zeros of H' in the interval (0.5, 20 r21); if there are none -- min |H'| as a candidate degenerate point
        mask = (u > 0.3) & (u < 20 * r21)
        idx = np.nonzero(np.sign(Hp[mask][:-1]) * np.sign(Hp[mask][1:]) < 0)[0]
        if idx.size == 0:
            j = int(np.argmin(np.abs(Hp[mask])))
            val = abs(Hp[mask][j])
            cand = dict(r21=float(r21), u_s=float(u[mask][j]), Hp=float(Hp[mask][j]), Hpp=float(Hpp[mask][j]), H=float(H[mask][j]), n_zero=0)
            if best is None or val < best["Hp_abs"]:
                cand["Hp_abs"] = val
                best = cand
        else:
            # there are stationary points (local max/min): degeneracy = merger of two zeros; fix as a boundary
            cand = dict(r21=float(r21), n_zero=int(idx.size), u_zero=[float(u[mask][i]) for i in idx], Hp_abs=0.0)
            if best is None or best.get("n_zero", 0) == 0:
                best = best if (best and best["Hp_abs"] < 1e-6) else best
    return best


def scan_family(a=2.0, b=6.0):
    """Scan over r21: number of stationary points of H and minimum |H'| -- boundary between 'S-shaped' and monotone H."""
    rows = []
    for r21 in np.geomspace(0.3, 100, 120):
        u, s, mI, H, Hp, Hpp = H_and_derivs(r21, a, b)
        mask = (u > 0.2) & (u < 25 * r21)
        sgn = np.sign(Hp[mask])
        zeros = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
        j = int(np.argmin(np.abs(Hp[mask])))
        rows.append(dict(r21=float(r21), n_stationary=int(zeros.size), min_abs_Hp=float(abs(Hp[mask][j])), u_at=float(u[mask][j]),
                         H_at=float(H[mask][j]), Hmax=float(H[mask].max()), u_Hmax=float(u[mask][np.argmax(H[mask])])))
    return rows


def build_family(r21, scale, a=2.0, b=6.0, M=1.0):
    """Build a Family for the framework: rho_c R1^2 = scale/(8 pi); R1 from the mass normalization M."""
    u, s, mI, H, Hp, Hpp = H_and_derivs(r21, a, b)
    Itot = mI[-1]                      # int s u^2 du (in units of R1^3 rho_c)
    # M = 4 pi rho_c R1^3 Itot,  rho_c R1^2 = scale/(8 pi)  =>  R1 = M / (4 pi Itot scale/(8 pi)) = 2 M/(Itot scale)
    R1 = 2 * M / (Itot * scale)
    rho_c = scale / (8 * np.pi * R1**2)
    R2 = r21 * R1

    class Fam(am.Family):
        def __init__(self):
            super().__init__("monotone_core", f"monotone a={a} b={b} r21={r21:.3g} scale={scale:.4g}", None, dict(m=M), x_min=1e-3, x_max=12.0)
            Rg = u * R1
            self._Rg, self._mg = Rg, 4 * np.pi * rho_c * R1**3 * mI

        def rho(self, R):
            return rho_c * shape(np.asarray(R, float) / R1, r21, a, b)

        def drho(self, R, h=1e-6):
            R = np.asarray(R, float)
            return (self.rho(R * (1 + h)) - self.rho(R * (1 - h))) / (2 * R * h)

        def geometry(self, x):
            R = np.asarray(x, float)
            m = np.interp(R, self._Rg, self._mg)
            m1 = 4 * np.pi * R**2 * self.rho(R)
            m2 = 8 * np.pi * R * self.rho(R) + 4 * np.pi * R**2 * self.drho(R)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = 1 - 2 * m / R
                fp = 2 * m / R**2 - 2 * m1 / R
                fpp = -4 * m / R**3 + 4 * m1 / R**2 - 2 * m2 / R
                D = (2 * m / R) / R**2
            return dict(R=R, Rp=np.ones_like(R), Rpp=np.zeros_like(R), f=f, fp=fp, fpp=fpp, D=D)
    fam = Fam()
    return fam, dict(R1=R1, R2=R2, rho_c=rho_c, ell=np.sqrt(3 / (8 * np.pi * rho_c)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    # 1. mass inflation for the de Sitter-core transition family (from ks_curvature_transition): kappa_- ~ -1/ell
    print("1. Rate of mass inflation for de Sitter cores: e-folding over dv = 1/|kappa_-| ~ ell/c")
    rows = []
    for name, ell_m in (("50 rho_nuc", 3.39e3), ("V_EW", 0.0763), ("Planck", 5.59e-36)):
        dv = ell_m / c_SI
        rows.append(dict(rho_c=name, ell_m=ell_m, efold_time_s=dv))
        print(f"   rho_c={name:10s}: ell={ell_m:.3g} m, e-folding {dv:.3g} s; per 1 s -- {1/dv:.3g} e-foldings")
    summary["mass_inflation_rate"] = rows

    # 2. scan of the monotone profile family: is there an S-shaped H (two stationary points) and the degeneracy boundary
    print("\n2. Monotone profile rho = rho_c/((1+(R/R1)^2)(1+(R/R2)^6)): structure of H(u) = 2m/(R 8 pi rho_c R1^2)")
    rows = scan_family()
    summary["scan_monotone"] = rows
    n_with = [r for r in rows if r["n_stationary"] > 0]
    print(f"   r21 from {rows[0]['r21']:.3g} to {rows[-1]['r21']:.3g}: number of values with stationary points of H: {len(n_with)} out of {len(rows)}")
    for r in rows[::12]:
        print(f"   r21={r['r21']:7.3g}: stationary points of H' = {r['n_stationary']}, min|H'|={r['min_abs_Hp']:.3g} at u={r['u_at']:.3g}, H there={r['H_at']:.4g}; Hmax={r['Hmax']:.4g} at u={r['u_Hmax']:.3g}")

    # 3. if H increases monotonically to a maximum and then decreases (no S-shape), a triple root is impossible:
    #    f = 1 - scale*H; horizons are intersections of scale*H = 1; kappa_- = -(scale H'(u_-))/(2 R1). Let's check how small
    #    |kappa_-| can be for a monotone profile: kappa_- -> 0 only as u_- -> u_Hmax, but there f'' > 0 and f does not change sign (double root).
    print("\n3. Minimum achievable |kappa_-| for a monotone profile: scan of scale at r21=3 (inner root approaching the maximum of H)")
    r21 = 3.0
    u, s, mI, H, Hp, Hpp = H_and_derivs(r21)
    Hmax = H.max()
    rows3 = []
    for frac in (1.5, 1.2, 1.05, 1.01, 1.001):
        scale = frac / Hmax                 # scale*Hmax = frac > 1: two horizons; frac -> 1: merger
        fam, meta = build_family(r21, scale)
        s_, x, T = am.characterize(fam)
        hs = s_["horizons"]
        row = dict(scale_Hmax=frac, n_horizons=len(hs), R_minus=s_.get("R_minus"), R_plus=s_.get("R_plus"), kappa_minus=s_.get("kappa_minus"),
                   kappa_plus=s_.get("kappa_plus"), ell=meta["ell"], kappa_minus_ell=(s_.get("kappa_minus") or np.nan) * meta["ell"],
                   K_max=s_["K_max"], K_center=s_["K_center"], WEC_violated=s_["WEC_violated"], NEC_violated=s_["NEC_violated"],
                   min_nec_t=s_["min_nec_t"], T_H_over_schw=s_.get("T_H_over_schw"))
        rows3.append(row)
        print(f"   scale*Hmax={frac:6g}: horizons={len(hs)}, R-={row['R_minus']}, R+={row['R_plus']}, kappa-={row['kappa_minus']}, kappa+={row['kappa_plus']}, "
              f"|kappa-|*ell={abs(row['kappa_minus_ell']):.3g}, K_max={row['K_max']:.3g}, WEC viol={row['WEC_violated']}, NEC viol={row['NEC_violated']}, T_H/S={row['T_H_over_schw']}")
    summary["monotone_kappa_scan"] = rows3
    (OUT / "inner_horizon.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n->", OUT)


if __name__ == "__main__":
    main()
