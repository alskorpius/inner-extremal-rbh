"""The cost of the nonlocal law ell = lambda*M under accretion: generalized Vaidya metric
    ds^2 = -f(v,R) dv^2 + 2 dv dR + R^2 dOmega^2,  f = 1 - 2 m(M(v), R)/R,  G = c = 1.
Two laws: (F) ell fixed (the profile does not track the mass); (T) ell = lambda*M(v) -- the profile tracks the mass (scenario).
Einstein tensor (sympy): G^v_v = G^R_R = -2 m_R/R^2 (rho = m_R/(4 pi R^2) = -p_r), G^th_th = -m_RR/R (p_perp),
    G_vv|extra = 2 m_v/R^2 -- the ingoing light flux mu = m_v/(4 pi R^2) = m_M Mdot/(4 pi R^2);
NEC along the ingoing null vector n = -d/dR: T(n,n) = rho + p_r = 0; along the outgoing l = d/dv + (f/2) d/dR: T(l,l) = mu (plus 0).
The sign of the flux equals the sign of m_M(M, R): for law T inside the core m_M < 0 -- an INGOING NEGATIVE-ENERGY FLUX is required.
Kodama surface gravity: kappa = (1/2) d f/dR at f = 0 (Box_2 R = d_R f, verified with sympy).
Run: python src/accretion_tracking/accretion_tracking.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "inhomogeneous_collapse"))
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- symbolic check of the components for the generalized Vaidya metric
def sympy_check():
    v, R, th, ph = sp.symbols("v R theta phi", real=True)
    m = sp.Function("m")(v, R)
    f = 1 - 2 * m / R
    x = [v, R, th, ph]
    g = sp.Matrix([[-f, 1, 0, 0], [1, 0, 0, 0], [0, 0, R**2, 0], [0, 0, 0, R**2 * sp.sin(th) ** 2]])
    gi = g.inv()
    n = 4
    Gam = [[[sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d])) for d in range(n)) / 2) for c in range(n)] for b in range(n)] for a in range(n)]
    Ric = sp.zeros(4)
    for b in range(n):
        for d in range(n):
            Ric[b, d] = sp.simplify(sum(sp.diff(Gam[a][b][d], x[a]) - sp.diff(Gam[a][b][a], x[d]) + sum(Gam[a][a][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][a] for e in range(n)) for a in range(n)))
    Rs = sp.simplify(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
    Gdn = sp.simplify(Ric - Rs * g / 2)
    Gmix = sp.simplify(gi * Gdn)
    mR, mv, mRR = sp.diff(m, R), sp.diff(m, v), sp.diff(m, R, 2)
    c1 = sp.simplify(Gmix[0, 0] + 2 * mR / R**2)
    c2 = sp.simplify(Gmix[1, 1] + 2 * mR / R**2)
    c3 = sp.simplify(Gmix[2, 2] + mRR / R)
    # T(l,l) for l = d_v + (f/2) d_R ; T(n,n) for n = -d_R
    l = sp.Matrix([1, f / 2, 0, 0]); nn = sp.Matrix([0, -1, 0, 0])
    Tll = sp.simplify((l.T * Gdn * l)[0] / (8 * sp.pi))
    Tnn = sp.simplify((nn.T * Gdn * nn)[0] / (8 * sp.pi))
    box2 = sp.simplify(sp.diff(gi[0, 1], v) + sp.diff(gi[1, 1], R))  # sqrt(-g2) = 1
    say(f"sympy: G^v_v + 2m_R/R^2 = {c1}; G^R_R + 2m_R/R^2 = {c2}; G^th_th + m_RR/R = {c3}; "
        f"T(l,l) = {Tll} (expected m_v/(4 pi R^2)); T(n,n) = {Tnn}; Box_2 R = {box2} (expected d_R f)")


# ---------------- laws m(M, R)
scen = lb.ScenarioMass()
lam = scen.ell            # ell = lambda*M at M = 1
Itot = scen.Itot


def m_scen_fixed(M, R):
    d = scen.all(M, R)
    return d["m"], d["m_M"], d["m_R"], d["m_RR"]


def m_scen_track(M, R):
    """ell = lam*M: R1 = M (2 lam^2/(3 Itot))^{1/3}; m = (M/Itot) mI(u); m_M = (mI - s u^3)/Itot."""
    M = np.asarray(M, float); R = np.asarray(R, float)
    R1 = M * (2 * lam**2 / (3 * Itot)) ** (1 / 3)
    u = R / R1; lu = np.log(u)
    s = np.exp(np.interp(lu, scen.LU, scen.ln_s, left=0.0, right=-np.inf))
    sig = np.interp(lu, scen.LU, scen.sig, left=0.0, right=scen.sig[-1])
    mI = np.where(lu < scen.LU[0], u**3 / 3, np.interp(lu, scen.LU, scen.mI, right=Itot))
    m = (M / Itot) * mI
    m_M = (mI - s * u**3) / Itot
    rho_c = 3 / (8 * np.pi * (lam * M) ** 2)
    m_R = 4 * np.pi * R**2 * rho_c * s
    m_RR = 4 * np.pi * R * rho_c * s * (2 - sig)
    return m, m_M, m_R, m_RR


def hayward_laws(lam_h):
    M, R = sp.symbols("M R", positive=True)
    out = {}
    for name, ell in (("fixed", sp.Float(lam_h)), ("track", lam_h * M)):
        expr = M * R**3 / (R**3 + 2 * M * ell**2)
        fs = [sp.lambdify((M, R), e, "numpy") for e in (expr, sp.diff(expr, M), sp.diff(expr, R), sp.diff(expr, R, 2))]
        out[name] = lambda Mv, Rv, fs=fs: tuple(np.asarray(f(Mv, Rv), float) + 0.0 * np.asarray(Rv, float) for f in fs)
    return out


def horizons(law, M):
    Rg = np.geomspace(1e-3, 4.0, 20001)
    m = law(M, Rg)[0]
    h = 2 * m / Rg - 1
    idx = np.nonzero(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]
    res = []
    for i in idx:
        Rr = brentq(lambda R: 2 * float(law(M, R)[0]) / R - 1, Rg[i], Rg[i + 1], xtol=1e-13)
        m_, mM, mR, _ = law(M, Rr)
        res.append((Rr, float(0.5 * (2 * m_ / Rr**2 - 2 * mR / Rr))))
    # triple/double root (tangency): minima of |h| without a sign change
    return res, Rg, h


def analyse(name, law, M0=1.0):
    say(f"\n=== {name}")
    Rg = np.geomspace(1e-3, 3.0, 30001)
    for M in (M0, 1.01 * M0, 1.1 * M0, 2 * M0):
        m, mM, mR, mRR = law(M, Rg)
        hz, _, h = horizons(law, M)
        neg = Rg[mM < 0]
        Rstar = float(neg.max()) if len(neg) else 0.0
        worst = float(mM.min()); Rw = float(Rg[np.argmin(mM)])
        rho_v = mR / (4 * np.pi * Rg**2)
        # fraction of the accretion rate going into the negative flux: max over R |m_M^-| ; integral over R: I = int_0^{R*} |m_M| dR / R*
        I = float(np.trapezoid(np.where(mM < 0, -mM, 0.0), Rg))
        # flux magnitude relative to the core density: mu/rho_v = m_M Mdot / m_R -> per unit Mdot
        ratio = float(np.nanmin(np.where(mR > 0, mM / mR, np.nan)))
        touch = ""
        if len(hz) < 2:
            # look for a tangency h = 0 (degenerate root)
            j = np.argmin(np.abs(h[(Rg > 0.05) & (Rg < 1.5)]))
            touch = f"; min|2m/R-1| in 0.05<R<1.5: {np.abs(h[(Rg > 0.05) & (Rg < 1.5)])[j]:.2e} at R={Rg[(Rg > 0.05) & (Rg < 1.5)][j]:.4f}"
        say(f"M={M:.3g}: horizons (R, kappa): {[(round(R, 5), round(k, 5)) for R, k in hz]}{touch}; "
            f"m_M<0 for R < {Rstar:.4f} (min m_M = {worst:.4f} at R={Rw:.4f}); int |m_M^-| dR = {I:.4f}; min (mu/rho_v)/Mdot = {ratio:.3g}")
    return None


def main():
    sympy_check()
    say(f"\nScenario: lambda = ell/M = {lam:.5f}; branch-A profile. Hayward: lambda = {lam:.4f} (same ell at M = 1).")
    hl = hayward_laws(lam)
    analyse("Hayward, ell fixed (law F)", hl["fixed"])
    analyse("Hayward, ell = lambda*M (law T)", hl["track"])
    analyse("Scenario (branch A), ell fixed (law F)", m_scen_fixed)
    analyse("Scenario (branch A), ell = lambda*M (law T)", m_scen_track)
    # physical estimate: Mdot in units of c^3/G
    G, c, Msun, yr = 6.67430e-11, 2.99792458e8, 1.98847e30, 3.15576e7
    for Mdot_sun_yr, label in ((2e-8, "10 M_sun, Eddington rate ~2e-8 M_sun/yr"), (1e-2, "super-critical accretion 1e-2 M_sun/yr")):
        Mdot = Mdot_sun_yr * Msun / yr * G / c**3
        say(f"Mdot ({label}) = {Mdot:.2e} (dimensionless, G=c=1): mu/rho_v in the core ~ {Mdot:.1e} x (m_M/m_R)")
    (LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
