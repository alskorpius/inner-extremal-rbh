"""Generalized Ori model (Carballo-Rubio et al. 2021, arXiv:2101.05006, equations (4), (10), (32)) for the scenario profile
with a triple interior root (kappa_0 = 0), compared against Reissner-Nordstrom and Hayward.

Metric ds^2 = -f dv^2 + 2 dv dr + r^2 dOmega^2, f = 1 - 2 M(m(v), r)/r; the outgoing lightlike shell Sigma splits
space into R- (mass m-(v) = m0 - beta/v^p, Price tail, p >= 11) and R+ (mass m+(v)).
Equations: (10) dR/dv = f-(m-(v), R)/2;  (32) (1/f+) dM+/dv|_r = (1/f-) dM-/dv|_r  =>
    m+'(v) = m-'(v) (f+/f-) (dM/dm)(m-,R) / (dM/dm)(m+,R).
The physical quantity is the Misner-Sharp mass on the shell M+(v) = M(m+(v), R(v)).
Expectations: RN - M+ ~ e^{|kappa0| v}/v^{p+1} (Ori); Hayward - exponential, then M+ ~ v^{p+1} (their (20), (24));
triple root - f- ~ -c (R - r0)^3, R - r0 ~ v^{-1/2}, m+' ~ v^{1/2-p}: bounded growth (the integral converges for p >= 11).
Run: python src/ori_triple_root/ori_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

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


scen = lb.ScenarioMass()


def M_rn(m, r, e=0.9):
    return m - e**2 / (2 * r), 1.0


def M_hay(m, r, ell=0.3):
    return m * r**3 / (r**3 + 2 * m * ell**2), r**6 / (r**3 + 2 * m * ell**2) ** 2


# d_r M at fixed m as a function of (M+, R): RN: e^2/(2R^2); Hayward: 6 ell^2 M^2/R^4 (CR 2021, (15), (17)) - valid even when m+ passes through infinity
def dMdr_rn(Mp, r, e=0.9):
    return e**2 / (2 * r**2)


def dMdr_hay(Mp, r, ell=0.3):
    return 6 * ell**2 * Mp**2 / r**4


def M_scen(m, r):
    d = scen.all(m, r)
    return float(d["m"]), float(d["m_M"])


def horizons(Mfun, m, lo=1e-3, hi=4.0):
    rg = np.geomspace(lo, hi, 20001)
    h = np.array([2 * Mfun(m, r)[0] / r - 1 for r in rg])
    idx = np.nonzero(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]
    from scipy.optimize import brentq
    out = []
    for i in idx:
        rr = brentq(lambda r: 2 * Mfun(m, r)[0] / r - 1, rg[i], rg[i + 1], xtol=1e-13)
        eps = 1e-6
        fp = ((1 - 2 * Mfun(m, rr + eps)[0] / (rr + eps)) - (1 - 2 * Mfun(m, rr - eps)[0] / (rr - eps))) / (2 * eps)
        out.append((rr, 0.5 * fp))
    return out


def run(name, Mfun, m0=1.0, p=11, eps0=1e-3, v0=50.0, v_end=1e7, R0=None, dm_plus=1e-4, dMdr=None, hay_ell=None):
    """dMdr: if given, integrate the physical pair (R, M+) via dM+/dv = m-' (f+/f-) A- + dMdr(M+,R) R_v (passage of m+ through infinity is safe)."""
    beta = eps0 * m0 * v0**p
    hz = horizons(Mfun, m0)
    r_minus = hz[0][0]; kap0 = hz[0][1]
    R0 = R0 or 0.5 * (hz[0][0] + hz[-1][0])

    def m_minus(v):
        return m0 - beta / v**p

    def dm_minus(v):
        return p * beta / v ** (p + 1)

    def rhs(lv, y):
        v = np.exp(lv)
        R, q = y
        mm = m_minus(v)
        Mm, Am = Mfun(mm, R)
        fm = 1 - 2 * Mm / R
        dR = fm / 2
        if hay_ell is not None:   # q = w = 1/m+ (Hayward): regular as m+ -> inf
            Mp = R**3 / (R**3 * q + 2 * hay_ell**2)
            fpl = 1 - 2 * Mp / R
            dq = -dm_minus(v) * (fpl / fm) * Am * (R**3 * q + 2 * hay_ell**2) ** 2 / R**6
        elif dMdr is None:
            Mp, Ap = Mfun(q, R)
            fpl = 1 - 2 * Mp / R
            dq = dm_minus(v) * (fpl / fm) * Am / Ap
        else:
            fpl = 1 - 2 * q / R
            dq = dm_minus(v) * (fpl / fm) * Am + dMdr(q, R) * dR
        return [v * dR, v * dq]

    y0 = [R0, m_minus(v0) * (1 + dm_plus) if dMdr is None else Mfun(m_minus(v0) * (1 + dm_plus), R0)[0]]
    if hay_ell is not None:
        y0 = [R0, 1.0 / (m_minus(v0) * (1 + dm_plus))]
    lv = np.linspace(np.log(v0), np.log(v_end), 1500)
    sol = solve_ivp(rhs, (lv[0], lv[-1]), y0, method="LSODA", rtol=1e-9, atol=1e-13, t_eval=lv)
    say(f"  [{name[:20]}] integrator: status {sol.status}, {sol.message}, points {len(sol.t)}")
    v = np.exp(sol.t); R, mp = sol.y
    if dMdr is not None:
        Mplus_direct = mp.copy(); mp = np.full_like(mp, np.nan)
    if hay_ell is not None:
        w = mp.copy(); Mplus_direct = R**3 / (R**3 * w + 2 * hay_ell**2); mp = np.where(np.abs(w) > 1e-300, 1.0 / w, np.inf); dMdr = True
    if len(v) < 3:
        say("  too few points - skipping"); return dict(name=name, status=int(sol.status), samples=[], M_plus_end=np.nan, r_minus=hz[0][0]), dict(v=v, R=R, mp=mp, Mplus=R*0, fm=R*0)
    Mplus = Mplus_direct if dMdr is not None else np.array([Mfun(a, b)[0] for a, b in zip(mp, R)])
    dm_total = float(mp[-1] - mp[0]) if (dMdr is None and hay_ell is None) else None
    fm = np.array([1 - 2 * Mfun(m_minus(vv), r)[0] / r for vv, r in zip(v, R)])
    # local growth rates
    dlnM_dv = np.gradient(np.log(np.abs(Mplus)), v)
    dlnM_dlnv = np.gradient(np.log(np.abs(Mplus)), np.log(v))
    res = dict(name=name, status=int(sol.status), v_last=float(v[-1]), kappa0=float(kap0), r_minus=float(r_minus), horizons=[(float(a), float(b)) for a, b in hz],
               R_end=float(R[-1]), R_minus_r0_end=float(R[-1] - r_minus), f_minus_end=float(fm[-1]), m_plus_end=float(mp[-1]), M_plus_end=float(Mplus[-1]),
               M_plus_max=float(np.nanmax(np.abs(Mplus))), dm_plus_total=dm_total, samples=[])
    for vv in (v0 * 2, 1e2, 1e3, 1e4, 1e5, 1e6, v[-1]):
        i = int(np.argmin(np.abs(v - vv)))
        res["samples"].append(dict(v=float(v[i]), R_minus_r0=float(R[i] - r_minus), f_minus=float(fm[i]), m_plus=float(mp[i]), M_plus=float(Mplus[i]),
                                   dlnM_dv=float(dlnM_dv[i]), dlnM_dlnv=float(dlnM_dlnv[i])))
    say(f"\n=== {name}: horizons (R, kappa) = {[(round(a,5), round(b,5)) for a, b in hz]}; kappa0 = {kap0:+.5f}; start R0 = {R0:.4f}, m+(v0) = m-(v0)(1 + {dm_plus}); p = {p}, delta m/m0 = {eps0} at v0 = {v0}; status {sol.status}, v_last = {v[-1]:.3g}")
    if dm_total is not None:
        say(f"  total change of the m+ parameter over the integration: {dm_total:+.3e} (relative {dm_total/mp[0]:+.3e})")
    for s_ in res["samples"]:
        say(f"  v = {s_['v']:9.3g}: R - r0 = {s_['R_minus_r0']:+.3e}, f- = {s_['f_minus']:+.3e}, m+ = {s_['m_plus']:.6g}, M+(v,R) = {s_['M_plus']:.6g}, d ln M+/dv = {s_['dlnM_dv']:+.3e}, d ln M+/d ln v = {s_['dlnM_dlnv']:+.3f}")
    return res, dict(v=v, R=R, mp=mp, Mplus=Mplus, fm=fm)


def main():
    results = []
    say("Generalized Ori model: m-(v) = m0 - beta/v^p, dR/dv = f-/2, m+' = m-' (f+/f-) A-/A+ (equation (32) CR 2021).")
    r1, d1 = run("Reissner-Nordstrom e = 0.9 (control: expected d ln M+/dv -> |kappa0| - (p+1)/v)", lambda m, r: M_rn(m, r, 0.9), v0=20.0, v_end=90.0, dMdr=dMdr_rn)
    results.append(r1)
    r2, d2 = run("Hayward ell = 0.3 (expectation: exponential, then M+ ~ v^{p+1} = v^12)", lambda m, r: M_hay(m, r, 0.3), v0=20.0, v_end=3e3, hay_ell=0.3)
    results.append(r2)
    r3, d3 = run("Scenario (branch A, triple root, ell fixed; kappa0 = 0)", M_scen, v_end=1e7)
    results.append(r3)
    # scenario sensitivity: sign and magnitude of the initial m+ shift, tail amplitude, p = 12
    for dmp in (-1e-4, 1e-2):
        r_, _ = run(f"Scenario, m+(v0) = m-(v0)(1 + {dmp})", M_scen, v_end=1e6, dm_plus=dmp)
        results.append(r_)
    r_, _ = run("Scenario, p = 12, delta m/m0 = 1e-2", M_scen, v_end=1e6, p=12, eps0=1e-2)
    results.append(r_)
    r_, _ = run("Scenario, worst case: delta m/m0 = 0.1 at v0 = 50, start R0 = r0 + 0.05", M_scen, v_end=1e6, eps0=0.1, R0=0.67604 + 0.05)
    results.append(r_)
    # triple root: analytic expectation R - r0 ~ v^{-1/2}
    v, R = d3["v"], d3["R"]
    k = (v > 1e4)
    slope = np.polyfit(np.log(v[k]), np.log(R[k] - r3["r_minus"]), 1)[0]
    say(f"\nScenario: exponent of the shell's approach to r0: d ln(R - r0)/d ln v = {slope:.3f} (expected -1/2 for a triple root); "
        f"total increase of M+ for v from 50 to 1e7: {r3['M_plus_end'] - r3['samples'][0]['M_plus']:+.3e} (bounded)")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for d, lab in ((d1, "RN e=0.9"), (d2, "Hayward ell=0.3"), (d3, "scenario (triple root)")):
            ax[0].loglog(d["v"], np.abs(d["Mplus"]), label=lab)
            ax[1].loglog(d["v"], np.abs(d["fm"]), label=lab)
        ax[0].set_xlabel("v"); ax[0].set_ylabel("|M+(v, R(v))|"); ax[0].legend(); ax[0].set_title("Misner-Sharp mass behind the shell")
        ax[1].set_xlabel("v"); ax[1].set_ylabel("|f-(R(v))|"); ax[1].legend(); ax[1].set_title("shell's approach to the inner horizon")
        fig.tight_layout(); fig.savefig(OUT / "ori_model.png", dpi=110)
    except Exception as e:  # noqa: BLE001
        say(f"plot not produced: {e}")
    json.dump(results, open(OUT / "ori_model.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
