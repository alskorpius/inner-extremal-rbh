"""Eikonal (light-ring) quasinormal modes of the rotating inner-extremal regular BH of
Franzin-Liberati-Mazza-Vellucci 2022 (arXiv:2207.08864; coefficients (29)-(32) per the companion paper's rotating-metric audit) against Kerr.
Metric (equator, theta = pi/2, Sigma = r^2): g_tt = -(1 - 2m/r), g_tphi = -2 a m/r, g_phiphi = ((r^2+a^2)^2 - Delta a^2)/r^2,
g_rr = r^2/Delta, Delta = r^2 - 2 m r + a^2, m(r) = M (r^2 + alpha r + beta)/(r^2 + gamma r + mu); the conformal factor Psi/Sigma
does not affect null geodesics (the light-ring position, Omega_c, and the Lyapunov exponent in t are invariant).
Eikonal (Cardoso et al. 2009): omega_QNM ≈ m Omega_c - i (n + 1/2) lambda_L, lambda_L = sqrt(V_r''(r_c)/(2 tdot^2)),
V_r = rdot^2 = (E^2 g_pp + 2 E L g_tp + L^2 g_tt)/(D g_rr), D = g_tp^2 - g_tt g_pp, tdot = (E g_pp + L g_tp)/D.
Controls: Schwarzschild Omega_c = lambda = 1/(3 sqrt 3); Kerr r_c = 2M[1 + cos((2/3) arccos(∓a/M))], Omega_c = ±1/(r_c^{3/2} ± a); e = 0 is exactly Kerr.
Run: python src/rotating_eikonal/eikonal_qnm.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
LOGS = Path(__file__).resolve().parents[2] / "logs" / Path(__file__).resolve().parent.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def coefficients(M, a, e):
    r_p = M + np.sqrt(M**2 - a**2)
    r_m = a**2 / (M + (1 - e) * np.sqrt(M**2 - a**2))
    alpha = (a**4 + r_m**3 * r_p - 3 * a**2 * r_m * (r_m + r_p)) / (2 * a**2 * M)
    beta = (a**2 * (2 * M - 3 * r_m - r_p) + r_m**2 * (r_m + 3 * r_p)) / (2 * M)
    gamma = 2 * M - 3 * r_m - r_p
    mu = r_m**3 * r_p / a**2
    return r_p, r_m, alpha, beta, gamma, mu


r, bsym = sp.symbols("r b", real=True)
al_, be_, ga_, mu_, a_, M_ = sp.symbols("alpha beta gamma mu a M", real=True)
m_expr = M_ * (r**2 + al_ * r + be_) / (r**2 + ga_ * r + mu_)


def build(mexpr):
    Delta = r**2 - 2 * mexpr * r + a_**2
    gtt = -(1 - 2 * mexpr / r); gtp = -2 * a_ * mexpr / r; gpp = ((r**2 + a_**2) ** 2 - Delta * a_**2) / r**2; grr = r**2 / Delta
    D = gtp**2 - gtt * gpp
    Vr = (gpp + 2 * bsym * gtp + bsym**2 * gtt) / (D * grr)     # E = 1, L = b
    tdot = (gpp + bsym * gtp) / D
    Vr1 = sp.diff(Vr, r); Vr2 = sp.diff(Vr, r, 2)
    args = (r, bsym, al_, be_, ga_, mu_, a_, M_)
    return [sp.lambdify(args, f, "numpy") for f in (Vr, Vr1, Vr2, tdot, gtt, gtp, gpp)]


FUN = build(m_expr)


def light_ring(M, a, e, sign):
    """sign=+1 prograde (Omega>0), -1 retrograde. Find r_c: Omega_± from the null condition, then radial equilibrium."""
    if e == 0.0:
        r_p, r_m, al, be, ga, mu = coefficients(M, a if a > 0 else 1e-9, 0.0)
        p = (al, be, ga, mu)
    else:
        r_p, r_m, al, be, ga, mu = coefficients(M, a, e)
        p = (al, be, ga, mu)
    Vr, Vr1, Vr2, tdot, gtt, gtp, gpp = FUN

    def Om(rr):
        A, B, C = gpp(rr, 0, *p, a, M), gtp(rr, 0, *p, a, M), gtt(rr, 0, *p, a, M)
        disc = B**2 - A * C
        return (-B + sign * np.sqrt(disc)) / A

    def bal(rr):
        h = 1e-6 * rr
        w = Om(rr)
        f = lambda x: gtt(x, 0, *p, a, M) + 2 * w * gtp(x, 0, *p, a, M) + w**2 * gpp(x, 0, *p, a, M)
        return (f(rr + h) - f(rr - h)) / (2 * h)
    rs = np.linspace(r_p * 1.001, 6 * M, 4000)
    vals = np.array([bal(x) for x in rs])
    idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    rc = brentq(bal, rs[idx[0]], rs[idx[0] + 1], xtol=1e-13)
    w = Om(rc)
    b = -(gtp(rc, 0, *p, a, M) + gpp(rc, 0, *p, a, M) * w) / (gtt(rc, 0, *p, a, M) + gtp(rc, 0, *p, a, M) * w)
    V0, V1, V2, td = Vr(rc, b, *p, a, M), Vr1(rc, b, *p, a, M), Vr2(rc, b, *p, a, M), tdot(rc, b, *p, a, M)
    lam = np.sqrt(max(V2, 0.0) / (2 * td**2))
    return dict(r_c=rc, Omega=w, b=b, lam=lam, V0=V0, V1=V1, kappa_plus=None)


def kerr_ref(M, a, sign):
    rc = 2 * M * (1 + np.cos(2 / 3 * np.arccos(-sign * a / M)))
    Om = sign / (rc**1.5 / np.sqrt(M) + sign * a)
    return rc, Om


def main():
    say("Eikonal QNMs of the rotating inner-extremal regular BH (Franzin et al. 2022) against Kerr; M = 1.")
    # controls
    lr = light_ring(1.0, 1e-6, 0.0, +1)
    say(f"Schwarzschild (a -> 0, e = 0): r_c = {lr['r_c']:.6f}, Omega_c = {lr['Omega']:.6f}, lambda = {lr['lam']:.6f} (expected 3, 0.192450, 0.192450); V = {lr['V0']:.1e}, V' = {lr['V1']:.1e}")
    for a in (0.3, 0.6, 0.9):
        for sgn in (+1, -1):
            lr = light_ring(1.0, a, 0.0, sgn)
            rk, Ok = kerr_ref(1.0, a, sgn)
            say(f"Kerr a={a}, {'prograde' if sgn>0 else 'retrograde'}: r_c = {lr['r_c']:.6f} (analytic {rk:.6f}), Omega = {lr['Omega']:+.6f} (analytic {Ok:+.6f}), lambda = {lr['lam']:.6f}")
    rows = []
    say("\nShifts relative to Kerr at the same (M, a): dOmega/Omega, dlambda/lambda; r_c; e is the inner-horizon displacement parameter (e = 0 is Kerr).")
    for a in (0.3, 0.6, 0.9, 0.99):
        for e in (0.1, 0.3, 0.5, 1.0, 1.5):
            r_p, r_m, al, be, ga, mu = coefficients(1.0, a, e)
            line = f"a={a:4.2f} e={e:3.1f} (r-={r_m:.4f}, alpha-gamma={al-ga:+.2e}): "
            row = dict(a=a, e=e, r_minus=r_m, alpha_minus_gamma=al - ga)
            for sgn, tag in ((+1, "pro"), (-1, "retro")):
                lr = light_ring(1.0, a, e, sgn); lk = light_ring(1.0, a, 0.0, sgn)
                if lr is None or lk is None:
                    line += f"{tag}: no ring; "; continue
                dO = lr["Omega"] / lk["Omega"] - 1; dl = lr["lam"] / lk["lam"] - 1; dr = lr["r_c"] / lk["r_c"] - 1
                row[tag] = dict(r_c=lr["r_c"], Omega=lr["Omega"], lam=lr["lam"], dOmega=dO, dlam=dl, dr=dr)
                line += f"{tag}: r_c {lr['r_c']:.4f} ({dr:+.1e}), dOmega/Omega = {dO:+.2e}, dlambda/lambda = {dl:+.2e}; "
            say(line); rows.append(row)
    # scaling with e at a = 0.6: exponent
    es = np.array([0.1, 0.2, 0.3, 0.5])
    d = np.array([light_ring(1.0, 0.6, e, +1)["Omega"] / light_ring(1.0, 0.6, 0.0, +1)["Omega"] - 1 for e in es])
    k = np.polyfit(np.log(es), np.log(np.abs(d)), 1)[0]
    say(f"\nPower law in e (a = 0.6, prograde, dOmega/Omega): exponent d ln|dOmega|/d ln e = {k:.2f} (companion paper: kappa_+ = Kerr + O(e^3))")
    json.dump(rows, open(OUT / "eikonal_qnm.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
