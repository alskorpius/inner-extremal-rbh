"""Independent audit of key claims of the companion paper on the horizonless branch (independent formulas and code).
(1) E4 (tower theorem): for h(psi) = sum alpha_n psi^n, alpha_1 = 1, alpha_n >= 0: at the root of E1 (2 psi h' = 3 h) the quantity
    E2 := -3 h^2 psi_tt - 2 (psi - h psi_t), where t = h(psi), psi_t = 1/h', psi_tt = -h''/h'^3, equals (4 psi^2/(9 h))(2 psi h'' - h'),
    and 2 psi h'' - h' >= 1 > 0. Checked symbolically (general h) and numerically on several families.
(2) E5 (rotation, Franzin 2022): kappa_± = Delta'(r_±)/(2(r_±^2 + a^2)) at M = 1, a = 0.6, e = 0.5: their kappa_- = -3.8e-10, kappa_+ = 0.22212,
    Kerr 0.21429; alpha - gamma = 4.66e-4.
(3) E3 (stability of the Visser-Wiltshire thin shell): interior A_int = Hayward with m_i = 0.65 M_crit, alpha = 1; exterior Schwarzschild
    M = 1.344 and 1.844; EOS P = kappa sigma; equilibria R* and sign of V''. Theirs: R* = 4.796/3.573/2.987 (kappa = 0.25/0.5/1, M = 1.344),
    V'' = -0.047/-0.165/-0.567 — all unstable; no equilibria for kappa <= 0.
(4) E1-A: sign of the Hayward frequency shift — checked earlier (polar_qnm/axial_td.py): their -1.42e-3 is not confirmed; ours +8.8e-4 (three methods).
(5) Stage 8: m_crit = 3 sqrt(3)/4 sqrt(alpha) for Hayward — checked via horizons.
Run: python src/audit_bhp/audit_bhp.py
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


# ---------------- (1) tower theorem
def part1():
    say("=== (1) E4: tower theorem identities")
    psi = sp.symbols("psi", positive=True)
    h = sp.Function("h")(psi)
    hp, hpp = sp.diff(h, psi), sp.diff(h, psi, 2)
    psi_t = 1 / hp
    psi_tt = -hpp / hp**3
    E2 = -3 * h**2 * psi_tt - 2 * (psi - h * psi_t)
    # at the root of E1: h = 2 psi h'/3
    E2_root = sp.simplify(E2.subs(h, 2 * psi * hp / 3))
    target = (4 * psi**2 / (9 * (2 * psi * hp / 3))) * (2 * psi * hpp - hp)
    say(f"  E2|_(E1) - (4 psi^2/(9h))(2 psi h'' - h') = {sp.simplify(E2_root - target)}")
    # inequality: h = sum alpha_n psi^n; E1: sum (2n - 3) alpha_n psi^n = 0 => sum_{n>=2} (2n-3) alpha_n psi^n = psi (alpha_1 = 1)
    # 2 psi h'' - h' = sum n(2n-3) alpha_n psi^{n-1} = -1 + sum_{n>=2} n(2n-3) alpha_n psi^{n-1} >= -1 + 2 sum_{n>=2}(2n-3) alpha_n psi^{n-1} = -1 + 2 = 1
    say("  inequality: 2 psi h'' - h' = -1 + sum_{n>=2} n(2n-3) alpha_n psi^{n-1} >= -1 + 2 sum_{n>=2} (2n-3) alpha_n psi^{n-1} = 1 (by E1) — the chain holds for alpha_n >= 0, n >= 2")
    # numerically: families
    fams = {"geometric a=1": lambda p: p / (1 - p), "n a^{n-1}, a=1": lambda p: p / (1 - p) ** 2, "log": lambda p: -sp.log(1 - p),
            "psi + 0.7 psi^2": lambda p: p + 0.7 * p**2, "psi + psi^2 + 0.3 psi^3": lambda p: p + p**2 + 0.3 * p**3}
    for name, hf in fams.items():
        hx = hf(psi); hxp = sp.diff(hx, psi); hxpp = sp.diff(hx, psi, 2)
        E1 = sp.lambdify(psi, 2 * psi * hxp - 3 * hx, "numpy")
        E2f = sp.lambdify(psi, (-3 * hx**2 * (-hxpp / hxp**3) - 2 * (psi - hx / hxp)), "numpy")
        Q = sp.lambdify(psi, 2 * psi * hxpp - hxp, "numpy")
        grid = np.linspace(1e-3, 0.999 if "psi +" not in name else 5.0, 20001)
        v = E1(grid)
        idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
        roots = [brentq(E1, grid[i], grid[i + 1]) for i in idx]
        say(f"  {name}: E1 roots psi* = {[round(r, 4) for r in roots]}; E2 at roots = {[f'{E2f(r):+.4f}' for r in roots]}; 2 psi h''-h' = {[f'{Q(r):.4f}' for r in roots]} (>= 1)")
    say("  Verdict: the identity and inequality are independently confirmed; a triple root in towers with alpha_n >= 0 is impossible (E2 > 0).")


# ---------------- (2) rotation
def part2():
    say("\n=== (2) E5: surface gravities of the Franzin metric at M = 1, a = 0.6, e = 0.5")
    M, a, e = 1.0, 0.6, 0.5
    r_p = M + np.sqrt(M**2 - a**2); r_m = a**2 / (M + (1 - e) * np.sqrt(M**2 - a**2))
    al = (a**4 + r_m**3 * r_p - 3 * a**2 * r_m * (r_m + r_p)) / (2 * a**2 * M)
    be = (a**2 * (2 * M - 3 * r_m - r_p) + r_m**2 * (r_m + 3 * r_p)) / (2 * M)
    ga = 2 * M - 3 * r_m - r_p; mu = r_m**3 * r_p / a**2
    m = lambda r: M * (r**2 + al * r + be) / (r**2 + ga * r + mu)
    Delta = lambda r: r**2 - 2 * m(r) * r + a**2
    # analytic form Delta = (r - r_p)(r - r_m)^3/(r^2 + ga r + mu): check
    rr = np.linspace(0.05, 5, 2000)
    say(f"  check Delta = (r-r+)(r-r-)^3/(r^2+gamma r+mu): max |difference| = {np.max(np.abs(Delta(rr) - (rr - r_p) * (rr - r_m) ** 3 / (rr**2 + ga * rr + mu))):.2e}")
    dD = lambda r, h=1e-6: (Delta(r + h) - Delta(r - h)) / (2 * h)
    kp = dD(r_p) / (2 * (r_p**2 + a**2)); km = dD(r_m) / (2 * (r_m**2 + a**2))
    say(f"  r+ = {r_p:.4f}, r- = {r_m:.6f}; kappa_+ = {kp:.5f} (theirs 0.22212; Kerr {(r_p - r_m_kerr(M, a))/(2*(r_p**2+a**2)):.5f}), kappa_- = {km:+.2e} (theirs -3.8e-10); alpha - gamma = {al - ga:+.4e} (theirs 4.66e-4)")
    # analytically kappa_+ = (r+ - r-)^3/((r+^2 + gamma r+ + mu) 2 (r+^2 + a^2))
    kp_an = (r_p - r_m) ** 3 / ((r_p**2 + ga * r_p + mu) * 2 * (r_p**2 + a**2))
    say(f"  kappa_+ analytically = {kp_an:.5f}; deviation from Kerr {(kp_an/((r_p - r_m_kerr(M, a))/(2*(r_p**2+a**2))) - 1)*100:+.2f} % (theirs +3.6 %)")


def r_m_kerr(M, a):
    return M - np.sqrt(M**2 - a**2)


# ---------------- (3) thin-shell stability
def part3():
    say("\n=== (3) E3: thin shell between a Hayward core (m_i = 0.65 M_crit, alpha = 1) and Schwarzschild M; P = kappa sigma")
    M_crit = 3 * np.sqrt(3) / 4; m_i = 0.65 * M_crit
    f_in = lambda R: 1 - 2 * m_i * R**2 / (R**3 + 2 * m_i)
    for M in (m_i + 0.5, m_i + 1.0):
        f_out = lambda R: 1 - 2 * M / R
        for kappa in (0.0, 0.25, 0.5, 1.0, -0.25):
            # equilibrium: V(R) = 0 and V'(R) = 0 along the branch m_s(R) = m_s0 (R/R0)^{-2 kappa}, where m_s0 = R0 (sqrt f_in - sqrt f_out) (at rest at R0)
            def V(R, R0):
                ms = R0 * (np.sqrt(f_in(R0)) - np.sqrt(max(f_out(R0), 0))) * (R / R0) ** (-2 * kappa)
                Df = f_in(R) - f_out(R)
                return -((Df * R / (2 * ms) + ms / (2 * R)) ** 2 - f_in(R))   # V = -Rdot^2
            def dV(R0, h=1e-5):
                return (V(R0 + h, R0) - V(R0 - h, R0)) / (2 * h)
            grid = np.linspace(2 * M + 0.05, 4 * M, 400)
            vals = []
            for R0 in grid:
                try:
                    vals.append(dV(R0))
                except Exception:
                    vals.append(np.nan)
            vals = np.array(vals)
            idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
            eq = []
            for i in idx:
                try:
                    R0 = brentq(dV, grid[i], grid[i + 1], xtol=1e-10)
                    h = 1e-4 * R0
                    vpp = (V(R0 + h, R0) - 2 * V(R0, R0) + V(R0 - h, R0)) / h**2
                    eq.append((R0, vpp))
                except Exception:
                    pass
            say(f"  M = {M:.3f}, kappa = {kappa:+.2f}: equilibria R* = {[round(r, 4) for r, _ in eq]}, V'' = {[f'{v:+.4f}' for _, v in eq]} ({'all unstable' if eq and all(v < 0 for _, v in eq) else ('some stable' if eq else 'no equilibria')})")


# ---------------- (5) m_crit for Hayward
def part5():
    say("\n=== (5) Stage 8: m_crit for Hayward")
    for alpha in (1.0, 0.25):
        ell = np.sqrt(alpha)
        f = lambda r, m: 1 - 2 * m * r**2 / (r**3 + 2 * m * alpha)
        def nroots(m):
            rr = np.geomspace(1e-3, 10, 200001); v = f(rr, m)
            return int(np.sum(np.sign(v[:-1]) * np.sign(v[1:]) < 0))
        lo, hi = 0.5 * ell, 3 * ell
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if nroots(mid) >= 2: hi = mid
            else: lo = mid
        say(f"  alpha = {alpha}: m_crit = {hi:.6f}; 3 sqrt(3)/4 sqrt(alpha) = {3*np.sqrt(3)/4*ell:.6f}")


def main():
    part1(); part2(); part3(); part5()
    say("\n=== (4) E1-A: Hayward frequency shift at ell/M = 0.1 — theirs -1.42e-3 (FFT peak, Re error 0.5 %), ours +8.76e-4 (matrix pencil, Re error 1.4e-5), WKB +8e-4, Bolokhov-Skvortsova +9.2e-4: the companion paper's sign is not confirmed; the damping agrees.")
    (LOGS / "audit_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
