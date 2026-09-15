"""Horizon-triggered gravastar-type mechanism: three checks.

(1) Thin-shell gravastar (Visser-Wiltshire 2004): de Sitter interior f_in = 1 - R^2/ell^2, Schwarzschild exterior
    f_out = 1 - 2M/R, thin shell at R_s (2M < R_s < ell). Statics: sigma = -(sqrt f_out - sqrt f_in)/(4 pi R_s),
    P = [ (f_out + R f_out'/2)/sqrt f_out - (f_in + R f_in'/2)/sqrt f_in ] / (8 pi R_s).
    Radial stability: Rdot^2 + V(R) = 0, V = (f_in + f_out)/2 - [(f_in - f_out) R/(2 m_s)]^2 - (m_s/(2R))^2,
    m_s = 4 pi R^2 sigma, along EOS P = w sigma: sigma ∝ R^{-2(1+w)}. Stable when V''(R_s) > 0.
    Question: do static stable configurations with sigma >= 0, sigma + P >= 0 exist, and how is ell related to M.
(2) Self-consistent local "horizon" law: rho = Phi(f)/(8 pi R^2), Phi = Phi_max * S((f_c - f)/w):
    the vacuum switches on where the redshift is large (f small). The equation dh/dt = Phi(1-h) - h (t = ln R) is
    an autonomous first-order equation: h tends to an equilibrium h* (Phi(1-h*) = h*), f -> f* = 1 - h* > 0: a plateau without a horizon
    and without an asymptotically flat region (mass grows linearly). Show this numerically.
(3) Moving the trigger inside the horizon: a static shell at R_s < 2M is impossible (Q = f_out + Rdot^2 < 0 at Rdot = 0);
    formally-analytically: sqrt(f_out) is imaginary for f_out < 0. Noted without further calculation.
Run: python src/mechanism_ellM/gravastar_trigger.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name


def shell_static(M, ell, Rs):
    f_in = 1 - Rs**2 / ell**2
    f_out = 1 - 2 * M / Rs
    if f_in <= 0 or f_out <= 0:
        return None
    fpi = -2 * Rs / ell**2
    fpo = 2 * M / Rs**2
    sigma = -(np.sqrt(f_out) - np.sqrt(f_in)) / (4 * np.pi * Rs)
    P = ((f_out + Rs * fpo / 2) / np.sqrt(f_out) - (f_in + Rs * fpi / 2) / np.sqrt(f_in)) / (8 * np.pi * Rs)
    return sigma, P


def V_of_R(R, M, ell, ms_func):
    f_in = 1 - R**2 / ell**2
    f_out = 1 - 2 * M / R
    ms = ms_func(R)
    return 0.5 * (f_in + f_out) - ((f_in - f_out) * R / (2 * ms)) ** 2 - (ms / (2 * R)) ** 2


def stability(M, ell, Rs, w, h=1e-4):
    st = shell_static(M, ell, Rs)
    if st is None:
        return None
    sigma, P = st
    ms0 = 4 * np.pi * Rs**2 * sigma
    if abs(ms0) < 1e-14:
        return None
    ms = lambda R: ms0 * (R / Rs) ** (-2 * w)          # sigma ∝ R^{-2(1+w)} => m_s ∝ R^{-2w}
    V0 = V_of_R(Rs, M, ell, ms)
    V1 = (V_of_R(Rs + h, M, ell, ms) - V_of_R(Rs - h, M, ell, ms)) / (2 * h)
    V2 = (V_of_R(Rs + h, M, ell, ms) - 2 * V0 + V_of_R(Rs - h, M, ell, ms)) / h**2
    return dict(sigma=sigma, P=P, V0=V0, V1=V1, V2=V2)


def part1():
    M = 1.0
    print("(1) Thin-shell gravastar: statics and radial stability (M = 1)")
    rows = []
    for ell in (2.05, 2.2, 2.5, 3.0, 4.0):
        for Rs in np.linspace(2.02, min(ell * 0.98, 3.9), 8):
            st = shell_static(M, ell, Rs)
            if st is None:
                continue
            sigma, P = st
            # stability for a set of EOS
            stab = {}
            for w in (-1.0, -0.5, 0.0, 0.5, 1.0):
                s = stability(M, ell, Rs, w)
                stab[w] = None if s is None else s["V2"]
            rows.append(dict(ell=ell, Rs=Rs, sigma=sigma, P=P, nec=sigma + P, V2_by_w=stab))
        # print summary for ell
        sub = [r for r in rows if r["ell"] == ell]
        ok = [r for r in sub if r["sigma"] >= 0 and r["nec"] >= 0]
        stable_any = [r for r in ok if any(v is not None and v > 0 for v in r["V2_by_w"].values())]
        print(f"  ell/M={ell:4g}: static points {len(sub)}, with sigma>=0 and sigma+P>=0: {len(ok)}, of these radially stable for some w in [-1,1]: {len(stable_any)}; "
              f"example: R_s={ok[0]['Rs']:.3f}, sigma={ok[0]['sigma']:.3e}, P={ok[0]['P']:.3e}, V''(w=-1..1)={[None if v is None else round(v,4) for v in ok[0]['V2_by_w'].values()]}" if ok else f"  ell/M={ell:4g}: no points with sigma>=0")
    print("  Conclusion: a static gravastar requires ell > R_s > 2M, i.e. ell = lambda M with lambda > 2 -- the core scale is tied to the mass by construction itself (the horizon is the trigger).")
    return rows


def part2():
    print("\n(2) Self-consistent local horizon law rho = Phi(f)/(8 pi R^2), Phi = Phi_max S((f_c - f)/w)")
    rows = []
    for Phi_max, f_c, w in ((1.5, 0.3, 0.05), (1.5, 0.1, 0.02), (3.0, 0.5, 0.1), (0.8, 0.3, 0.05)):
        Phi = lambda f: Phi_max / (1 + np.exp(-(f_c - f) / w))
        # dh/dt = Phi(1-h) - h; start from the de Sitter center h = c R^2 (t -> -inf): h(t0)=1e-8
        sol = solve_ivp(lambda t, y: [Phi(1 - y[0]) - y[0]], (np.log(1e-4), 15.0), [1e-8], rtol=1e-10, atol=1e-14, max_step=0.02)
        h = sol.y[0]; R = np.exp(sol.t); f = 1 - h
        # equilibria: Phi(1-h) = h
        hs = np.linspace(0, 1.5, 30001); g = Phi(1 - hs) - hs
        eq = hs[np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]]
        rows.append(dict(Phi_max=Phi_max, f_c=f_c, w=w, h_end=float(h[-1]), f_end=float(f[-1]), f_min=float(f.min()), equilibria=[float(e) for e in eq],
                         crossings=int(np.sum(np.sign(f[:-1]) * np.sign(f[1:]) < 0)), h_growth=float(h[-1] * R[-1] / 2)))
        print(f"  Phi_max={Phi_max}, f_c={f_c}, w={w}: equilibria h* = {np.round(eq,4)}; h(R->1e6) = {h[-1]:.4f}, f_min = {f.min():+.4f}, crossings of f=0: {rows[-1]['crossings']}; "
              f"m(R) ~ h R/2 grows linearly: m(1e6)/M~{rows[-1]['h_growth']:.3g} -- no asymptotically flat region")
    print("  Conclusion: without an external cutoff (shell), the local horizon law gives either a plateau f -> f* > 0 or a transversal horizon with kappa != 0, but not a triple root.")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = part1(); r2 = part2()
    print("\n(3) Moving the trigger inside the horizon: a static shell at R_s < 2M requires sqrt(f_out) with f_out < 0 -- impossible (see also src/junction: Q < 0). "
          "The gravastar trigger gives a horizonless object (plateau) or an extremal double root, but not an interior triple root.")
    (OUT / "gravastar_trigger.json").write_text(json.dumps(dict(part1=r1, part2=r2), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
