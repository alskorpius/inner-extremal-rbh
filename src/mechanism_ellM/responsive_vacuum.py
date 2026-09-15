"""Mechanism for ell ∝ M: local scale-free vacuum-response laws and the impossibility of a triple root.

Class g_tt g_rr = -1: f = 1 - h, h = 2m/R, m' = 4 pi R^2 rho. A scale-free (no fundamental length) local
response law: rho = Phi(h, R h', R^2 h'', ...)/(8 pi R^2). Then in the variable t = ln R the equation is autonomous:
    m' = (h + R h')/2 = Phi/2   =>   dh/dt = Phi(h, dh/dt, ...) - h.
A triple root of f (kappa_- = 0 at merged horizons) requires h = 1, dh/dt = 0, d^2h/dt^2 = 0.
Claim: for an autonomous law of order <= 2 in t, the point (h, h', h'') = (1, 0, 0) is an equilibrium
(Phi(1,0,0) = 1), which the solution cannot cross in finite t; hence there is no triple root at finite R:
the solution either asymptotically approaches f = 0 (a plateau, an extremal-like horizonless object) or
never reaches h = 1. Below is a numerical illustration for a second-order law
    beta R^2 f'' + R f' ... equivalent to  beta d^2h/dt^2 + (1 - beta) dh/dt + h - Phi0(h) = 0
(a damped oscillator in the potential U(h) = int (h - Phi0) dh) with Phi0(1) = 1, starting from the de Sitter center.
Run: python src/mechanism_ellM/responsive_vacuum.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name


def run(beta, a, b, h0=1e-6, t_end=12.0):
    """Phi0(h) = a h + b h^2 + (1 - a - b) h^3 => Phi0(0) = 0 (a flat asymptotic h -> 0 is allowed), Phi0(1) = 1.
    Start: de Sitter center h = c R^2 => dh/dt = 2h as t -> -inf."""
    Phi0 = lambda h: a * h + b * h**2 + (1 - a - b) * h**3

    def rhs(t, y):
        h, hd = y
        return [hd, (-(1 - beta) * hd - h + Phi0(h)) / beta]
    t0 = np.log(np.sqrt(h0))  # h = R^2 => t = ln R
    sol = solve_ivp(rhs, (t0, t_end), [h0, 2 * h0], rtol=1e-10, atol=1e-13, max_step=0.01)
    t, h, hd = sol.t, sol.y[0], sol.y[1]
    R = np.exp(t)
    f = 1 - h
    crossings = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    return dict(beta=beta, a=a, b=b, h_max=float(h.max()), R_at_hmax=float(R[np.argmax(h)]), f_min=float(f.min()),
                n_crossings=int(crossings.size), h_end=float(h[-1]), hd_end=float(hd[-1]),
                approaches_1=bool(abs(h[-1] - 1) < 1e-3 and abs(hd[-1]) < 1e-3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Local scale-free second-order law: beta h'' + (1-beta) h' + h - Phi0(h) = 0 (prime denotes d/d ln R), Phi0(1) = 1")
    for beta in (0.05, 0.2, 0.5, 1.0, 2.0):
        for (a, b) in ((0.5, 0.5), (0.2, 1.2), (0.0, 0.5), (0.8, 0.8), (-0.5, 2.0), (1.5, -1.0)):
            r = run(beta, a, b)
            rows.append(r)
            print(f"  beta={beta:4g} Phi0=({a:+.1f}h {b:+.1f}h^2 {1-a-b:+.1f}h^3): h_max={r['h_max']:.4f} at R={r['R_at_hmax']:.3g}; f_min={r['f_min']:+.4f}; "
                  f"crossings of f=0: {r['n_crossings']}; h(end)={r['h_end']:.4f}, h'={r['hd_end']:+.2e}; approach to h=1: {r['approaches_1']}")
    n_cross = sum(r["n_crossings"] for r in rows)
    print(f"\nSummary: total number of crossings f = 0 over all {len(rows)} runs = {n_cross} (theorem expectation: 0, since a transversal crossing is impossible, "
          f"h = 1 is an equilibrium); trajectories either never reach h = 1 or approach it asymptotically (f -> 0+ plateau).")
    (OUT / "responsive_vacuum.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
