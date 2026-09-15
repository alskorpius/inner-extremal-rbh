"""Rerun of the frequency-domain method for variants where the secant method starting from Schwarzschild drifted to an overtone: start
from the time-domain estimate, horizon start closer in (x0 = 1e-7 r_h, c1 correction), secant step 1e-5. Run: python src/polar_qnm/axial_fd_fix.py u3=20 [more tags]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
tags = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))
import axial_fd as fd  # noqa: E402
import qnm_band as qb  # noqa: E402


def inner_solution(bg, l, omega, r_m, x0rel=1e-7):
    rh = bg.rh
    h = 1e-6
    fh_p = fd.f_and_V(bg, rh + h, l)[0][0] / h
    kappa = fh_p / 2
    V1 = fd.f_and_V(bg, rh + h, l)[1][0] / h
    c1 = V1 / (4 * kappa * (kappa - 1j * omega))
    x0 = x0rel * rh
    y0 = [1 + c1 * x0, -1j * omega * (1 + c1 * x0) + 2 * kappa * x0 * c1]

    def rhs(r, y):
        f, V = fd.f_and_V(bg, r, l)
        return [y[1] / f[0], (V[0] - omega**2) * y[0] / f[0]]

    s = solve_ivp(rhs, (rh + x0, r_m), np.array(y0, dtype=complex), rtol=1e-11, atol=1e-30, method="DOP853")
    return s.y[0, -1], s.y[1, -1]


def W(bg, l, omega, r_m):
    pi, dpi = inner_solution(bg, l, omega, r_m)
    po, dpo = fd.outer_solution(l, omega, r_m)
    return pi * dpo - dpi * po


def find(bg, l, omega0, r_m):
    w0, w1 = omega0, omega0 * (1 + 1e-5)
    F0, F1 = W(bg, l, w0, r_m), W(bg, l, w1, r_m)
    for _ in range(60):
        w2 = w1 - F1 * (w1 - w0) / (F1 - F0)
        if abs(w2 - w1) < 1e-10:
            return w2
        w0, F0, w1 = w1, F1, w2
        F1 = W(bg, l, w1, r_m)
    return w1


def main():
    band = json.load(open(OUT / "qnm_band.json", encoding="utf-8"))
    td = {r["tag"]: r for r in band["rows"]}
    refS = {2: 0.373672 - 0.088962j, 3: 0.599443 - 0.092703j}
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    allv = {"base (scenario)": (base0, 1.0), "u2=3": (dict(base0, u2=3.0), 1.0), "u3=12": (dict(base0, u3=12.0), 1.0), "u3=20": (dict(base0, u3=20.0), 1.0),
            "s_end=8": (dict(base0, s_end=8.0), 1.0), "s_end=48": (dict(base0, s_end=48.0), 1.0), "sigma_min=0.5": (base0, 0.5), "w2=0.6": (dict(base0, w2=0.6), 1.0)}
    tdkey = {"u3=20": "u3 = 20", "u2=3": "u2 = 3", "u3=12": "u3 = 12", "s_end=8": "s_end = 8", "s_end=48": "s_end = 48", "sigma_min=0.5": "sigma_min = 0.5", "w2=0.6": "w2 = 0.6", "base (scenario)": "base (scenario)"}
    out = {}
    for tag in tags:
        base, smin = allv[tag]
        bg = qb.FamilyBG(qb.build_family(base, smin))
        r_m = fd.schw_radius(bg)
        out[tag] = {}
        for l in (2, 3):
            t = td[tdkey[tag]]
            w0 = complex(refS[l].real * (1 + t[f"ax_l{l}_dRe"]), refS[l].imag * (1 + t[f"ax_l{l}_dIm"]))
            w = find(bg, l, w0, r_m)
            dre, dim = w.real / refS[l].real - 1, w.imag / refS[l].imag - 1
            print(f"{tag} l={l}: frequency domain dRe/Re = {dre:+.3e}, dIm/Im = {dim:+.3e}; time domain {t[f'ax_l{l}_dRe']:+.3e}, {t[f'ax_l{l}_dIm']:+.3e}; discrepancy {abs(dre-t[f'ax_l{l}_dRe']):.1e}, {abs(dim-t[f'ax_l{l}_dIm']):.1e} (start from TD, r_m = {r_m})", flush=True)
            out[tag][f"l{l}"] = dict(omega=[w.real, w.imag], dRe=dre, dIm=dim, td_dRe=t[f"ax_l{l}_dRe"], td_dIm=t[f"ax_l{l}_dIm"])
    json.dump(out, open(OUT / "axial_fd_fix.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
