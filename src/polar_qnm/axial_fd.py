"""Independent check of the QNM band (qnm_band.py) by the frequency-domain method: direct integration of the Regge-Wheeler equation
psi_{r*r*} + (omega^2 - V) psi = 0 with complex omega.
Inner branch: from the horizon (r = r_h (1 + 1e-9), ingoing wave psi = e^{-i omega r*}(1 + c1 x), c1 = V'(r_h)/(4 kappa (kappa - i omega)))
along real r out to r_m, where the profile is already exactly Schwarzschild (|m/M - 1| < 1e-12).
Outer branch: outgoing wave psi = e^{+i omega r*}(1 + i l(l+1)/(2 omega r)) from complex r = r_m + S e^{i theta} along a straight line to r_m -
along such a path (theta > arctan(|Im omega|/Re omega)) the sought solution dominates, the ingoing-wave admixture is suppressed as e^{-2 Im(omega r*)}.
QNM: a zero of the Wronskian W(omega) = psi_in psi_out' - psi_in' psi_out at r_m (secant method in the complex plane, start from the Leaver value).
Control runs: Schwarzschild (Leaver 0.373672 - 0.088962 i, l = 2; 0.599443 - 0.092703 i, l = 3), Hayward ell/M = 0.1 (time domain: dRe/Re = +8.8e-4).
Run: python src/polar_qnm/axial_fd.py
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
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "0.05"]
import qnm_band as qb  # noqa: E402
import axial_td as ax  # noqa: E402
import polar_qnm as pq  # noqa: E402

LOG = []
LEAVER = {2: 0.373672 - 0.088962j, 3: 0.599443 - 0.092703j}


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def f_and_V(bg, r, l):
    r = np.atleast_1d(np.asarray(r, float))
    d = bg.fields(r)
    m = d["m"]
    if "mp" in d:
        mp = d["mp"]
    else:                                   # HaywardBG: m' analytically
        M, ell = bg.M, bg.ell
        mp = 3 * M * r**2 * 2 * M * ell**2 / (r**3 + 2 * M * ell**2) ** 2
    f = 1 - 2 * m / r
    fp = 2 * m / r**2 - 2 * mp / r
    V = f * ((l * (l + 1) - 2 + 2 * f - r * fp) / r**2)
    return f, V


def inner_solution(bg, l, omega, r_m):
    rh = bg.rh
    h = 1e-6
    fh_p = (f_and_V(bg, rh + h, l)[0][0] - f_and_V(bg, rh - h * 0, l)[0][0]) / h  # f'(r_h) (f(r_h) = 0)
    kappa = fh_p / 2
    V1 = (f_and_V(bg, rh + h, l)[1][0]) / h                                      # V'(r_h)
    c1 = V1 / (4 * kappa * (kappa - 1j * omega))
    x0 = 1e-9 * rh
    y0 = [1 + c1 * x0, -1j * omega * (1 + c1 * x0) + 2 * kappa * x0 * c1]

    def rhs(r, y):
        f, V = f_and_V(bg, r, l)
        return [y[1] / f[0], (V[0] - omega**2) * y[0] / f[0]]

    s = solve_ivp(rhs, (rh + x0, r_m), np.array(y0, dtype=complex), rtol=1e-11, atol=1e-30, method="DOP853")
    return s.y[0, -1], s.y[1, -1]


def outer_solution(l, omega, r_m, M=1.0, S=300.0, theta=np.deg2rad(40)):
    def rstar(r):
        return r + 2 * M * np.log(r / (2 * M) - 1)

    def fV(r):
        f = 1 - 2 * M / r
        return f, f * (l * (l + 1) / r**2 - 6 * M / r**3)

    def rhs(t, y):   # r = r_m + (1 - t) S e^{i theta}, t from 0 to 1; dr/dt = -S e^{i theta}
        r = r_m + (1 - t) * S * np.exp(1j * theta)
        f, V = fV(r)
        dr = -S * np.exp(1j * theta)
        return [y[1] / f * dr, (V - omega**2) * y[0] / f * dr]

    r0 = r_m + S * np.exp(1j * theta)
    b1 = 1j * l * (l + 1) / (2 * omega)
    u = 1 + b1 / r0
    # overall factor e^{i omega r*(r0)} (~1e-41 at l = 3) omitted: otherwise the start falls below atol and the solution is lost in noise
    psi0 = u
    dpsi0 = 1j * omega * psi0 + fV(r0)[0] * (-b1 / r0**2)
    s = solve_ivp(rhs, (0.0, 1.0), np.array([psi0, dpsi0], dtype=complex), rtol=1e-11, atol=1e-30, method="DOP853")
    return s.y[0, -1], s.y[1, -1]


def wronskian(bg, l, omega, r_m):
    pi, dpi = inner_solution(bg, l, omega, r_m)
    po, dpo = outer_solution(l, omega, r_m)
    return pi * dpo - dpi * po          # Wronskian: holomorphic in omega, no poles; normalizations psi_in ~ e^{-i omega r*}, psi_out ~ e^{+i omega r*} mutually cancel


def find_qnm(bg, l, omega0, r_m, tol=1e-10):
    w0, w1 = omega0, omega0 * (1 + 1e-3)
    F0, F1 = wronskian(bg, l, w0, r_m), wronskian(bg, l, w1, r_m)
    for _ in range(40):
        w2 = w1 - F1 * (w1 - w0) / (F1 - F0)
        if abs(w2 - w1) < tol:
            break
        w0, F0, w1 = w1, F1, w2
        F1 = wronskian(bg, l, w1, r_m)
    else:
        w2 = w1
    if abs(w2 - omega0) > 0.05:
        say(f"    warning: secant method drifted to {w2:.4f} (start {omega0:.4f}); retrying with a smaller step")
        w0, w1 = omega0, omega0 * (1 + 1e-5)
        F0, F1 = wronskian(bg, l, w0, r_m), wronskian(bg, l, w1, r_m)
        for _ in range(60):
            w2 = w1 - F1 * (w1 - w0) / (F1 - F0)
            if abs(w2 - w1) < tol:
                break
            w0, F0, w1 = w1, F1, w2
            F1 = wronskian(bg, l, w1, r_m)
    return w2


def schw_radius(bg, M=1.0):
    """r_m: smallest r >= 4 from which |m/M - 1| < 1e-12 for all larger r."""
    r = np.linspace(4.0, 40.0, 3601)
    dev = np.abs(bg.fields(r)["m"] / M - 1)
    ok = np.nonzero(dev >= 1e-12)[0]
    return 4.0 if ok.size == 0 else float(min(max(r[ok[-1]] + 0.5, 4.0), 20.0))


def main():
    say("Frequency-domain method (Wronskian, complex contour) - independent check of the QNM band, axial sector")
    bgS = pq.Background(None)
    ref = {}
    for l in (2, 3):
        w = find_qnm(bgS, l, LEAVER[l], 6.0)
        ref[l] = w
        say(f"  Schwarzschild l={l}: {w.real:.6f} {w.imag:+.6f}i (Leaver {LEAVER[l]}): dRe {w.real/LEAVER[l].real-1:+.1e}, dIm {w.imag/LEAVER[l].imag-1:+.1e}")
    # Hayward control run
    for ell in (0.1, 0.05):
        bgH = ax.HaywardBG(ell)
        r_m = schw_radius(bgH)
        w = find_qnm(bgH, 2, ref[2], r_m)
        say(f"  Hayward ell/M = {ell} (r_m = {r_m}): dRe/Re = {w.real/ref[2].real-1:+.3e}, dIm/Im = {w.imag/ref[2].imag-1:+.3e}; on (ell/M)^2: {(w.real/ref[2].real-1)/ell**2:+.3f}, {(w.imag/ref[2].imag-1)/ell**2:+.3f} (time domain: +0.088, -0.162)")
    band = json.load(open(OUT / "qnm_band.json", encoding="utf-8"))
    td = {r["tag"]: r for r in band["rows"]}
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    variants = [("base (scenario)", base0, 1.0), ("u2 = 3", dict(base0, u2=3.0), 1.0), ("u3 = 12", dict(base0, u3=12.0), 1.0),
                ("u3 = 20", dict(base0, u3=20.0), 1.0), ("s_end = 8", dict(base0, s_end=8.0), 1.0), ("s_end = 48", dict(base0, s_end=48.0), 1.0),
                ("sigma_min = 0.5", base0, 0.5), ("w2 = 0.6", dict(base0, w2=0.6), 1.0)]
    rows = []
    say("\n  variant | l | frequency domain: dRe/Re, dIm/Im | time domain: dRe/Re, dIm/Im | discrepancy (abs.)")
    for tag, base, smin in variants:
        fam = qb.build_family(base, smin)
        bg = qb.FamilyBG(fam)
        r_m = schw_radius(bg)
        row = dict(tag=tag, r_m=r_m)
        for l in (2, 3):
            w = find_qnm(bg, l, ref[l], r_m)
            dre, dim = w.real / ref[l].real - 1, w.imag / ref[l].imag - 1
            t_re, t_im = td[tag][f"ax_l{l}_dRe"], td[tag][f"ax_l{l}_dIm"]
            row[f"l{l}"] = dict(omega=[w.real, w.imag], dRe=dre, dIm=dim, td_dRe=t_re, td_dIm=t_im, diff_re=dre - t_re, diff_im=dim - t_im)
            say(f"  {tag:16s} | {l} | {dre:+.2e}, {dim:+.2e} | {t_re:+.2e}, {t_im:+.2e} | {abs(dre-t_re):.1e}, {abs(dim-t_im):.1e}   (r_m = {r_m})")
        rows.append(row)
    dr = [abs(r[f"l{l}"]["diff_re"]) for r in rows for l in (2, 3)]; di = [abs(r[f"l{l}"]["diff_im"]) for r in rows for l in (2, 3)]
    say(f"\nMaximum discrepancy between methods: {max(dr):.1e} in frequency, {max(di):.1e} in damping (absolute relative shifts)")
    sgn = all(np.sign(r[f"l{l}"]["dRe"]) == np.sign(r[f"l{l}"]["td_dRe"]) or abs(r[f"l{l}"]["dRe"]) < 3e-4 for r in rows for l in (2, 3))
    say(f"Frequency shift signs agree (or shift < 3e-4): {sgn}")
    json.dump(dict(ref={l: [ref[l].real, ref[l].imag] for l in ref}, rows=rows), open(OUT / "axial_fd.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "axial_fd_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
