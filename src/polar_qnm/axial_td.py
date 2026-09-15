"""Axial QNMs in the time domain by the same method (leapfrog + matrix pencil) as the polar sector: resolving the tension
between our WKB result (+0.08 (ell/M)^2 in frequency for Hayward; Bolokhov-Skvortsova 2025 gives the same sign) and the companion paper's claim E1-A (time domain,
FFT peak: dRe = -1.42e-3 at ell/M = 0.1). Potential V = f[(l(l+1) - 2 + 2f - r f')/r^2] (cross-checked by both projects).
Run: python src/polar_qnm/axial_td.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "0.05"]
import polar_qnm as pq  # noqa: E402
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class HaywardBG:
    def __init__(self, ell, M=1.0):
        self.ell, self.M, self.scen = ell, M, "hayward"
        rg = np.geomspace(0.01, 4.0, 200001)
        f = 1 - 2 * self.m(rg) / rg
        i = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0][-1]
        from scipy.optimize import brentq
        self.rh = brentq(lambda r: 1 - 2 * self.m(r) / r, rg[i], rg[i + 1], xtol=1e-14)

    def m(self, r):
        return self.M * r**3 / (r**3 + 2 * self.M * self.ell**2)

    def fields(self, r):
        return dict(m=self.m(r))


def axial_potential(bg, r, l):
    m = bg.fields(r)["m"]
    h = 1e-6 * r
    mp = (bg.fields(r + h)["m"] - bg.fields(r - h)["m"]) / (2 * h)
    f = 1 - 2 * m / r
    fp = 2 * m / r**2 - 2 * mp / r
    return f * ((l * (l + 1) - 2 + 2 * f - r * fp) / r**2)


def evolve_axial(bg, l, drs=0.05, rs_min=-80.0, rs_max=400.0, t_end=320.0, r_obs=40.0, sigma0=1.5, r0=20.0):
    grid, r = pq.tortoise_grid(bg, rs_min, rs_max, drs)
    U = axial_potential(bg, r, l)
    dt = 0.5 * drs
    nsteps = int(t_end / dt)
    iobs = int(np.argmin(np.abs(grid - r_obs)))
    psi = np.exp(-((grid - r0) ** 2) / (2 * sigma0**2)); psi_prev = psi.copy()
    ts, ys = [], []
    for n in range(nsteps):
        lap = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / drs**2
        psi_new = 2 * psi - psi_prev + dt**2 * (lap - U * psi)
        psi_new[0] = psi_new[-1] = 0.0
        psi_prev, psi = psi, psi_new
        ts.append((n + 1) * dt); ys.append(psi[iobs])
    return np.array(ts), np.array(ys)


def main():
    say("Axial QNMs, time domain + matrix pencil (window [80,200]), l = 2, Δr* = 0.05")
    known = (0.37367, -0.08896)
    bgs = [("Schwarzschild", pq.Background(None))]
    for ell in (0.1, 0.05, 0.03):
        bgs.append((f"Hayward ell/M = {ell}", HaywardBG(ell)))
    bgs.append(("Scenario (branch A)", pq.Background(lb.ScenarioMass())))
    rows = []
    wS = None
    for name, bg in bgs:
        t, y = evolve_axial(bg, 2)
        np.savez_compressed(OUT / f"axial_series_{name.split()[0]}_{getattr(bg, 'ell', 'A')}.npz", t=t, y=y)
        wr, wi = pq.matrix_pencil(t, y, 80.0, 200.0)
        if wS is None:
            wS = (wr, wi)
            say(f"  {name}: {wr:.6f} {wi:+.6f}i (Leaver {known[0]} {known[1]:+}i: dRe {wr/known[0]-1:+.1e}, dIm {wi/known[1]-1:+.1e})")
        else:
            dre, dim = wr - wS[0], wi - wS[1]
            extra = ""
            if isinstance(bg, HaywardBG):
                extra = f"; on (ell/M)^2: dRe/Re = {(wr/wS[0]-1)/bg.ell**2:+.3f}, dIm/Im = {(wi/wS[1]-1)/bg.ell**2:+.3f} (our WKB: +0.08, -0.17; Bolokhov-Skvortsova: +0.092, -0.17)"
            say(f"  {name}: {wr:.6f} {wi:+.6f}i; ΔRe = {dre:+.2e}, ΔIm = {dim:+.2e}; dRe/Re = {wr/wS[0]-1:+.2e}, dIm/Im = {wi/wS[1]-1:+.2e}{extra}")
        rows.append(dict(name=name, omega=[wr, wi]))
    json.dump(rows, open(OUT / "axial_td.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    (LOGS / "axial_td_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
