"""Prediction band: QNM shifts (axial - Regge-Wheeler potential with m(r); polar - Moreno-Sarbach system with the NED matter
model) for the full family of triple-root profiles from conversion_law/lambda_scan.py (8 variants, lambda = ell/M in [0.13, 0.30]).
Method same as in axial_td.py / polar_qnm.py: time evolution (leapfrog, dr* = 0.05), matrix pencil in the window t in [80, 200],
shift = difference from Schwarzschild in the same code (control: Leaver). For each variant, also: lambda, R-, R+, kappa+, T_H/T_Schw,
m(R+)/M and the maximum |N V11(variant) - N V11(Schw.)| outside the horizon (a measure of the potential's deviation).
Run: python src/polar_qnm/qnm_band.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "stability"))
sys.argv = [sys.argv[0], "0.05"]
import polar_qnm as pq  # noqa: E402
import axial_td as ax  # noqa: E402
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class FamilyBG:
    """Background from MonotoneFamily: m, m', m'', sigma, sigma' for polar_qnm.potential_matrix and axial_td.axial_potential."""

    def __init__(self, fam, M=1.0):
        self.fam, self.M, self.scen = fam, M, "family"
        self._sigp = np.gradient(fam.sig, trm.LU)
        self.rh = brentq(lambda r: 1 - 2 * self._m(r) / r, 1.0, 3.0, xtol=1e-14)
        self.rminus = brentq(lambda r: 1 - 2 * self._m(r) / r - 1e-9, 1e-3, 1.0, xtol=1e-14) if False else None

    def _m(self, r):
        return np.interp(np.asarray(r, float), self.fam._Rg, self.fam._mg)

    def fields(self, r):
        r = np.asarray(r, float)
        m = self._m(r)
        rho = self.fam.rho(r); drho = self.fam.drho(r)
        lu = np.log(r / self.fam.R1)
        sig = np.interp(lu, trm.LU, self.fam.sig)
        sigp = np.interp(lu, trm.LU, self._sigp)
        mp = 4 * np.pi * r**2 * rho
        mpp = 8 * np.pi * r * rho + 4 * np.pi * r**2 * drho
        return dict(m=m, mp=mp, mpp=mpp, sigma=sig, sigp=sigp)


def build_family(base, sigma_min):
    s1s, u_s, info = trm.find_merge_first(sigma_min=sigma_min, base=base)
    if s1s is None:
        return None
    kw = info["kw"]; d = s1s - sigma_min
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    return trm.MonotoneFamily(d, scale, kw=kw)


def horizons(bg):
    r = np.geomspace(1e-2, 4.0, 400001)
    f = 1 - 2 * bg.fields(r)["m"] / r
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = [brentq(lambda x: 1 - 2 * bg.fields(np.array([x]))["m"][0] / x, r[i], r[i + 1], xtol=1e-13) for i in idx]
    # triple root: f touches zero - look for the minimum of |f| inside
    j = np.argmin(np.abs(f[r < 1.5]))
    return roots, float(r[j]), float(f[j])


def kappa_at(bg, rh):
    h = 1e-6
    fp = ((1 - 2 * bg.fields(np.array([rh + h]))["m"][0] / (rh + h)) - (1 - 2 * bg.fields(np.array([rh - h]))["m"][0] / (rh - h))) / (2 * h)
    return fp / 2


def main():
    t0 = time.time()
    say("QNM prediction band for the family of triple-root profiles (dr* = 0.05, window [80, 200], l = 2, 3)")
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    variants = [("base (scenario)", base0, 1.0), ("u2 = 3", dict(base0, u2=3.0), 1.0), ("u3 = 12", dict(base0, u3=12.0), 1.0),
                ("u3 = 20", dict(base0, u3=20.0), 1.0), ("s_end = 8", dict(base0, s_end=8.0), 1.0), ("s_end = 48", dict(base0, s_end=48.0), 1.0),
                ("sigma_min = 0.5", base0, 0.5), ("w2 = 0.6", dict(base0, w2=0.6), 1.0)]
    known = {2: (0.37367, -0.08896), 3: (0.59944, -0.09270)}
    bgS = pq.Background(None)
    ref = {}
    for l in (2, 3):
        t, y = ax.evolve_axial(bgS, l); ref[("ax", l)] = pq.matrix_pencil(t, y, 80.0, 200.0)
        t, y, p, _, _ = pq.evolve(bgS, l); ref[("po", l)] = pq.matrix_pencil(t, y, 80.0, 200.0)
        say(f"  Schwarzschild l={l}: axial {ref[('ax', l)][0]:.6f} {ref[('ax', l)][1]:+.6f}i, polar {ref[('po', l)][0]:.6f} {ref[('po', l)][1]:+.6f}i "
            f"(Leaver {known[l][0]} {known[l][1]:+}i; error Re {ref[('ax', l)][0]/known[l][0]-1:+.1e}/{ref[('po', l)][0]/known[l][0]-1:+.1e}, Im {ref[('ax', l)][1]/known[l][1]-1:+.1e}/{ref[('po', l)][1]/known[l][1]-1:+.1e})")
    rows = []
    for tag, base, smin in variants:
        fam = build_family(base, smin)
        if fam is None:
            say(f"  {tag}: triple root not found"); continue
        bg = FamilyBG(fam)
        roots, r_touch, f_touch = horizons(bg)
        lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
        kp = kappa_at(bg, bg.rh)
        m_plus = float(bg.fields(np.array([bg.rh]))["m"][0])
        # deviation of the potentials outside the horizon
        rr = np.linspace(bg.rh * 1.001, 60, 6000)
        N, V11, V12, V22, info = pq.potential_matrix(bg, rr, 2)
        NS, V11S, _, _, _ = pq.potential_matrix(bgS, rr, 2)
        dV = np.abs(N * V11 - NS * V11S); iV = int(np.argmax(dV))
        Ua = ax.axial_potential(bg, rr, 2); UaS = ax.axial_potential(bgS, rr, 2)
        dVa = np.abs(Ua - UaS); iA = int(np.argmax(dVa))
        row = dict(tag=tag, lam=lam, R_minus=r_touch, f_touch=f_touch, R_plus=bg.rh, kappa_plus=kp, T_ratio=4 * kp, m_plus=m_plus,
                   dV_polar_max=float(dV[iV]), r_dV_polar=float(rr[iV]), dV_axial_max=float(dVa[iA]), r_dV_axial=float(rr[iA]),
                   kappa_MS_min_out=float(info["kappa"].min()), kappa_MS_max_out=float(info["kappa"].max()))
        for l in (2, 3):
            t, y = ax.evolve_axial(bg, l); wa = pq.matrix_pencil(t, y, 80.0, 200.0)
            t, y, p, _, _ = pq.evolve(bg, l); wp = pq.matrix_pencil(t, y, 80.0, 200.0)
            row[f"ax_l{l}"] = list(wa); row[f"po_l{l}"] = list(wp)
            row[f"ax_l{l}_dRe"] = wa[0] / ref[("ax", l)][0] - 1; row[f"ax_l{l}_dIm"] = wa[1] / ref[("ax", l)][1] - 1
            row[f"po_l{l}_dRe"] = wp[0] / ref[("po", l)][0] - 1; row[f"po_l{l}_dIm"] = wp[1] / ref[("po", l)][1] - 1
            row[f"po_l{l}_phi_ratio"] = float(np.max(np.abs(p)) / np.max(np.abs(y)))
        say(f"  {tag}: lambda = {lam:.3f}, R- = {r_touch:.3f} (|f| = {abs(f_touch):.1e}), R+ = {bg.rh:.4f}, kappa+ = {kp:.5f}, T_H/T_S = {4*kp:.4f}, m(R+)/M = {m_plus:.4f}; "
            f"max|dU| axial {row['dV_axial_max']:.1e} at r = {row['r_dV_axial']:.2f}, polar {row['dV_polar_max']:.1e} at r = {row['r_dV_polar']:.2f}; "
            f"kappa_MS outside the horizon [{row['kappa_MS_min_out']:.2f}, {row['kappa_MS_max_out']:.2f}]")
        for l in (2, 3):
            say(f"      l={l}: axial dRe/Re = {row[f'ax_l{l}_dRe']:+.2e}, dIm/Im = {row[f'ax_l{l}_dIm']:+.2e}; polar dRe/Re = {row[f'po_l{l}_dRe']:+.2e}, dIm/Im = {row[f'po_l{l}_dIm']:+.2e} (|Phi|/|Psi| = {row[f'po_l{l}_phi_ratio']:.1e})   [{time.time()-t0:.0f} s]")
        rows.append(row)
    say("\nBand (across 8 variants):")
    for key, name in (("ax_l2_dRe", "axial l=2, frequency"), ("ax_l2_dIm", "axial l=2, damping"), ("ax_l3_dRe", "axial l=3, frequency"), ("ax_l3_dIm", "axial l=3, damping"),
                      ("po_l2_dRe", "polar l=2, frequency"), ("po_l2_dIm", "polar l=2, damping"), ("po_l3_dRe", "polar l=3, frequency"), ("po_l3_dIm", "polar l=3, damping")):
        v = np.array([r[key] for r in rows])
        say(f"  {name}: from {v.min():+.2e} to {v.max():+.2e} (|max| = {np.abs(v).max():.2e})")
    # correlation with T_H and with 1 - m(R+)/M
    T = np.array([r["T_ratio"] for r in rows]); dm = np.array([1 - r["m_plus"] for r in rows]); lamv = np.array([r["lam"] for r in rows])
    for key in ("ax_l2_dRe", "ax_l2_dIm", "po_l2_dRe"):
        v = np.array([r[key] for r in rows])
        say(f"  correlation {key}: with T_H/T_S {np.corrcoef(T, v)[0,1]:+.2f}, with 1-m(R+)/M {np.corrcoef(dm, v)[0,1]:+.2f}, with lambda {np.corrcoef(lamv, v)[0,1]:+.2f}")
    json.dump(dict(ref={f"{k[0]}_l{k[1]}": list(v) for k, v in ref.items()}, rows=rows), open(OUT / "qnm_band.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "qnm_band_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
