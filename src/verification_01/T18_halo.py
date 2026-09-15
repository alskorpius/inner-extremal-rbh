"""T18/T19: is a halo (mass outside R+) necessary for a monotone triple root; halo over the extended family (T15).
Units G = c = M = 1. Family — stability/triple_root_monotone (via polar_qnm/qnm_band.build_family, FamilyBG).
(A) hard truncation of the base profile at R_c inside the trapped region, renormalization M' = m(R_c) — the triple root is preserved,
    halo = 0, the exterior is exactly Schwarzschild; axial QNM l = 2 (time domain) — control for a zero shift.
(B) smooth versions: scan (s_end, w3, sigma_min) — halo, K_max in the truncation layer (R > R-) versus K(0) = 24/ell^4,
    Kretschmann for g_tt g_rr = -1: K = f''^2 + 4 f'^2/R^2 + 4 (1-f)^2/R^4 (control: Schwarzschild 48 M^2/R^6, de Sitter 24/ell^4).
(C) QNM for smooth nearly compact profiles.  (D) T19: halo and truncation location for the T15 points.
Run: python src/verification_01/T18_halo.py
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
OUT.mkdir(parents=True, exist_ok=True)
PQ = HERE.parents[0] / "polar_qnm"
PQD = HERE.parents[1] / "data" / "polar_qnm"
sys.path.insert(0, str(PQ))
sys.path.insert(0, str(HERE.parents[0] / "stability"))
sys.argv = [sys.argv[0], "0.05"]
import qnm_band as qb  # noqa: E402
import axial_td as ax  # noqa: E402
import polar_qnm as pq  # noqa: E402

LOG = []
BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
REF = json.load(open(PQD / "qnm_band.json", encoding="utf-8"))["ref"]["ax_l2"]   # Schwarzschild, same code and grid


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def kretschmann(R, f, fp, fpp):
    return fpp**2 + 4 * fp**2 / R**2 + 4 * (1 - f) ** 2 / R**4


def geometry(bg, R):
    d = bg.fields(R)
    m, mp, mpp = d["m"], d["mp"], d["mpp"]
    f = 1 - 2 * m / R
    fp = 2 * m / R**2 - 2 * mp / R
    fpp = -4 * m / R**3 + 4 * mp / R**2 - 2 * mpp / R
    return m, f, fp, fpp, kretschmann(R, f, fp, fpp)


def horizons(bg):
    R = np.geomspace(1e-2, 4.0, 400001)
    f = 1 - 2 * bg.fields(R)["m"] / R
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = [brentq(lambda x: 1 - 2 * bg.fields(np.array([x]))["m"][0] / x, R[i], R[i + 1], xtol=1e-13) for i in idx]
    j = int(np.argmin(np.abs(f[R < 1.5]))); return roots, float(R[j]), float(f[j])


def summarize(bg, fam, tag):
    lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    roots, Rm, fmin = horizons(bg)
    Rp = bg.rh
    kp = qb.kappa_at(bg, Rp)
    m_plus = float(bg.fields(np.array([Rp]))["m"][0])
    R = np.geomspace(1e-3, 6.0, 60000)
    m, f, fp, fpp, K = geometry(bg, R)
    K0 = 24 / lam**4
    lay = R > Rm * 1.02
    iK = int(np.argmax(K[lay])); Kmax_layer = float(K[lay][iK]); R_Kmax = float(R[lay][iK])
    rho = fam.rho(R); drho = fam.drho(R); pperp = -rho - R * drho / 2
    return dict(tag=tag, lam=lam, R_minus=Rm, f_min=fmin, R_plus=Rp, kappa_plus=kp, T_ratio=4 * kp, halo=1 - m_plus,
                K0=K0, K_center_num=float(K[0]), Kmax_layer=Kmax_layer, R_Kmax=R_Kmax, K_ratio=Kmax_layer / K0,
                pperp_max=float(pperp.max() / fam.rho_c), NEC=bool(np.all(-R * drho / 2 >= -1e-12 * fam.rho_c)))


def qnm_shift(bg):
    t, y = ax.evolve_axial(bg, 2)
    wr, wi = pq.matrix_pencil(t, y, 80.0, 200.0)
    return wr / REF[0] - 1, wi / REF[1] - 1


class CutBG:
    """Base profile truncated at R_c (rho = 0 for R > R_c), renormalized to M' = m(R_c) = 1."""

    def __init__(self, fam, Rc):
        self.fam, self.scen, self.M = fam, "cut", 1.0
        self.Mp = float(np.interp(Rc, fam._Rg, fam._mg))
        self.Rc = Rc / self.Mp
        self.rh = 2.0
        self.ell_over_Mp = float(np.sqrt(3 / (8 * np.pi * fam.rho_c))) / self.Mp
        self.rho_c = fam.rho_c * self.Mp**2

    def rho(self, R):
        R = np.asarray(R, float); return np.where(R < self.Rc, self.fam.rho(R * self.Mp) * self.Mp**2, 0.0)

    def drho(self, R):
        R = np.asarray(R, float); return np.where(R < self.Rc, self.fam.drho(R * self.Mp) * self.Mp**3, 0.0)

    def fields(self, r):
        r = np.asarray(r, float); rr = r * self.Mp
        inside = r < self.Rc
        m = np.where(inside, np.interp(rr, self.fam._Rg, self.fam._mg) / self.Mp, 1.0)
        rho = self.rho(r); drho = self.drho(r)
        return dict(m=m, mp=4 * np.pi * r**2 * rho, mpp=8 * np.pi * r * rho + 4 * np.pi * r**2 * drho, sigma=np.ones_like(r), sigp=np.zeros_like(r))


def main():
    t0 = time.time()
    say("T18: halo and the triple root. Kretschmann formula control:")
    R = np.array([2.5, 4.0]); f = 1 - 2 / R; say(f"  Schwarzschild: K/(48/R^6) = {kretschmann(R, f, 2 / R**2, -4 / R**3) / (48 / R**6)}")
    ell = 0.3; R = np.array([0.1, 0.2]); f = 1 - R**2 / ell**2; say(f"  de Sitter: K/(24/ell^4) = {kretschmann(R, f, -2 * R / ell**2, -2 / ell**2 + 0 * R) / (24 / ell**4)}")
    fam0 = qb.build_family(BASE0, 1.0); bg0 = qb.FamilyBG(fam0)
    s0 = summarize(bg0, fam0, "base"); say(f"  base: λ = {s0['lam']:.3f}, R- = {s0['R_minus']:.4f}, R+ = {s0['R_plus']:.4f}, halo = {100*s0['halo']:.2f} %, K(0)/(24/ell^4) = {s0['K_center_num']/s0['K0']:.4f}, K_max(layer)/K(0) = {s0['K_ratio']:.3f} at R = {s0['R_Kmax']:.3f}")
    out = dict(control=s0)

    # (A) hard truncation
    say("\n(A) Hard truncation of the base profile at R_c (inside the trapped region), M' = m(R_c):")
    rowsA = []
    for Rc in (1.2, 1.4, 1.6):
        fRc = 1 - 2 * float(np.interp(Rc, fam0._Rg, fam0._mg)) / Rc
        cut = CutBG(fam0, Rc)
        roots, Rm, fmin = horizons(cut)
        row = dict(Rc=Rc, f_Rc=fRc, Mp=cut.Mp, lam_new=cut.ell_over_Mp, R_minus_over_Mp=Rm, f_min=fmin, R_plus_over_Mp=2.0, kappa_plus=0.25, halo=0.0,
                   roots=[float(r) for r in roots])
        if Rc == 1.4:
            dre, dim = qnm_shift(cut); row["qnm_dRe"], row["qnm_dIm"] = dre, dim
        rowsA.append(row)
        say(f"  R_c = {Rc}: f(R_c) = {fRc:+.3f} (< 0: {fRc < 0}), M' = {cut.Mp:.4f}, λ' = ℓ/M' = {cut.ell_over_Mp:.3f}, R-/M' = {Rm:.4f} (|f|min = {abs(fmin):.1e}), R+ = 2M' exactly, κ+ = 1/(4M') exactly, halo 0"
            + (f"; QNM l=2: δRe/Re = {row['qnm_dRe']:+.1e}, δIm/Im = {row['qnm_dIm']:+.1e} (method error level ~1e-5/3e-4)" if "qnm_dRe" in row else "") + f"; roots of f: {[round(r, 4) for r in roots]}")
    out["A"] = rowsA

    # (B) smooth versions: scan of the truncation steepness
    say("\n(B) Smooth versions: halo and curvature of the truncation layer (K_max for R > R-) versus K(0) = 24/ell^4:")
    rowsB = []
    for s_end in (24.0, 100.0, 500.0):
        for w3 in (0.1, 0.05, 0.02, 0.01):
            for smin in (1.0, 0.5, 0.2, 0.0):
                base = dict(BASE0, s_end=s_end, w3=w3)
                try:
                    fam = qb.build_family(base, smin)
                except Exception as e:  # noqa: BLE001
                    say(f"  s_end={s_end:>5}, w3={w3:<5}, σ_min={smin}: error {e}"); continue
                if fam is None:
                    say(f"  s_end={s_end:>5}, w3={w3:<5}, σ_min={smin}: triple root not found"); continue
                bg = qb.FamilyBG(fam)
                s = summarize(bg, fam, f"s_end={s_end}, w3={w3}, σ_min={smin}")
                s.update(s_end=s_end, w3=w3, sigma_min=smin)
                rowsB.append(s)
                say(f"  s_end={s_end:>5}, w3={w3:<5}, σ_min={smin}: λ = {s['lam']:.3f}, R- = {s['R_minus']:.3f}, R+ = {s['R_plus']:.4f}, T_H/T_S = {s['T_ratio']:.3f}, halo = {100*s['halo']:.3f} %, K_max(layer)/K(0) = {s['K_ratio']:.3f} at R = {s['R_Kmax']:.2f}, p⊥max/ρc = {s['pperp_max']:.2f}, NEC {s['NEC']}")
    out["B"] = rowsB
    ok1 = [r for r in rowsB if r["K_ratio"] <= 1.0]; ok2 = [r for r in rowsB if r["K_ratio"] <= 2.0]
    if ok1:
        b1 = min(ok1, key=lambda r: r["halo"]); say(f"  minimum halo at K_layer ≤ K(0): {100*b1['halo']:.3f} % ({b1['tag']}, K_ratio {b1['K_ratio']:.2f})")
    else:
        say("  no configuration gives K_layer ≤ K(0)")
    if ok2:
        b2 = min(ok2, key=lambda r: r["halo"]); say(f"  minimum halo at K_layer ≤ 2K(0): {100*b2['halo']:.3f} % ({b2['tag']}, K_ratio {b2['K_ratio']:.2f})")
    out["min_halo_K1"] = b1 if ok1 else None; out["min_halo_K2"] = b2 if ok2 else None

    # (C) QNM for nearly compact smooth profiles
    say("\n(C) QNM l = 2 (axial, time domain) for nearly compact smooth profiles:")
    rowsC = []
    for tag, base, smin in (("w3 = 0.01", dict(BASE0, w3=0.01), 1.0), ("σ_min = 0.2", BASE0, 0.2), ("σ_min = 0.0", BASE0, 0.0),
                            ("s_end = 100, w3 = 0.02", dict(BASE0, s_end=100.0, w3=0.02), 1.0)):
        fam = qb.build_family(base, smin)
        if fam is None: continue
        bg = qb.FamilyBG(fam); s = summarize(bg, fam, tag)
        dre, dim = qnm_shift(bg); s.update(qnm_dRe=dre, qnm_dIm=dim)
        rowsC.append(s)
        say(f"  {tag}: λ = {s['lam']:.3f}, halo = {100*s['halo']:.3f} %, K_ratio = {s['K_ratio']:.2f}, δRe/Re = {dre:+.2e}, δIm/Im = {dim:+.2e}   [{time.time()-t0:.0f} s]")
    out["C"] = rowsC

    # (D) T19: points of the extended family (T15) — what sets the halo:
    say("\n(D) T19: points of the extended family (T15) — what sets the halo:")
    q = json.load(open(OUT / "T15_qnm.json", encoding="utf-8"))
    lb = json.load(open(OUT / "T15_lambda_bound.json", encoding="utf-8"))
    rowsD = []
    pts = [(p["tag"], p["base"], p["sigma_min"], p) for p in q["points"]] + [("T15 maximum", q["opt30"]["base"], q["opt30"]["sigma_min"], q["opt30"])]
    pts += [(r["tag"], r["base"], r["sigma_min"], r) for r in lb["rows"] if r.get("ok") and r["tag"].startswith(("s_end", "u3 =", "u1 ="))]
    for tag, base, smin, p in pts:
        fam = qb.build_family(base, smin)
        if fam is None: continue
        bg = qb.FamilyBG(fam)
        R1 = fam.R1; Rp = bg.rh
        row = dict(tag=tag, lam=p["lam"], halo=p["halo"], T_ratio=p["T_ratio"], dRe=p.get("dRe"), dIm=p.get("dIm"), u3R1_over_Rp=base["u3"] * R1 / Rp, s_end=base["s_end"], w3=base["w3"], R1=R1, R_plus=Rp)
        rowsD.append(row)
        say(f"  {tag:14s}: λ = {p['lam']:.3f}, halo = {100*p['halo']:.2f} %, T_H/T_S = {p['T_ratio']:.3f}, QNM δRe/δIm = {('%+.1e/%+.1e' % (p['dRe'], p['dIm'])) if p.get('dRe') is not None else '—'}, u3·R1/R+ = {row['u3R1_over_Rp']:.2f}, s_end = {base['s_end']:.0f}, w3 = {base['w3']:.3f}")
    h = np.array([r["halo"] for r in rowsD]); x = np.array([r["u3R1_over_Rp"] for r in rowsD]); l = np.array([r["lam"] for r in rowsD])
    say(f"  correlation of ln(halo) with u3·R1/R+: {np.corrcoef(x, np.log(np.maximum(h, 1e-6)))[0,1]:+.2f}; with λ: {np.corrcoef(l, np.log(np.maximum(h, 1e-6)))[0,1]:+.2f}")
    out["D"] = rowsD
    json.dump(out, open(OUT / "T18_halo.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T18_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
