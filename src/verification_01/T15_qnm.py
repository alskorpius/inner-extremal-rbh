"""T15, part 2: (a) repeated maximization of lambda with an extended range of s1 (up to 30) — check whether the optimum ran into the search boundary;
(b) for the companion-paper synthesis points (lambda ~ 0.45, 0.58, 0.77) among the found configurations: axial QNM l = 2 (time domain, as in qnm_band.py)
and the shadow (critical impact parameter b_c against 3 sqrt(3) M). Run after T15_lambda_bound.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "stability"))
sys.path.insert(0, str(ROOT / "src" / "approach_map"))
sys.path.insert(0, str(ROOT / "src" / "polar_qnm"))
sys.argv = [sys.argv[0], "0.05"]
import T15_lambda_bound as T15  # noqa: E402
import triple_root_monotone as trm  # noqa: E402
import qnm_band as qb  # noqa: E402
import axial_td as ax  # noqa: E402
import polar_qnm as pq  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class BG(qb.FamilyBG):
    """FamilyBG with the outer horizon found as the largest root of f on (0.3, 3)."""

    def __init__(self, fam, M=1.0):
        self.fam, self.M, self.scen = fam, M, "family"
        self._sigp = np.gradient(fam.sig, trm.LU)
        r = np.geomspace(0.3, 3.0, 200001)
        f = 1 - 2 * self._m(r) / r
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        i = idx[-1]
        self.rh = brentq(lambda x: 1 - 2 * self._m(x) / x, r[i], r[i + 1], xtol=1e-14)


def family_from(base, smin, s1_hi=30.0):
    s1, u_s, kw, reason = T15.find_merge(base, smin, s1_hi=s1_hi)
    if s1 is None:
        return None
    sig, s, mI, H, Hp = trm.profile(s1 - smin, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    return trm.MonotoneFamily(s1 - smin, scale, kw=kw)


def shadow(bg):
    r = np.linspace(bg.rh * 1.01, 8.0, 40000)
    m = bg.fields(r)["m"]
    f = 1 - 2 * m / r
    g = f / r**2
    i = int(np.argmax(g))
    r_ph = r[i]; b_c = r_ph / np.sqrt(f[i])
    return r_ph, b_c, b_c / (3 * np.sqrt(3)) - 1


def main():
    t0 = time.time()
    data = json.load(open(OUT / "T15_lambda_bound.json", encoding="utf-8"))
    say("T15-2. (a) Repeated maximization of lambda with s1 up to 30 (checking the s1 search boundary)")
    best = data["best"]
    keys = ["u1", "u2", "u3", "w1", "w2", "w3", "s_end"]
    x0 = np.array([np.log(best["base"][k]) for k in keys] + [best["sigma_min"]])

    def obj(x):
        base = {k: float(np.exp(v)) for k, v in zip(keys, x[:7])}
        smin = float(np.clip(x[7], 0.0, 1.95))
        if base["u3"] <= base["u2"] * 1.05 or base["s_end"] < 2.5:
            return 10.0
        try:
            r = T15.characterize(base, smin, s1_hi=30.0)
        except Exception:  # noqa: BLE001
            return 10.0
        if not r["ok"] or not r["NEC_WEC"] or r["kappa_plus"] < 0:
            return 10.0
        return -r["lam"]
    res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=200, xatol=1e-3, fatol=1e-5))
    base_opt = {k: float(np.exp(v)) for k, v in zip(keys, res.x[:7])}; smin_opt = float(np.clip(res.x[7], 0.0, 1.95))
    r_opt = T15.characterize(base_opt, smin_opt, s1_hi=30.0)
    say(f"  parameters: {', '.join(f'{k} = {v:.3g}' for k, v in base_opt.items())}, sigma_min = {smin_opt:.3g}")
    say(T15.fmt(r_opt, "optimum (s1 up to 30)"))
    say("\n(b) companion-paper synthesis points: QNM (axial, l = 2) and shadow")
    pool = [r for r in data["random_found"] if r["ok"] and r["NEC_WEC"] and r["kappa_plus"] > 0.02] + [r for r in data["rows"] if r["ok"] and r["NEC_WEC"]]
    bgS = pq.Background(None)
    tS, yS = ax.evolve_axial(bgS, 2); wS = pq.matrix_pencil(tS, yS, 80.0, 200.0)
    rows = []
    picks = [("base", data["rows"][0])]
    for target in (0.45, 0.58, 0.77):
        cands = sorted(pool, key=lambda r: abs(r["lam"] - target))[:5]
        picks.append((f"lambda ~ {target}", cands))
    for tag, cands in picks:
        if isinstance(cands, dict):
            cands = [cands]
        fam, r = None, None
        for c in cands:
            fam = family_from(c["base"], c["sigma_min"])
            if fam is not None:
                r = c; break
        if fam is None:
            say(f"  {tag}: family not reproduced"); continue
        bg = BG(fam)
        t, y = ax.evolve_axial(bg, 2); w = pq.matrix_pencil(t, y, 80.0, 200.0)
        r_ph, b_c, db = shadow(bg)
        rr = np.linspace(3.0, 60.0, 500); dm = float(np.max(np.abs(bg.fields(rr)["m"] - 1.0)))
        row = dict(tag=tag, lam=r["lam"], R_minus=r["R_minus"], R_plus=bg.rh, kappa_plus=r["kappa_plus"], T_ratio=r["T_ratio"], halo=r["halo"],
                   dRe=w[0] / wS[0] - 1, dIm=w[1] / wS[1] - 1, r_ph=r_ph, b_c=b_c, shadow_dev=db, m_dev_3M=dm, base=r["base"], sigma_min=r["sigma_min"])
        rows.append(row)
        say(f"  {tag}: lam = {r['lam']:.3f}, R- = {r['R_minus']:.3f}, R+ = {bg.rh:.3f}, kappa+ = {r['kappa_plus']:.4f}, T_H/T_S = {r['T_ratio']:.3f}, halo = {r['halo']:.4f}; "
            f"QNM l=2: dRe/Re = {row['dRe']:+.2e}, dIm/Im = {row['dIm']:+.2e}; shadow: r_ph = {r_ph:.4f}, b_c/b_Schw - 1 = {db:+.1e}; max|m - M| at r >= 3M: {dm:.1e}   [{time.time()-t0:.0f} s]")
    json.dump(dict(opt30=r_opt, base_opt=base_opt, sigma_min_opt=smin_opt, points=rows), open(OUT / "T15_qnm.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T15_qnm_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
