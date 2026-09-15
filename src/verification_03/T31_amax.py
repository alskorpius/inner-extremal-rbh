"""T31 (a): maximum spin at which the triple root of Delta is recoverable by retuning (d, scale) at fixed shape
(u1, u2, u3, w1, w2, w3, s_end, s1): bisection over a for several family shapes. Run: python src/verification_03/T31_amax.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent; OUT = HERE.parents[1] / "data" / HERE.name
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "stability"))
import T31_rotating_core as tc, T31_retune as tr, triple_root_monotone as trm

def shape_kw(base, smin):
    s1s, u_s, info = trm.find_merge_first(sigma_min=smin, base=base)
    return info["kw"] if s1s is not None else None

def triple_exists(kw, a):
    rows = []
    for scale in np.geomspace(0.7, 1.8, 34):
        o = tr.d_merge(scale, kw, a)
        if o is None: continue
        d, rs = o; mf = tc.MassFn(d, scale, kw)
        rows.append((scale, d, rs, rs**2 - 2 * mf.m(rs) * rs + a**2))
    for (s1_, d1, r1, D1), (s2_, d2, r2, D2) in zip(rows[:-1], rows[1:]):
        if D1 * D2 < 0:
            lev = lambda sc: (lambda o: (lambda d, rs: rs**2 - 2 * tc.MassFn(d, sc, kw).m(rs) * rs + a**2)(*o) if o else np.nan)(tr.d_merge(sc, kw, a))
            try:
                sc = brentq(lev, s1_, s2_, xtol=1e-8)
            except ValueError:
                continue
            d, rs = tr.d_merge(sc, kw, a); mf = tc.MassFn(d, sc, kw)
            lam = float(np.sqrt(3 / (8 * np.pi * (sc / (4 * np.pi * mf.R1**2)))))
            hz, _, _ = tc.horizons(mf, a)
            return dict(d=float(d), scale=float(sc), r_star=float(rs), lam=lam, NEC=bool(d <= kw["s1"] + 1e-9), roots=[(float(r), float(k)) for r, k in hz])
    return None

base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
shapes = [("base", base0, 1.0), ("u2 = 3", dict(base0, u2=3.0), 1.0), ("s_end = 8", dict(base0, s_end=8.0), 1.0), ("u3 = 12", dict(base0, u3=12.0), 1.0),
          ("sigma_min = 0.5", base0, 0.5), ("T15-optimum", dict(u1=2.05, u2=2.99, u3=3.99, w1=0.25, w2=0.35, w3=0.1, s_end=32.0), 0.46)]
out = {}
for tag, base, smin in shapes:
    kw = shape_kw(base, smin)
    if kw is None:
        print(tag, ": spherical triple root not found"); continue
    res = {}
    for a in (0.1, 0.15, 0.2, 0.3, 0.45, 0.6):
        t = triple_exists(kw, a)
        res[a] = t
        print(f"{tag:16s} a = {a:.2f}: " + (f"yes: d* = {t['d']:.4f}, scale = {t['scale']:.4f}, r* = {t['r_star']:.4f}, lambda = {t['lam']:.3f}, NEC {'ok' if t['NEC'] else 'VIOLATED'}, kappa_+ = {t['roots'][-1][1]:.4f}" if t else "no"), flush=True)
    ok = [a for a in res if res[a]]
    print(f"  -> {tag}: the triple root of Delta is recoverable for a <= {max(ok) if ok else 0} (from the grid), not recoverable for a >= {min([a for a in res if not res[a]] or [np.nan])}")
    out[tag] = {str(a): res[a] for a in res}
json.dump(out, open(OUT / "T31_amax.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
