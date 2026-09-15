"""T31 (a, continued): recovery of the triple root of Delta = r^2 - 2 m(r) r + a^2 at fixed a by retuning the profile.
The triple_root_monotone family with the base shape (u1..s_end, s1 = base s1*); d (dip depth) and scale (density normalization) are free.
Triple root = merger of the inner local minimum and local maximum of Delta (a stationary inflection) at the level Delta = 0:
for a given scale we look for d_c(scale) at which the two inner stationary points of Delta merge; then scale*, at which
Delta at the inflection point equals zero. Codimension 2 in (d, scale) — as in the spherical case. NEC ⇔ sigma >= 0 ⇔ d <= s1.
Run: python src/verification_03/T31_retune.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import T31_rotating_core as tc  # noqa: E402

LOG = []
RR = np.geomspace(2e-3, 2.5, 80001)


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def delta(mf, a):
    return RR**2 - 2 * mf.m(RR) * RR + a**2


def inner_stationary(mf, a):
    """Stationary points of Delta to the left of the global minimum (inner S-shape); Delta' computed analytically."""
    D = delta(mf, a)
    ig = int(np.argmin(D))
    dD = 2 * RR - 2 * mf.mp(RR) * RR - 2 * mf.m(RR)
    idx = np.nonzero(np.sign(dD[:-1]) * np.sign(dD[1:]) < 0)[0]
    idx = [i for i in idx if i < ig - 5]
    return [float(RR[i]) for i in idx], float(RR[ig]), float(D[ig])


def d_merge(scale, kw, a, d_hi=2.99):
    """d_c: the largest d at which the inner pair of stationary points of Delta still exists (for d > d_c the pair annihilates —
    a stationary inflection). We find a bracket for d with n >= n_hi + 2 by scanning, then bisect toward d_hi."""
    def n(d):
        return len(inner_stationary(tc.MassFn(d, scale, kw), a)[0])
    n_hi = n(d_hi)
    grid = np.linspace(0.3, d_hi - 0.01, 28)
    ok = [d for d in grid if n(d) >= n_hi + 2]
    if not ok:
        return None
    lo, hi = max(ok), d_hi
    if n(hi) >= n_hi + 2:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if n(mid) >= n_hi + 2: lo = mid
        else: hi = mid
    d = lo
    pts, rg, Dg = inner_stationary(tc.MassFn(d, scale, kw), a)
    # pair = the two points closest to each other
    gaps = [(pts[i + 1] - pts[i], i) for i in range(len(pts) - 1)]
    _, i0 = min(gaps)
    rs = 0.5 * (pts[i0] + pts[i0 + 1])
    return d, rs


def main():
    base, d0, scale0, kw = tc.base_massfn()
    s1 = kw["s1"]
    say(f"Base: d0 = {d0:.5f}, scale0 = {scale0:.5f}, s1 = {s1:.4f} (NEC ⇔ d <= s1); a = 0 control below")
    res = {}
    for a in (0.0, 0.1, 0.2, 0.3, 0.6):
        say(f"\na = {a}: merger line of stationary points d_c(scale) and the Delta level at the inflection")
        rows = []
        for scale in np.geomspace(0.7, 1.6, 41):
            out = d_merge(scale, kw, a)
            if out is None:
                continue
            d, rs = out
            mf = tc.MassFn(d, scale, kw)
            D = float(rs**2 - 2 * mf.m(rs) * rs + a**2)
            rows.append((scale, d, rs, D))
        for scale, d, rs, D in rows[::5]:
            say(f"    scale = {scale:.4f}: d_c = {d:.4f} ({'NEC ok' if d <= s1 else 'NEC violated: sigma_min = %.2f' % (s1 - d)}), r* = {rs:.4f}, Delta(r*) = {D:+.4f}")
        found = None
        for (s_1, d_1, r_1, D_1), (s_2, d_2, r_2, D_2) in zip(rows[:-1], rows[1:]):
            if D_1 * D_2 < 0:
                def lev(scale):
                    o = d_merge(scale, kw, a)
                    if o is None: return np.nan
                    d, rs = o; mf = tc.MassFn(d, scale, kw)
                    return rs**2 - 2 * mf.m(rs) * rs + a**2
                try:
                    sc = brentq(lev, s_1, s_2, xtol=1e-9)
                except ValueError:
                    continue
                d, rs = d_merge(sc, kw, a); mf = tc.MassFn(d, sc, kw)
                hz, _, _ = tc.horizons(mf, a)
                rho_c = sc / (4 * np.pi * mf.R1**2); lam = float(np.sqrt(3 / (8 * np.pi * rho_c)))
                Dp = 2 * rs - 2 * mf.mp(rs) * rs - 2 * mf.m(rs); Dpp = 2 - 2 * mf.mpp(rs) * rs - 4 * mf.mp(rs)
                found = dict(d=float(d), scale=float(sc), r_star=float(rs), lam=lam, sigma_min=float(s1 - d), NEC=bool(d <= s1 + 1e-9),
                             Dp=float(Dp), Dpp=float(Dpp), roots=[(float(r), float(k)) for r, k in hz])
                say(f"  triple root of Delta at a = {a}: d* = {d:.5f} (s1 = {s1:.4f}, sigma_min = {s1-d:+.3f} → NEC {'satisfied' if d <= s1 else 'VIOLATED'}), scale = {sc:.5f}, r* = {rs:.5f}; "
                    f"Delta' = {Dp:+.1e}, Delta'' = {Dpp:+.1e}; lambda = ell/M = {lam:.4f}; roots: " + "; ".join(f"r = {r:.4f}, kappa = {k:+.2e}" for r, k in hz))
                break
        if found is None:
            say(f"  at a = {a}: the Delta level at the inflection does not cross zero in the range scale ∈ [0.7, 1.6] (at larger scale the inner pair of stationary points of Delta disappears) — the triple root is not recovered")
        res[f"a{a}"] = dict(rows=[(float(s), float(d), float(r), float(D)) for s, d, r, D in rows], triple=found)
    json.dump(res, open(OUT / "T31_retune.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T31_retune_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
