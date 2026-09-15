"""T31: rotating analog of the scenario in the Gurses-Gursey class (Kerr with M -> m(r)).
(a) Horizons Delta = r^2 - 2 m(r) r + a^2 and kappa = Delta'/(2 (r_h^2 + a^2)) for m(r) of the base profile and Hayward (ell/M = 0.271) at a = 0.3, 0.6, 0.9;
    disruption of the triple root: kappa_-(a) (expectation: delta Delta = a^2 => kappa ∝ a^{4/3}); recovery of degeneracy at a = 0.6 —
    fsolve over (d, scale, r*) for Delta = Delta' = Delta'' = 0 (codimension 2 in the profile parameters: d and scale at fixed shape).
(b) the symbolic part is factored out into T31_ring_curvature.py.
Run: python src/verification_03/T31_rotating_core.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, fsolve

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "polar_qnm"))
sys.path.insert(0, str(ROOT / "src" / "stability"))
sys.argv = [sys.argv[0], "0.05"]
import qnm_band as qb  # noqa: E402
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)


class MassFn:
    """m(r) = scale R1 mI(ln(r/R1)) of the triple_root_monotone family; derivatives via a cubic spline over ln u."""

    def __init__(self, d, scale, kw):
        sig, s, mI, H, Hp = trm.profile(d, **kw)
        self.Itot = float(mI[-1]); self.scale = scale; self.R1 = 1.0 / (scale * self.Itot)
        self.S = CubicSpline(trm.LU, mI)
        self.sig_min = float(sig.min())

    def m(self, r):
        x = np.log(np.asarray(r, float) / self.R1)
        return self.scale * self.R1 * self.S(x)

    def mp(self, r):
        r = np.asarray(r, float); x = np.log(r / self.R1)
        return self.scale * self.R1 * self.S(x, 1) / r

    def mpp(self, r):
        r = np.asarray(r, float); x = np.log(r / self.R1)
        return self.scale * self.R1 * (self.S(x, 2) - self.S(x, 1)) / r**2


def base_massfn():
    s1s, u_s, info = trm.find_merge_first(sigma_min=1.0, base=BASE0)
    kw = info["kw"]; d = s1s - 1.0
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    return MassFn(d, scale, kw), d, scale, kw


class Hayward:
    def __init__(self, ell, M=1.0):
        self.ell, self.M = ell, M

    def m(self, r):
        r = np.asarray(r, float); return self.M * r**3 / (r**3 + 2 * self.M * self.ell**2)

    def mp(self, r):
        r = np.asarray(r, float); L = 2 * self.M * self.ell**2
        return self.M * 3 * r**2 * L / (r**3 + L) ** 2

    def mpp(self, r):
        r = np.asarray(r, float); L = 2 * self.M * self.ell**2
        return self.M * L * (6 * r * (r**3 + L) - 18 * r**4) / (r**3 + L) ** 3


def horizons(mf, a):
    D = lambda r: r**2 - 2 * mf.m(r) * r + a**2
    rr = np.geomspace(1e-3, 4.0, 200001); v = D(rr)
    idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    roots = [brentq(D, rr[i], rr[i + 1], xtol=1e-13) for i in idx]
    out = []
    for rh in roots:
        Dp = 2 * rh - 2 * mf.mp(rh) * rh - 2 * mf.m(rh)
        out.append((rh, Dp / (2 * (rh**2 + a**2))))
    imin = int(np.argmin(v[rr < 1.5])); rmin, vmin = float(rr[rr < 1.5][imin]), float(v[rr < 1.5][imin])
    return out, rmin, vmin


def main():
    t0 = time.time()
    say("T31: Gurses-Gursey class (Kerr with M -> m(r)) with the scenario profile and Hayward")
    base, d0, scale0, kw = base_massfn()
    hay = Hayward(0.271)
    say(f"  base: d = {d0:.5f}, scale = {scale0:.5f}, R1 = {base.R1:.5f}, sigma_min = {base.sig_min:.3f}; Hayward ell = 0.271")
    res = {"horizons": {}}
    say("\n(a) Horizons and kappa = Delta'/(2(r_h^2 + a^2))")
    for name, mf in (("base", base), ("Hayward", hay)):
        for a in (0.0, 0.3, 0.6, 0.9):
            hz, rmin, vmin = horizons(mf, a)
            txt = "; ".join(f"r = {r:.4f}, kappa = {k:+.4e}" for r, k in hz)
            say(f"  {name}, a = {a}: {len(hz)} roots: {txt}; min Delta for r < 1.5: {vmin:+.3e} (r = {rmin:.3f})")
            res["horizons"][f"{name}_a{a}"] = dict(roots=[(float(r), float(k)) for r, k in hz], Dmin=vmin, r_Dmin=rmin)
    # disruption law kappa_-(a) for the base
    say("\n  disruption of the base triple root by small spin: kappa_-(a) (expectation ∝ a^{4/3}: delta Delta = a^2 when f''' != 0)")
    ka = []
    for a in (0.003, 0.01, 0.03, 0.1, 0.3):
        hz, _, _ = horizons(base, a)
        inner = [h for h in hz if h[0] < 1.0]
        if inner:
            ka.append((a, inner[0][0], inner[0][1]))
            say(f"    a = {a}: r_- = {inner[0][0]:.5f}, kappa_- = {inner[0][1]:+.4e}, kappa/a^(4/3) = {inner[0][1]/a**(4/3):+.4f}")
        else:
            say(f"    a = {a}: no inner root (Delta > 0 inside) — the triple root is lifted, trapped region ...")
    if len(ka) >= 2:
        p = np.polyfit(np.log([k[0] for k in ka]), np.log([abs(k[2]) for k in ka]), 1)[0]
        say(f"    fit exponent: {p:.3f}")
    res["kappa_vs_a"] = ka
    # recovery of degeneracy at a = 0.6
    say("\n  recovery of the triple root of Delta at a = 0.6: fsolve over (d, scale, r*)")
    a = 0.6

    def eqs(p):
        d, scale, rs = p
        if not (0.05 < d < 2.99 and 0.1 < scale < 20 and 0.05 < rs < 1.9):
            return [1e3, 1e3, 1e3]
        mf = MassFn(d, scale, kw)
        m, mp, mpp = mf.m(rs), mf.mp(rs), mf.mpp(rs)
        D = rs**2 - 2 * m * rs + a**2
        Dp = 2 * rs - 2 * mp * rs - 2 * m
        Dpp = 2 - 2 * mpp * rs - 4 * mp
        return [D, Dp, Dpp]
    best = None
    for guess in ([d0, scale0, 0.68], [d0, scale0 * 1.2, 0.8], [d0 * 1.1, scale0, 0.9], [d0 * 0.9, scale0 * 0.8, 0.6]):
        p, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-13)
        r_ = np.max(np.abs(eqs(p)))
        if r_ < 1e-8 and (best is None or r_ < best[1]):
            best = (p, r_)
    if best is None:
        say("    matching not found")
        res["retuned_a0.6"] = None
    else:
        d, scale, rs = best[0]
        mf = MassFn(d, scale, kw)
        hz, _, _ = horizons(mf, a)
        ell = np.sqrt(3 / (8 * np.pi * (scale / (4 * np.pi * mf.R1**2))))
        say(f"    found: d = {d:.5f} (base {d0:.5f}), scale = {scale:.5f} (base {scale0:.5f}), r* = {rs:.5f}; residual {best[1]:.1e}; sigma_min = {mf.sig_min:.3f} (NEC: >= 0); "
            f"lambda = ell/M = {ell:.4f} (base {np.sqrt(3/(8*np.pi*(scale0/(4*np.pi*base.R1**2)))):.4f}); all roots of Delta: " + "; ".join(f"r = {r:.4f}, kappa = {k:+.2e}" for r, k in hz))
        say("    conclusion: the condition Delta = Delta' = Delta'' = 0 at fixed a — three equations for (d, scale, r*): degeneracy is recovered by tuning two profile parameters (the same codimension as in the spherical case)")
        res["retuned_a0.6"] = dict(d=float(d), scale=float(scale), r_star=float(rs), lam=float(ell), sigma_min=mf.sig_min, roots=[(float(r), float(k)) for r, k in hz])

    res["ring"] = "see T31_ring_curvature.py"
    json.dump(res, open(OUT / "T31_rotating_core.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T31_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
