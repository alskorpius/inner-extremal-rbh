"""T36 helper: the base triple-root profile, shared by all T36 scripts.

Wraps the family builder of src/polar_qnm/qnm_band.py (same object as in
src/verification_02/T26_ks.py, BASE0 with sigma_min = 1.0) and caches the
sampled fields to data/no_attractor/t36_profile_cache.npz so the later scripts start in ~0.1 s.

Exported: Profile (f, f', f'', rho, rho' with the mass rescaling m -> (1+eps) m),
horizon finder, and the T-region source functions p_x = -rho, p_perp = -rho - b rho'/2.

Run: python src/no_attractor/t36_profile.py   (builds the cache)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "t36_profile_cache.npz"

BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
SIGMA_MIN = 1.0


def _build_grid(n=200001, lo=1e-3, hi=6.0):
    """Sample m, m', m'' of the base family on a log grid (slow: rebuilds the family)."""
    sys.path.insert(0, str(ROOT / "src" / "polar_qnm"))
    sys.path.insert(0, str(ROOT / "src" / "stability"))
    import qnm_band as qb  # noqa: E402

    bg = qb.FamilyBG(qb.build_family(BASE0, SIGMA_MIN))
    r = np.geomspace(lo, hi, n)
    d = bg.fields(r)
    return r, d["m"], d["mp"], d["mpp"]


def load_grid(rebuild=False):
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE)
        return z["r"], z["m"], z["mp"], z["mpp"]
    r, m, mp, mpp = _build_grid()
    np.savez_compressed(CACHE, r=r, m=m, mp=mp, mpp=mpp)
    return r, m, mp, mpp


class Profile:
    """Base profile with m -> (1 + eps) m.  Units G = c = M = 1.

    parts(r) -> f, f', f'', rho, rho'   (all from the cached m, m', m'' by interpolation
    in log r, which keeps f''' smooth enough for the root structure used here)."""

    def __init__(self, eps=0.0, grid=None):
        self.eps = eps
        self.r, self.m_g, self.mp_g, self.mpp_g = grid if grid is not None else load_grid()
        self.lr = np.log(self.r)

    def _interp(self, r):
        lr = np.log(np.atleast_1d(np.asarray(r, float)))
        k = 1.0 + self.eps
        return (k * np.interp(lr, self.lr, self.m_g),
                k * np.interp(lr, self.lr, self.mp_g),
                k * np.interp(lr, self.lr, self.mpp_g))

    def parts(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        m, mp, mpp = self._interp(r)
        f = 1 - 2 * m / r
        fp = 2 * m / r**2 - 2 * mp / r
        fpp = -4 * m / r**3 + 4 * mp / r**2 - 2 * mpp / r
        rho = mp / (4 * np.pi * r**2)
        drho = (mpp - 2 * mp / r) / (4 * np.pi * r**2)
        return f, fp, fpp, rho, drho

    def f(self, r):
        return self.parts(r)[0]

    def rho(self, r):
        return self.parts(r)[3]

    def p_perp(self, r):
        """Transverse pressure of the class T^t_t = T^r_r: p_perp = -rho - b rho'/2."""
        _, _, _, rho, drho = self.parts(r)
        return -rho - np.asarray(r, float) * drho / 2

    def roots(self, lo=5e-2, hi=4.0, n=400001):
        rr = np.geomspace(lo, hi, n)
        f = self.f(rr)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        return [brentq(lambda x: self.f(np.array([x]))[0], rr[i], rr[i + 1], xtol=1e-14)
                for i in idx]

    def triple_root(self, lo=0.4, hi=1.2):
        """Location of the (near-)triple root: minimum of |f| inside the trapped region."""
        rr = np.linspace(lo, hi, 400001)
        f = self.f(rr)
        j = int(np.argmin(np.abs(f)))
        return float(rr[j]), float(f[j])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r, m, mp, mpp = load_grid(rebuild=True)
    P = Profile(0.0, grid=(r, m, mp, mpp))
    rm, fmin = P.triple_root()
    rp = P.roots()[-1]
    f, fp, fpp, rho, drho = P.parts(np.array([rm]))
    print(f"cache: {CACHE}  ({len(r)} points)")
    print(f"R_minus = {rm:.6f}  f = {fmin:.2e}  f' = {fp[0]:.2e}  f'' = {fpp[0]:.2e}")
    print(f"R_plus  = {rp:.6f}")
    print(f"8 pi rho R_minus^2 = {8*np.pi*rho[0]*rm**2:.8f}   p_perp(R_minus) = {P.p_perp(np.array([rm]))[0]:.3e}")
