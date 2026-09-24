"""Polar operator of the de Sitter core at large plateau index p, without float cancellation.

polar_qnm.potential_matrix builds H_P Q^2 = r^2 (2m' - r m'')/2 in float64; for a plateau of index p this is
~ r^(p+4), obtained as a difference of O(r^2) terms, so for p = 10 it is pure round-off below r ~ 0.03.
Here the same potential (polar_qnm.potential_matrix, formulas copied verbatim) is evaluated with H_P Q^2, its
square root and the static term H_P^{-1/2} (N (H_P^{1/2})')' built symbolically (sympy, cancelled exactly), then
  (1) the limit r^2 V22 -> -(p+2)/4 [2 l(l+1) - (p+4)] is checked at r = 1e-6 with 30-digit arithmetic;
  (2) the lowest omega^2 of the two-channel Sturm-Liouville problem (matrices of t37_core_stability.py) is
      computed against the inner cut-off for p = 4 and p = 10, l = 2, 3, 5.
Core: m = M r^3/(r^p + L^p)^(3/p) (generalised Bardeen, plateau index p), M = 1, L = 0.35.

Run: python src/nonspherical/t37_large_p.py        (~15-20 min: 30-digit arithmetic at every grid point)
"""
import json
import sys
from pathlib import Path

import mpmath as mpm
import numpy as np
import sympy as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import t37_core_stability as t  # noqa: E402

mpm.mp.dps = 40
r = sp.symbols("r", positive=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def build(p, L=0.35, M=1):
    Ls = sp.Rational(str(L))
    m = M * r**3 / (r**p + Ls**p) ** sp.Rational(3, p)
    mp, mpp = sp.diff(m, r), sp.diff(m, r, 2)
    HPQ2 = sp.factor(sp.simplify(r**2 * (2 * mp - r * mpp) / 2))
    sq = sp.sqrt(HPQ2)
    N = 1 - 2 * m / r
    term = sp.simplify(sp.diff(N * sp.diff(sq, r), r) / sq)
    rho = mp / (4 * sp.pi * r**2)
    sig = sp.simplify(-r * sp.diff(sp.log(rho), r))
    kappa = sp.simplify(sig / 2 - 1 - r * sp.diff(sig, r) / (2 * sig))
    return dict(m=m, N=N, H=-mp / r**2, HPQ2=HPQ2, term=term, kappa=kappa)


def V_sym(B, l):
    lam = (l - 1) * (l + 2)
    N, H, HPQ2, kappa, term, m = B["N"], B["H"], B["HPQ2"], B["kappa"], B["term"], B["m"]
    rDr = 1 - N + 2 * r**2 * H
    a = 6 * m / r + 2 * r**2 * H
    b = lam + 4 * HPQ2 / r**2
    V11 = (l * (l + 1) * lam - 2 * N * lam + a * rDr) / (r**2 * (a + lam)) + 2 * N * lam * b / (r**2 * (a + lam) ** 2)
    W = (lam + 1 - N - 2 * r**2 * H + 2 * N * kappa) / (a + lam) + 2 * N * b / (a + lam) ** 2
    V12 = -sp.sqrt(4 * lam * HPQ2) * W / r**3
    V22 = (kappa * l * (l + 1) / r**2 + 4 * HPQ2 / (r**4 * (a + lam)) * (lam + 1 - N - 2 * r**2 * H + 4 * N * kappa)
           + term + 8 * N * HPQ2 * b / (r**4 * (a + lam) ** 2))
    return N, V11, V12, V22


def main():
    res = {}
    for p in [4, 10]:
        B = build(p)
        say(f"p = {p}: H_P Q^2 = {B['HPQ2']}")
        mfun = sp.lambdify(r, B["m"], "mpmath")

        class Core:  # minimal core for t._sl_matrices (needs fields()['m'])
            @staticmethod
            def fields(x):
                x = np.atleast_1d(np.asarray(x, float))
                return dict(m=np.array([float(mfun(mpm.mpf(v))) for v in x]))

        out = dict(limits={}, spectra={})
        for l in [2, 3, 5]:
            N, V11, V12, V22 = V_sym(B, l)
            lim = float(sp.N((r**2 * V22).subs(r, sp.Rational(1, 10**6)), 30))
            pred = -(p + 2) / 4 * (2 * l * (l + 1) - (p + 4))
            out["limits"][l] = dict(numeric=lim, predicted=pred)
            say(f"  l={l}: r^2 V22 (r=1e-6, 30 digits) = {lim:.6f}   predicted {pred:.6f}")
            fN, f11, f12, f22 = (sp.lambdify(r, e, "mpmath") for e in (N, V11, V12, V22))
            rg = np.linspace(0.05, 1.5, 3000)
            Nv = np.array([float(fN(mpm.mpf(x))) for x in rg])
            sc = np.nonzero(np.sign(Nv[:-1]) != np.sign(Nv[1:]))[0]
            Rm = float(rg[sc[0]]) if len(sc) else 1.0
            r_hi = 0.9 * Rm
            rows = []
            for rc in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                n = 1601
                rr = np.linspace(rc, r_hi, n)
                ev = lambda F: np.array([float(F(mpm.mpf(x))) for x in rr])
                A = t._sl_matrices(Core, rr, [[ev(f11), ev(f12)], [ev(f12), ev(f22)]])
                w, vec = np.linalg.eigh(A)
                k = n - 2
                v = vec[:, 0]
                rpk = float(rr[1:-1][np.argmax(v[k:] ** 2 + v[:k] ** 2)])
                nu = float(np.sqrt(-w[0])) if w[0] < 0 else 0.0
                rows.append(dict(r_cut=rc, omega2=float(w[0]), nu=nu, r_peak=rpk))
                say(f"     r_cut={rc:6.3f}: omega^2 = {w[0]:12.4f}  nu = {nu:9.3f}  nu*r_cut = {nu*rc:6.3f}  peak r = {rpk:.4f}")
            out["spectra"][l] = dict(R_minus=Rm, r_hi=r_hi, rows=rows)
        res[p] = out
    (OUT / "t37_large_p.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (LOGS / "t37_large_p_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
