"""T31 (b): regularity of the ring r = 0, x = cos(theta) = 0 for the Gurses-Gursey metric with m(r) = c3 r^3 + c5 r^5 (de Sitter core).
Coordinates (t, r, x = cos theta, phi): all components are rational. We compute the Ricci scalar and the Kretschmann scalar K = R_{abcd} R^{abcd}
(sequential index raising), control — Kerr: R = 0, K = 48 M^2 (r^2 - a^2 x^2)(Sigma^2 - 16 a^2 r^2 x^2)/Sigma^6.
Limits r -> 0: on the equator (x = 0), for x != 0, and along the directions x = k r. Run: python src/verification_03/T31_ring_curvature.py
"""
import json
import sys
import time
from pathlib import Path

import sympy as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
LOGS = Path(__file__).resolve().parents[2] / "logs" / Path(__file__).resolve().parent.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


t, r, x, ph = sp.symbols("t r x phi", real=True)
a, M, c3, c5, k, ell = sp.symbols("a M c3 c5 k ell", positive=True)
X = [t, r, x, ph]


def gg_metric(mexpr):
    Sig = r**2 + a**2 * x**2
    Del = r**2 - 2 * mexpr * r + a**2
    s2 = 1 - x**2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * mexpr * r / Sig)
    g[0, 3] = g[3, 0] = -2 * a * mexpr * r * s2 / Sig
    g[3, 3] = (r**2 + a**2 + 2 * a**2 * mexpr * r * s2 / Sig) * s2
    g[1, 1] = Sig / Del
    g[2, 2] = Sig / s2
    return g


def curvature(g):
    n = 4
    ginv = g.inv().applyfunc(sp.cancel)
    Gam = [[[sp.cancel(sum(ginv[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l])) for l in range(n)) / 2)
             for k in range(n)] for j in range(n)] for i in range(n)]
    pairs = [(k_, l_) for k_ in range(n) for l_ in range(k_ + 1, n)]
    Rup = {}   # R^i_{jkl}, k<l
    for i in range(n):
        for j in range(n):
            for (k_, l_) in pairs:
                e = sp.diff(Gam[i][j][l_], X[k_]) - sp.diff(Gam[i][j][k_], X[l_]) + sum(Gam[i][k_][s] * Gam[s][j][l_] - Gam[i][l_][s] * Gam[s][j][k_] for s in range(n))
                Rup[(i, j, k_, l_)] = sp.cancel(e)
    Ric = sp.zeros(n, n)
    for j in range(n):
        for l_ in range(n):
            Ric[j, l_] = sp.cancel(sum((Rup[(i, j, i, l_)] if i < l_ else (-Rup[(i, j, l_, i)] if l_ < i else 0)) for i in range(n)))
    Rs = sp.cancel(sum(ginv[j, l_] * Ric[j, l_] for j in range(n) for l_ in range(n)))
    # lower the first index: R_{ijkl} = g_{ie} R^e_{jkl}
    Rdn = {key: sp.cancel(sum(g[key[0], e] * Rup[(e,) + key[1:]] for e in range(n))) for key in Rup}
    # raise: R^{ij}_{kl} = g^{je} R^i_{ekl}
    R2 = {key: sp.cancel(sum(ginv[key[1], e] * Rup[(key[0], e, key[2], key[3])] for e in range(n))) for key in Rup}
    # R^{ijkl} = g^{ke} g^{lf} R^{ij}_{ef} (antisymmetric in ef)
    def R2v(i, j, e, f):
        if e == f: return 0
        return R2[(i, j, e, f)] if e < f else -R2[(i, j, f, e)]
    R4 = {}
    for i in range(n):
        for j in range(n):
            for (k_, l_) in pairs:
                R4[(i, j, k_, l_)] = sp.cancel(sum(ginv[k_, e] * ginv[l_, f] * R2v(i, j, e, f) for e in range(n) for f in range(n) if ginv[k_, e] != 0 and ginv[l_, f] != 0))
    K = sp.cancel(2 * sum(Rdn[key] * R4[key] for key in Rup))
    return Rs, K


def main():
    t0 = time.time()
    say("T31 (b): curvature of the Gurses-Gursey metric near the ring (sympy, coordinates t, r, x = cos theta, phi)")
    Rs_k, K_k = curvature(gg_metric(M))
    Sig = r**2 + a**2 * x**2
    K_ref = 48 * M**2 * (r**2 - a**2 * x**2) * (Sig**2 - 16 * a**2 * r**2 * x**2) / Sig**6
    say(f"  Kerr control: R = {sp.simplify(Rs_k)}; K - K_ref = {sp.simplify(K_k - K_ref)}   [{time.time()-t0:.0f} s]")
    t1 = time.time()
    m_poly = c3 * r**3 + c5 * r**5
    Rs, K = curvature(gg_metric(m_poly))
    Rs = sp.factor(Rs)
    say(f"  m = c3 r^3 + c5 r^5: Ricci scalar R = {Rs}   [{time.time()-t1:.0f} s]")
    out = dict(R=str(Rs))
    # Kretschmann limits
    K_eq = sp.cancel(K.subs(x, 0))
    K_eq_ser = sp.series(K_eq, r, 0, 4).removeO()
    say(f"  equator x = 0: K(r) = {sp.simplify(K_eq_ser)} + O(r^4)")
    K_eq_lim = sp.limit(K_eq, r, 0)
    say(f"    limit r -> 0 on the equator: K -> {sp.factor(K_eq_lim)}")
    K_x_lim = sp.limit(K, r, 0)
    say(f"  x != 0 fixed, r -> 0: K -> {sp.simplify(K_x_lim)}")
    K_dir = sp.cancel(K.subs(x, k * r))
    K_dir_lim = sp.factor(sp.limit(K_dir, r, 0))
    say(f"  along x = k r, r -> 0: K -> {K_dir_lim}")
    R_eq_lim = sp.limit(Rs.subs(x, 0), r, 0); R_x_lim = sp.limit(Rs, r, 0); R_dir_lim = sp.factor(sp.limit(sp.cancel(Rs.subs(x, k * r)), r, 0))
    say(f"  Ricci scalar: equator -> {R_eq_lim}; x != 0 -> {R_x_lim}; along x = k r -> {R_dir_lim}")
    # purely cubic core and spherical control
    K_eq0 = sp.factor(K_eq_lim.subs(c5, 0)); K_dir0 = sp.factor(K_dir_lim.subs(c5, 0))
    say(f"  c5 = 0 (pure de Sitter core): equator -> {K_eq0}; along x = k r -> {K_dir0}; at c3 = 1/(2 ell^2) equator -> {sp.simplify(K_eq0.subs(c3, 1/(2*ell**2)))} (spherical core: 24/ell^4)")
    # spherical limit a -> 0 of the full K as r -> 0
    K_a0 = sp.limit(sp.cancel(K.subs(a, 0)), r, 0)
    say(f"  a = 0, r -> 0: K -> {sp.factor(K_a0)}")
    # a-dependence of the limit on the equator: K(r=0, x=0) as a function of a?
    say(f"  does the limit on the equator depend on a: {sp.simplify(K_eq_lim)}; along x = k r on k and a: {K_dir_lim}")
    out.update(K_eq_lim=str(sp.factor(K_eq_lim)), K_x_lim=str(sp.simplify(K_x_lim)), K_dir_lim=str(K_dir_lim), R_eq=str(R_eq_lim), R_x=str(R_x_lim), R_dir=str(R_dir_lim), K_a0=str(sp.factor(K_a0)))
    json.dump(out, open(OUT / "T31_ring_curvature.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    (LOGS / "T31_ring_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
