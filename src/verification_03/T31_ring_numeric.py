"""T31 (b), fast variant: the same GG-metric curvatures, but with numeric (rational) parameters a, c3, c5 — expressions are rational
only in (r, x), so sympy handles them quickly. Kerr control at M = 1, a = 3/5. Limits r -> 0: equator x = 0, x fixed,
along x = k r (k = 1/2, 1, 2). Run: python src/verification_03/T31_ring_numeric.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import T31_ring_curvature as rc  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


r, x = rc.r, rc.x


def main():
    t0 = time.time()
    say("T31 (b) numeric-symbolic: rational parameters, curvatures are rational functions of (r, x)")
    a = sp.Rational(3, 5)
    Rs_k, K_k = rc.curvature(rc.gg_metric(sp.Integer(1)).subs(rc.a, a))
    Sig = r**2 + a**2 * x**2
    K_ref = 48 * (r**2 - a**2 * x**2) * (Sig**2 - 16 * a**2 * r**2 * x**2) / Sig**6
    say(f"  Kerr control (M = 1, a = 3/5): R = {sp.simplify(Rs_k)}, K - K_ref = {sp.simplify(K_k - K_ref)}   [{time.time()-t0:.0f} s]")
    out = {}
    for (c3v, c5v, tag) in ((sp.Rational(1, 2), 0, "m = r^3/2 (ell = 1, pure dS core)"), (sp.Rational(1, 2), sp.Rational(-1, 4), "m = r^3/2 - r^5/4"), (sp.Rational(1, 2), sp.Rational(1, 4), "m = r^3/2 + r^5/4")):
        for av in (sp.Rational(3, 5), sp.Rational(9, 10)):
            t1 = time.time()
            m_poly = c3v * r**3 + c5v * r**5
            Rs, K = rc.curvature(rc.gg_metric(m_poly).subs(rc.a, av))
            Rs = sp.factor(Rs)
            K_eq = sp.cancel(K.subs(x, 0))
            lim_eq = sp.limit(K_eq, r, 0)
            lim_x = sp.limit(K, r, 0)              # x fixed (symbol)
            lims_dir = {kv: sp.limit(sp.cancel(K.subs(x, kv * r)), r, 0) for kv in (sp.Rational(1, 2), 1, 2)}
            R_eq = sp.limit(Rs.subs(x, 0), r, 0); R_x = sp.limit(Rs, r, 0); R_dir = {kv: sp.limit(sp.cancel(Rs.subs(x, kv * r)), r, 0) for kv in (sp.Rational(1, 2), 1, 2)}
            ser = sp.series(K_eq, r, 0, 3).removeO()
            say(f"  {tag}, a = {av}: R = {Rs}")
            say(f"      K on the equator: {sp.simplify(ser)} + O(r^3); limit r->0: {lim_eq}; for x != 0: {sp.simplify(lim_x)}; along x = k r: {lims_dir}")
            say(f"      R: equator -> {R_eq}; x != 0 -> {R_x}; along x = k r -> {R_dir}   [{time.time()-t1:.0f} s]")
            out[f"{tag}|a={av}"] = dict(R=str(Rs), K_eq=str(lim_eq), K_x=str(sp.simplify(lim_x)), K_dir={str(kv): str(v) for kv, v in lims_dir.items()}, R_eq=str(R_eq), R_x=str(R_x), R_dir={str(kv): str(v) for kv, v in R_dir.items()})
    json.dump(out, open(OUT / "T31_ring_numeric.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    (LOGS / "T31_ring_numeric_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
