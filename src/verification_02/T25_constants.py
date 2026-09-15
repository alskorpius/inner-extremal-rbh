"""T25: closed forms for the family constants (lambda, R-/M, R+/M, kappa+ of the base profile; lambda_max from T15).
Steps: (1) values and actual precision (grid LU 60001 points in ln u, interpolation, merger of stationary points of H over the grid);
(2) mpmath.identify against the constants pi, e, sqrt2, sqrt3 and a search over integer polynomials of degree <= 3, |c| <= 12, with tolerance = precision;
(3) control: the same search on 10 random numbers of the same order; a match is accepted only if its complexity is lower than the best random one.
Run: python src/verification_02/T25_constants.py
"""
import itertools
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "polar_qnm"))
sys.path.insert(0, str(HERE.parents[0] / "stability"))
sys.argv = [sys.argv[0], "0.05"]
import qnm_band as qb  # noqa: E402
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def base_constants():
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    s1s, u_s, info = trm.find_merge_first(sigma_min=1.0, base=base0)
    fam = qb.build_family(base0, 1.0)
    bg = qb.FamilyBG(fam)
    lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    f = lambda R: 1 - 2 * bg.fields(np.array([R]))["m"][0] / R
    # R+: brentq; R-: minimum |f| (triple root -- tangency) + all sign changes
    Rp = brentq(f, 1.0, 3.0, xtol=1e-14)
    rr = np.linspace(0.6, 0.76, 400001)
    fv = np.array([f(x) for x in rr])
    i = int(np.argmin(np.abs(fv)))
    Rm = float(rr[i])
    sc = rr[np.nonzero(np.sign(fv[:-1]) * np.sign(fv[1:]) < 0)[0]]
    h = 1e-6
    kp = (f(Rp + h) - f(Rp - h)) / (4 * h)
    km = (f(Rm + h) - f(Rm - h)) / (4 * h)
    grid = float(np.diff(trm.LU)[0])
    return dict(s1=s1s, u_s=u_s, lam=lam, R_minus=Rm, R_plus=Rp, kappa_plus=kp, kappa_minus=km, sign_changes=[float(x) for x in sc],
                grid_dlnu=grid, R1=fam.R1, rho_c=fam.rho_c, note_precision="the merger point of the stationary points of H is set by the grid in ln u (step %.1e), so R-, lambda and R+ are reliable to ~4 significant figures; kappa+ ~4 digits" % grid)


def poly_search(x, tol, deg_max=3, cmax=12):
    """roots of polynomials sum c_k x^k = 0 with integer c, |c| <= cmax, leading term != 0, reduced (gcd = 1, leading > 0); complexity = sum|c| + deg."""
    best = None
    rng = range(-cmax, cmax + 1)
    for deg in range(1, deg_max + 1):
        for cs in itertools.product(rng, repeat=deg + 1):
            if cs[-1] <= 0:
                continue
            if np.gcd.reduce([abs(c) for c in cs if c != 0]) != 1:
                continue
            val = sum(c * x**k for k, c in enumerate(cs))
            dval = sum(k * c * x ** (k - 1) for k, c in enumerate(cs) if k > 0)
            if dval == 0:
                continue
            dist = abs(val / dval)   # estimate of distance to the nearest root
            if dist < tol:
                comp = sum(abs(c) for c in cs) + deg
                if best is None or comp < best[0]:
                    best = (comp, cs, dist)
    return best


def ident(x, tol):
    mp.mp.dps = 15
    res = mp.identify(mp.mpf(x), ["pi", "e", "sqrt(2)", "sqrt(3)"], tol=tol, maxcoeff=12)
    return res


def complexity_str(expr):
    if expr is None:
        return 99
    s = expr.replace(" ", "")
    return sum(ch.isdigit() for ch in s) + s.count("pi") * 2 + s.count("e") + s.count("sqrt") * 2 + s.count("/") + s.count("*") + s.count("+") + s.count("-")


def main():
    say("T25: closed forms for the family constants")
    c = base_constants()
    say(f"  base profile: s1* = {c['s1']:.12f}, u_s = {c['u_s']:.8f}, R1 = {c['R1']:.8f}, rho_c = {c['rho_c']:.8f}")
    say(f"  lambda = {c['lam']:.8f}, R-/M = {c['R_minus']:.8f} (sign changes of f: {c['sign_changes']}), R+/M = {c['R_plus']:.10f}, kappa+ = {c['kappa_plus']:.8f}, kappa- = {c['kappa_minus']:.2e}")
    say(f"  precision: {c['note_precision']}")
    t15 = {}
    for p in (OUT.parents[0] / "verification_01").glob("T15_*.json"):
        try:
            t15[p.name] = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    lam_max = 1.066
    say(f"  lambda_max (T15, search for the maximum over 8 parameters, precision ~3 digits): {lam_max}")
    targets = [("lambda", c["lam"], 1e-4), ("R-/M", c["R_minus"], 2e-4), ("R+/M", c["R_plus"], 1e-4), ("kappa+", c["kappa_plus"], 1e-4), ("lambda_max", lam_max, 2e-3)]
    say("\n  number | value | tolerance | mpmath.identify | polynomial (complexity, coefficients) | control (best complexity of 10 random) | verdict")
    rng = np.random.default_rng(7)
    rows = []
    for name, x, tol in targets:
        idn = ident(x, tol)
        pl = poly_search(x, tol)
        # control
        rand = rng.uniform(0.5 * x, 1.5 * x, 10)
        rand_best = min((poly_search(float(y), tol) or (99, None, None))[0] for y in rand)
        rand_id = sum(1 for y in rand if ident(float(y), tol))
        verdict = "match not simpler than random -- not accepted" if (pl is None or pl[0] >= rand_best) else "simpler than random -- requires explanation"
        rows.append(dict(name=name, x=x, tol=tol, identify=idn, poly=None if pl is None else dict(complexity=pl[0], coeffs=list(pl[1]), dist=pl[2]), random_best=rand_best, random_identify_hits=rand_id, verdict=verdict))
        say(f"  {name:10s} | {x:.6f} | {tol:.0e} | {idn} | {None if pl is None else (pl[0], pl[1])} | {rand_best} (identify triggers on {rand_id}/10 random) | {verdict}")
    say("\n  lambda_max: family boundary -- maximum of lambda over 8 parameters of the manual parametrization (sigmoids + Gaussian dip); analytically extremality = "
        "a fourfold root f = f' = f'' = f''' = 0 (4 conditions on the profile), which for the given parametrization is an equation on the parameters, not on the number; "
        "there is no algebraic equation for lambda_max.")
    json.dump(dict(constants=c, rows=rows), open(OUT / "T25_constants.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    (LOGS / "T25_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
