"""T15. Upper bound of the family of monotone profiles with a triple root: a limit of the class or a limit of the search?
Family src/stability/triple_root_monotone.py: sigma(u) = s1 S(u;u1,w1) - d exp(-((ln u - ln u2)/w2)^2) + (s_end - s1) S(u;u3,w3),
d = s1 - sigma_min; the triple root is the merging of the first maximum and minimum of H = 2 mI/u at s1 = s1* (bisection search over s1 in [2.02, s1_hi]).
lambda = ell/M = R1 sqrt(3/(2 scale)), R1 = 1/(Itot scale), scale = 1/H(u_s). Identity: lambda = (R-/M) sqrt(3 s(R-)), s = rho/rho_c
(from 2m(R-) = R- and <rho> = 3 rho(R-) at the triple root).
Run: python src/verification_01/T15_lambda_bound.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "stability"))
sys.path.insert(0, str(ROOT / "src" / "approach_map"))
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def find_merge(base, sigma_min, s1_lo=2.02, s1_hi=8.0):
    """Merging of the first pair of stationary points of H as s1 grows; returns (s1*, u_s, kw, reason)."""
    def pts(s1):
        kw = dict(base, s1=s1)
        _, _, _, _, Hp = trm.profile(s1 - sigma_min, **kw)
        return trm.stationary_points(Hp)[0], kw
    p_lo, _ = pts(s1_lo)
    if len(p_lo) >= 3:
        return None, None, None, "already 3 stationary points at s1 = 2.02 (the dip is too deep/wide)"
    # grid over s1: the first value at which the number of stationary points >= 3 (at larger s1 the count may change again)
    grid = np.linspace(s1_lo, s1_hi, 60)
    hi = None
    for a, b in zip(grid[:-1], grid[1:]):
        if len(pts(b)[0]) >= 3:
            lo, hi = a, b
            break
    if hi is None:
        return None, None, None, f"no 3 stationary points of H up to s1 = {s1_hi} (S-shape of H does not appear)"
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        p, _ = pts(mid)
        if len(p) >= 3:
            hi = mid
        else:
            lo = mid
    p_hi, kw = pts(hi)
    p_lo, _ = pts(lo)
    merged_first = abs(p_lo[-1] - p_hi[2]) < 0.1 * p_hi[2] if len(p_lo) else True
    if not merged_first:
        return None, None, None, "the outer pair merges (the triple root would be the outer horizon)"
    return 0.5 * (lo + hi), float(0.5 * (p_hi[0] + p_hi[1])), kw, "ok"


def characterize(base, sigma_min, s1_hi=8.0):
    s1, u_s, kw, reason = find_merge(base, sigma_min, s1_hi=s1_hi)
    if s1 is None:
        return dict(ok=False, reason=reason, base=dict(base), sigma_min=sigma_min)
    d = s1 - sigma_min
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    U, LU = trm.U, trm.LU
    Itot = mI[-1]
    Hs = float(np.interp(u_s, U, H))
    scale = 1.0 / Hs
    R1 = 1.0 / (Itot * scale)
    lam = R1 * np.sqrt(1.5 / scale)
    f = 1 - scale * H
    # outer horizon: first crossing of f = 0 from below at u > u_s
    k = np.nonzero((U > u_s * 1.02) & (f[:-1] < 0) & (f[1:] >= 0))[0] if False else None
    mask = U > u_s * 1.02
    fu = f[mask]; Uu = U[mask]; Hpu = Hp[mask]
    idx = np.nonzero((fu[:-1] < 0) & (fu[1:] >= 0))[0]
    if idx.size == 0:
        return dict(ok=False, reason="no outer horizon (f < 0 does not return to zero)", base=dict(base), sigma_min=sigma_min, lam=float(lam))
    i = idx[0]
    u_p = float(Uu[i] + (Uu[i + 1] - Uu[i]) * (-fu[i]) / (fu[i + 1] - fu[i]))
    Hp_p = float(np.interp(u_p, Uu, Hpu))
    kappa_p = -scale * Hp_p / (2 * R1)
    m_p = float(np.interp(u_p, U, mI)) / Itot
    # check of the triple root: f, f', f'' at u_s
    fp = np.gradient(f, U); fpp = np.gradient(fp, U)
    f_s, fp_s, fpp_s = float(np.interp(u_s, U, f)), float(np.interp(u_s, U, fp)) / R1, float(np.interp(u_s, U, fpp)) / R1**2
    s_s = float(np.interp(u_s, U, s))
    sig_min = float(sig.min())
    pperp = s * (sig / 2 - 1)
    # tail: fraction of mass outside 3M and 10M
    R = U * R1
    m_3 = float(np.interp(3.0, R, mI)) / Itot
    return dict(ok=True, reason="ok", base=dict(base), sigma_min=sigma_min, s1=float(s1), lam=float(lam), R_minus=float(u_s * R1), R_plus=float(u_p * R1),
                kappa_plus=float(kappa_p), T_ratio=float(4 * kappa_p), halo=float(1 - m_p), m_3M=m_3, f_s=f_s, fp_s=fp_s, fpp_s=fpp_s,
                s_at_minus=s_s, identity_lam=float(u_s * R1 * np.sqrt(3 * s_s)), sigma_min_actual=sig_min, NEC_WEC=bool(sig_min >= -1e-9),
                pperp_max=float(pperp.max()), min_R_minus_over_ell=float(u_s * R1 / lam))


def fmt(r, tag):
    if not r["ok"]:
        return f"  {tag}: NO — {r['reason']}" + (f" (lam would be {r['lam']:.3f})" if "lam" in r else "")
    return (f"  {tag}: lam = {r['lam']:.4f} (identity {r['identity_lam']:.4f}); R- = {r['R_minus']:.3f}, R+ = {r['R_plus']:.3f}, kappa+ = {r['kappa_plus']:.4f}, "
            f"T_H/T_S = {r['T_ratio']:.3f}, halo = {r['halo']:.4f}, m(3M)/M = {r['m_3M']:.6f}; s(R-) = {r['s_at_minus']:.3f}, R-/ell = {r['min_R_minus_over_ell']:.3f}; "
            f"sigma_min = {r['sigma_min_actual']:+.3f}, NEC/WEC {r['NEC_WEC']}, p_perp,max/rho_c = {r['pperp_max']:.2f}; |f,f',f''|(R-) = {abs(r['f_s']):.0e}, {abs(r['fp_s']):.0e}, {abs(r['fpp_s']):.0e}; s1* = {r['s1']:.3f}")


def main():
    t0 = time.time()
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    say("T15: upper bound of lambda = ell/M for the family of monotone profiles with a triple root")
    rows = []
    r0 = characterize(base0, 1.0); rows.append(("base", r0)); say(fmt(r0, "base"))
    say("\n1. One-dimensional continuations from the base")
    one_d = [("s_end", [8, 48, 100, 200, 500, 2000]), ("u3", [12, 8, 7, 6, 5.5, 5, 4.5, 4.2]), ("u2", [6, 3, 2.5, 2, 1.5]), ("w3", [0.3, 0.05, 0.02, 0.01]),
             ("u1", [2, 0.5, 0.3, 0.15]), ("w1", [0.5, 0.1, 0.05]), ("w2", [0.6, 0.2, 0.15]), ("sigma_min", [1.5, 1.9, 0.5, 0.2, 0.0])]
    for name, vals in one_d:
        for v in vals:
            if name == "sigma_min":
                r = characterize(base0, v)
            else:
                r = characterize(dict(base0, **{name: v}), 1.0)
            rows.append((f"{name} = {v}", r)); say(fmt(r, f"{name} = {v}"))
    # 2. random search
    say("\n2. Random search (300 points) over all parameters")
    rng = np.random.default_rng(7)
    best = max([r for _, r in rows if r["ok"]], key=lambda r: r["lam"])
    found = []
    for i in range(300):
        u2 = float(rng.uniform(1.5, 8.0)); u3 = float(u2 * rng.uniform(1.15, 4.0))
        base = dict(u1=float(rng.uniform(0.2, 2.5)), u2=u2, u3=u3, w1=float(rng.uniform(0.08, 0.5)), w2=float(rng.uniform(0.12, 0.8)),
                    w3=float(rng.uniform(0.02, 0.4)), s_end=float(np.exp(rng.uniform(np.log(6), np.log(500)))))
        smin = float(rng.uniform(0.0, 1.9))
        try:
            r = characterize(base, smin)
        except Exception as e:  # noqa: BLE001
            continue
        if r["ok"]:
            found.append(r)
            if r["lam"] > best["lam"]:
                best = r
    lams = np.array([r["lam"] for r in found])
    say(f"  triple roots found: {len(found)} out of 300; lam from {lams.min():.3f} to {lams.max():.3f}; median {np.median(lams):.3f}; all NEC/WEC: {all(r['NEC_WEC'] for r in found)}")
    say(fmt(best, "best"))
    # 3. local maximization of lambda from the best point (Nelder–Mead over log-parameters)
    say("\n3. Local maximization of lambda (Nelder–Mead from the best point)")
    keys = ["u1", "u2", "u3", "w1", "w2", "w3", "s_end"]
    x0 = np.array([np.log(best["base"][k]) for k in keys] + [best["sigma_min"]])

    def obj(x):
        base = {k: float(np.exp(v)) for k, v in zip(keys, x[:7])}
        smin = float(np.clip(x[7], 0.0, 1.95))
        if base["u3"] <= base["u2"] * 1.05 or base["s_end"] < 2.5:
            return 10.0
        try:
            r = characterize(base, smin)
        except Exception:  # noqa: BLE001
            return 10.0
        if not r["ok"] or not r["NEC_WEC"]:
            return 10.0
        return -r["lam"]
    res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-3, fatol=1e-5))
    base_opt = {k: float(np.exp(v)) for k, v in zip(keys, res.x[:7])}; smin_opt = float(np.clip(res.x[7], 0.0, 1.95))
    r_opt = characterize(base_opt, smin_opt)
    say(f"  parameters: {', '.join(f'{k} = {v:.3g}' for k, v in base_opt.items())}, sigma_min = {smin_opt:.3g}")
    say(fmt(r_opt, "optimum"))
    if r_opt["ok"] and r_opt["lam"] > best["lam"]:
        best = r_opt
    # 4. what breaks past the boundary: continuation of the best point along each parameter's direction of largest lambda growth (from the 1D scans)
    say("\n4. What breaks past the boundary: continuations of the best point along each parameter")
    breaks = []
    for k in keys:
        for fac in (1.3, 1.6, 2.0, 3.0):
            for direction in (+1, -1):
                b = dict(best["base"]); b[k] = b[k] * (fac if direction > 0 else 1 / fac)
                if b["u3"] <= b["u2"] * 1.05:
                    continue
                r = characterize(b, best["sigma_min"])
                breaks.append(dict(param=k, factor=fac if direction > 0 else 1 / fac, ok=r["ok"], reason=r["reason"], lam=r.get("lam")))
    for smin in (0.0, 0.5, 1.5, 1.9):
        r = characterize(best["base"], smin)
        breaks.append(dict(param="sigma_min", factor=smin, ok=r["ok"], reason=r["reason"], lam=r.get("lam")))
    for b in breaks:
        say(f"  {b['param']} x {b['factor']:.3g}: {'lam = %.3f' % b['lam'] if b['ok'] else 'NO — ' + b['reason']}")
    # 5. companion-paper synthesis points
    say("\n5. companion-paper synthesis points: lambda = 0.45, 0.58, 0.77")
    all_ok = [r for _, r in rows if r["ok"]] + found + ([r_opt] if r_opt["ok"] else [])
    lam_max = max(r["lam"] for r in all_ok)
    for target in (0.45, 0.58, 0.77):
        near = min(all_ok, key=lambda r: abs(r["lam"] - target))
        say(f"  lambda = {target}: {'yes' if lam_max >= target else 'NO — family maximum %.3f' % lam_max}; closest found lam = {near['lam']:.3f} (R- = {near['R_minus']:.3f}, R+ = {near['R_plus']:.3f}, halo {near['halo']:.3f})")
    # 6. analytic estimate: lam = (R-/M) sqrt(3 s(R-)), s(R-) <= 1/3 => lam < R-/M; R- < R+ <= 2M
    say("\n6. Analytical estimate: at the triple root <rho>(R-) = 3 rho(R-) and 2m(R-) = R-, hence lam = ell/M = (R-/M) sqrt(3 rho(R-)/rho_c) <= R-/M (equality only for a homogeneous core with a jump), "
        "and R- < R+ <= 2M: lam < 2. Found: max lam = %.3f at R-/M = %.3f, s(R-) = %.3f (3s = %.3f)." % (lam_max, best["R_minus"], best["s_at_minus"], 3 * best["s_at_minus"]))
    say(f"  across all found: R-/ell from {min(r['min_R_minus_over_ell'] for r in all_ok):.3f} to {max(r['min_R_minus_over_ell'] for r in all_ok):.3f} (analytically >= 1); "
        f"3 s(R-) from {min(3*r['s_at_minus'] for r in all_ok):.3f} to {max(3*r['s_at_minus'] for r in all_ok):.3f} (<= 1); R-/M from {min(r['R_minus'] for r in all_ok):.3f} to {max(r['R_minus'] for r in all_ok):.3f}")
    # 7. QNM for the best point (axial l = 2, time domain) — imported from polar_qnm without running its scripts
    say("\n7. QNM (axial, l = 2, time domain) for the point with maximum lambda")
    try:
        sys.path.insert(0, str(ROOT / "src" / "polar_qnm"))
        sys.argv = [sys.argv[0], "0.05"]
        import qnm_band as qb  # noqa: E402
        import axial_td as ax  # noqa: E402
        import polar_qnm as pq  # noqa: E402
        s1, u_s, kw, _ = find_merge(best["base"], best["sigma_min"])
        sig, s, mI, H, Hp = trm.profile(s1 - best["sigma_min"], **kw)
        scale = 1.0 / float(np.interp(u_s, trm.U, H))
        fam = trm.MonotoneFamily(s1 - best["sigma_min"], scale, kw=kw)
        bg = qb.FamilyBG(fam); bgS = pq.Background(None)
        tS, yS = ax.evolve_axial(bgS, 2); wS = pq.matrix_pencil(tS, yS, 80.0, 200.0)
        t, y = ax.evolve_axial(bg, 2); w = pq.matrix_pencil(t, y, 80.0, 200.0)
        rr = np.linspace(3.0, 60.0, 500); dm = float(np.max(np.abs(bg.fields(rr)["m"] - 1.0)))
        say(f"  lam = {best['lam']:.3f}: dRe/Re = {w[0]/wS[0]-1:+.2e}, dIm/Im = {w[1]/wS[1]-1:+.2e}; max|m(r) - M| at r >= 3M: {dm:.1e} (the shadow is Schwarzschild's when m = M at the photon sphere)")
        best["qnm_l2"] = dict(dRe=w[0] / wS[0] - 1, dIm=w[1] / wS[1] - 1, m_dev_3M=dm)
    except Exception as e:  # noqa: BLE001
        say(f"  QNM not computed: {e}")
    json.dump(dict(rows=[dict(tag=t, **r) for t, r in rows], random_found=found, best=best, breaks=breaks, lam_max=lam_max), open(OUT / "T15_lambda_bound.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T15_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
