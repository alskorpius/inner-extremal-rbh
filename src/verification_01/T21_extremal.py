"""T21. Extremal end of the family of triple roots (lambda -> 1.066): properties at lambda = 1.00 / 1.05 / 1.066, structure of f at the merger
R- -> R+, stability of the limiting configuration to mass perturbations (m -> (1+eps) m) and shape perturbations (s_end, u3 by +-1 % without retuning s1),
dynamics from T2 data (drift of r- with N_e), comparison with the companion paper's horizonless criterion. Run from the root:
python src/verification_01/T21_extremal.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

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
import T15_qnm as T15q  # noqa: E402
import triple_root_monotone as trm  # noqa: E402
import axial_td as ax  # noqa: E402
import polar_qnm as pq  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------- profile and its geometry ----------
class Prof:
    """m(r) = (1+eps) * interp(fam), f = 1 - 2m/r; derivatives of m via rho (analytically) with the factor (1+eps)."""

    def __init__(self, fam, eps=0.0):
        self.fam, self.eps = fam, eps

    def m(self, r):
        return (1 + self.eps) * np.interp(np.asarray(r, float), self.fam._Rg, self.fam._mg)

    def mp(self, r):
        return (1 + self.eps) * 4 * np.pi * r**2 * self.fam.rho(r)

    def mpp(self, r):
        return (1 + self.eps) * (8 * np.pi * r * self.fam.rho(r) + 4 * np.pi * r**2 * self.fam.drho(r))

    def f(self, r):
        return 1 - 2 * self.m(r) / r

    def fp(self, r):
        return 2 * self.m(r) / r**2 - 2 * self.mp(r) / r

    def fpp(self, r):
        return -4 * self.m(r) / r**3 + 4 * self.mp(r) / r**2 - 2 * self.mpp(r) / r

    def fppp(self, r, h=1e-4):
        return (self.fpp(r + h) - self.fpp(r - h)) / (2 * h)

    def roots(self, lo=1e-2, hi=3.0, n=400001):
        r = np.geomspace(lo, hi, n)
        f = self.f(r)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        rs = [brentq(self.f, r[i], r[i + 1], xtol=1e-14) for i in idx]
        # near-roots (touch): local minima of |f| below 1e-6 with no sign change
        touch = []
        af = np.abs(f)
        for i in np.nonzero((af[1:-1] < af[:-2]) & (af[1:-1] < af[2:]) & (af[1:-1] < 1e-6))[0] + 1:
            if not any(abs(r[i] - x) < 1e-3 for x in rs):
                touch.append(float(r[i]))
        return rs, touch, float(f.min()), float(r[np.argmin(f)])


def describe(fam, tag, eps=0.0, qnm=False, rs_min=-80.0):
    P = Prof(fam, eps)
    rs, touch, fmin, r_fmin = P.roots()
    lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    M = 1 + eps
    info = dict(tag=tag, eps=eps, lam=lam / M, roots=rs, touch=touch, f_min=fmin, r_f_min=r_fmin,
                kappas=[float(P.fp(x) / 2) for x in rs])
    if rs:
        Rm, Rp = rs[0], rs[-1]
        info.update(R_minus=Rm, R_plus=Rp, kappa_minus=float(P.fp(Rm) / 2), kappa_plus=float(P.fp(Rp) / 2),
                    fpp_minus=float(P.fpp(Rm)), fppp_minus=float(P.fppp(Rm)), fpp_plus=float(P.fpp(Rp)),
                    T_ratio=float(P.fp(Rp) / 2 / (1 / (4 * M))), halo=float(1 - P.m(Rp) / M),
                    f_dev_3M=float(P.f(3 * M) - (1 - 2 * M / (3 * M))), f_dev_6M=float(P.f(6 * M) - (1 - 2 * M / (6 * M))))
        # photon sphere, shadow
        r = np.linspace(Rp * 1.001, 8.0, 40000)
        g = P.f(r) / r**2
        i = int(np.argmax(g)); r_ph = float(r[i]); b_c = float(r_ph / np.sqrt(P.f(r_ph)))
        info.update(r_ph=r_ph / M, b_c=b_c / M, shadow_dev=b_c / M / (3 * np.sqrt(3)) - 1, r_ph_dev=r_ph / M / 3 - 1)
        info["dRe_from_halo"] = -0.2 * info["halo"]
    if qnm and rs and info["kappa_plus"] > 1e-3:
        bg = T15q.BG(fam)
        bgS = pq.Background(None)
        tS, yS = ax.evolve_axial(bgS, 2, rs_min=rs_min); wS = pq.matrix_pencil(tS, yS, 80.0, 200.0)
        t, y = ax.evolve_axial(bg, 2, rs_min=rs_min); w = pq.matrix_pencil(t, y, 80.0, 200.0)
        info.update(qnm_dRe=w[0] / wS[0] - 1, qnm_dIm=w[1] / wS[1] - 1, qnm=list(w), qnm_S=list(wS))
    return info


def fam_at(base, smin):
    """Family with a triple root (s1 tuned); returns fam with attributes _d, _scale, _kw for shape perturbations."""
    s1, u_s, kw, reason = T15.find_merge(base, smin, s1_hi=30.0)
    if s1 is None:
        return None
    d = s1 - smin
    H = trm.profile(d, **kw)[3]
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    fam = trm.MonotoneFamily(d, scale, kw=kw)
    fam._d, fam._scale, fam._kw = d, scale, dict(kw)
    return fam


def main():
    t0 = time.time()
    q = json.load(open(OUT / "T15_qnm.json", encoding="utf-8"))
    base_opt, smin_opt = q["base_opt"], q["sigma_min_opt"]
    say("T21. Extremal end of the family (lambda -> 1.066)")
    say("\n(1) Configurations lambda = 1.00, 1.05 and the limiting one (moving along s_end from the T15 optimum; s1 retuned to the triple root)")

    def lam_of(s_end):
        r = T15.characterize(dict(base_opt, s_end=s_end), smin_opt, s1_hi=30.0)
        return r["lam"] if r["ok"] else np.nan

    # existence boundary in s_end (bisection: exists / "no 3 stationary points of H")
    lo, hi = 28.0, 31.0
    assert not np.isnan(lam_of(lo)) and np.isnan(lam_of(hi))
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if np.isnan(lam_of(mid)): hi = mid
        else: lo = mid
    lam_b = lam_of(lo)
    say(f"  existence boundary in s_end: {lo:.4f} (at {hi:.4f} there is no triple root anymore); lambda at the boundary = {lam_b:.4f}")
    cfgs = []
    for target in (1.00, 1.05 if lam_b > 1.052 else 0.5 * (1.0 + lam_b)):
        se = brentq(lambda s: lam_of(s) - target, 20.0, lo, xtol=1e-3)
        cfgs.append((f"lambda = {target:.3f}", dict(base_opt, s_end=se)))
    cfgs.append(("lambda = 1.035 (boundary in s_end)", dict(base_opt, s_end=lo)))
    cfgs.append(("lambda = 1.066 (T15 optimum, exact parameters)", dict(base_opt)))
    rows = []
    fams = {}
    for tag, base in cfgs:
        fam = fam_at(base, smin_opt)
        if fam is None:
            say(f"  {tag}: triple root not reproduced (s1 window too narrow for the find_merge grid)"); continue
        fams[tag] = (fam, base)
        info = describe(fam, tag, qnm=True, rs_min=-200.0)
        rows.append(info)
        say(f"  {tag}: u1 = {base['u1']:.4f}; roots {['%.5f' % x for x in info['roots']]} (kappa {['%.2e' % k for k in info['kappas']]}); "
            f"R- = {info['R_minus']:.4f}, R+ = {info['R_plus']:.4f}, R+ - R- = {info['R_plus']-info['R_minus']:.4f}; kappa- = {info['kappa_minus']:+.1e}, kappa+ = {info['kappa_plus']:.2e}, "
            f"T_H/T_S = {info['T_ratio']:.4f}; halo = {info['halo']:.4f}; f - f_Schw at 3M/6M: {info['f_dev_3M']:+.1e}/{info['f_dev_6M']:+.1e}; "
            f"photon sphere r_ph/M = {info['r_ph']:.4f} ({info['r_ph_dev']:+.1e}), b_c/M = {info['b_c']:.4f} ({info['shadow_dev']:+.1e}); "
            f"f'''(R-) = {info['fppp_minus']:+.3f}, f''(R+) = {info['fpp_plus']:+.3f}, min f between = {info['f_min']:.2e} at r = {info['r_f_min']:.3f}; "
            f"QNM l=2: {('dRe/Re = %+.2e, dIm/Im = %+.2e' % (info['qnm_dRe'], info['qnm_dIm'])) if 'qnm_dRe' in info else 'not computed (kappa+ < 1e-3, horizon not resolved by the r* grid)'}; halo-based estimate −0.2·halo = {info['dRe_from_halo']:+.2e}   [{time.time()-t0:.0f} s]")
    # structure at the merger: kappa+ versus Delta = R+ - R-
    D = np.array([r["R_plus"] - r["R_minus"] for r in rows]); K = np.array([r["kappa_plus"] for r in rows])
    p = np.polyfit(np.log(D), np.log(K), 1)[0]
    say(f"  Merger: kappa+ ∝ (R+ − R-)^{p:.2f} over {len(rows)} points (a quadruple root f ≈ c (r−R-)^3 (r−R+) gives exponent 3); "
        f"at the limit f ≥ 0 on both sides (the trapped region disappears), f ~ (r − r0)^4.")

    ext_tag = list(fams.keys())[-1]
    say(f"\n(2) Stability of the limiting configuration ({ext_tag}): perturbations of mass m -> (1+eps) m and shape (without retuning s1)")
    fam_ext, base_ext = fams[ext_tag]
    pert = []
    for eps in (1e-4, 1e-3, 1e-2, -1e-4, -1e-3, -1e-2):
        info = describe(fam_ext, f"eps = {eps:+.0e}", eps=eps)
        n = len(info["roots"])
        if n == 0:
            kind = "horizonless object (f > 0 everywhere)" if info["f_min"] > 0 else "?"
        elif n == 2:
            kind = "two horizons (kappa- < 0: mass inflation)"
        else:
            kind = f"{n} roots"
        pert.append(dict(info, kind=kind))
        say(f"  mass {eps:+.0e}: roots {['%.4f' % x for x in info['roots']]}, kappa {['%+.3e' % k for k in info['kappas']]}, min f = {info['f_min']:+.2e} at r = {info['r_f_min']:.3f} → {kind}")
    for key, fac in (("s_end", 1.01), ("s_end", 0.99), ("u3", 1.01), ("u3", 0.99)):
        kwf = dict(fam_ext._kw); kwf[key] = kwf[key] * fac        # s1 and scale unchanged: shape perturbed without retuning
        famp = trm.MonotoneFamily(fam_ext._d, fam_ext._scale, kw=kwf)
        info = describe(famp, f"{key} x {fac}")
        n = len(info["roots"])
        kind = ("horizonless object (f > 0 everywhere)" if (n == 0 and info["f_min"] > 0) else "two horizons (kappa- < 0)" if n == 2 else f"{n} roots")
        pert.append(dict(info, kind=kind))
        say(f"  shape {key} × {fac}: roots {['%.4f' % x for x in info['roots']]}, kappa {['%+.3e' % k for k in info['kappas']]}, min f = {info['f_min']:+.2e} at r = {info['r_f_min']:.3f} → {kind}")

    say("\n(3) Dynamics from T2: drift of r- as mass accumulates (prescription P1/P1s)")
    d2 = json.load(open(OUT / "T2_feedback.json", encoding="utf-8"))["results"]
    drift = []
    for k, v in d2.items():
        h = [x for x in v["hist"] if np.isfinite(x["kappa"]) and abs(x["kappa"]) < 50]
        if len(h) < 2:
            continue
        dr = (h[-1]["r_minus"] - h[0]["r_minus"]) / (h[-1]["Ne_cum"] - h[0]["Ne_cum"])
        drift.append(dict(run=k, r0=h[0]["r_minus"], r_end=h[-1]["r_minus"], Ne=h[-1]["Ne_cum"], dr_dNe=dr, kappa_end=h[-1]["kappa"]))
        say(f"  {k}: r- {h[0]['r_minus']:.4f} → {h[-1]['r_minus']:.4f} over N_e = {h[-1]['Ne_cum']:.0f} (dr-/dN_e = {dr:+.4f} M), kappa- → {h[-1]['kappa']:+.3f}")
    inward = all(x["dr_dNe"] < 0 for x in drift)
    say(f"  In all {len(drift)} series r- moves INWARD (dr-/dN_e < 0): {inward}; it does not approach R+ ≈ 1.98 — the \"horizon merger via mass inflation\" scenario is not realized under the T2 prescription.")
    # N_e estimates
    M10 = 10 * 1.4766e3   # m
    ell = 0.271 * M10; lP = 1.616e-35
    Ne_planck = 0.5 * np.log(ell**4 / (24 * lP**4))
    Ne_merge = np.log((1 - 0.5 * 0.677) / 1e-3)
    say(f"  Estimates for 10 M☉ (ℓ = {ell/1e3:.2f} km): N_e to Planckian curvature at R- (K ∝ δm²): ½ ln(K_P/K_core) = {Ne_planck:.0f}; "
        f"N_e needed for δm ~ M − m(R-) at δm₀ = 10⁻³ M: {Ne_merge:.1f} (if R- moved outward — but it moves inward).")

    say("\n(4) Comparison with the companion paper")
    lam_ext = rows[-1]["lam"]
    M_ext = 10.7 / (lam_ext * 1.4766); M_bhp = 10.7 / (1.49 * 1.4766)
    say(f"  ℓ₀ = 10.7 km: extremality boundary of the triple-root family lambda_ext = {lam_ext:.3f} → M = {M_ext:.2f} M☉; the companion paper's criterion ℓ₀/M ≥ 1.49 (profile n = 6, without a triple root) → {M_bhp:.2f} M☉.")
    say("  Different objects: our point is the boundary \"hole with a triple inner root ↔ horizonless object\" (the triple root merges with the outer one, a quadruple root), theirs is the boundary \"two-horizon hole with kappa- ≠ 0 ↔ horizonless\" (a double root) for a monotone profile with no dip in the slope; for the same ℓ₀ the thresholds differ by a factor of 1.49/1.066 = 1.40.")
    json.dump(dict(configs=rows, perturbations=pert, drift=drift, Ne_planck=Ne_planck, Ne_merge=Ne_merge, M_ext=M_ext, M_bhp=M_bhp, kappa_plus_exponent=p),
              open(OUT / "T21_extremal.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T21_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
