"""T1 (the reviewer's verification tasks): sensitivity law kappa_- ~ eps^{2/3} near the triple root.
Profiles with a triple root — family stability/triple_root_monotone (base + u3=20, s_end=8, s_end=48). A profile is given by
(shape s(u), R1, rho_c): m(R) = (M/Itot) mI(R/R1), f = 1 - 2m/R; cubic splines in ln u are used (smooth derivatives).
Perturbation types (M = 1 for the unperturbed profile):
  (a) density amplitude at fixed shape and R1: m -> (1+eps) m  (total mass becomes 1+eps);
  (b) ell shift at fixed M and shape: ell -> (1+eps) ell  <=>  rho_c -> rho_c (1+eps)^{-2}, R1 -> R1 (1+eps)^{2/3};
  (g) M -> (1+eps) M at fixed ell (as in accretion_tracking): amplitude (1+eps) and R1 -> R1 (1+eps)^{1/3};
  (v) local mass addition dm = eps M S(R), S a smooth step 0 (R<0.9) -> 1 (R>1.3) (outside R_-: zero effect expected);
  (v') same step, but on [0.7 R_-, 1.3 R_-] (mass added across the inner horizon).
Prediction: f ~ a x^3 near R_-, a = f'''(R_-)/6; with df = eps g: |kappa_-| ~ (3/2)|a|^{1/3}|eps g(R_-)|^{2/3}, kappa_- < 0 for either sign of eps.
Run: python src/verification_01/T1_kappa_law.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "stability"))
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class Profile:
    def __init__(self, base, sigma_min, tag):
        s1s, u_s, info = trm.find_merge_first(sigma_min=sigma_min, base=base)
        kw = info["kw"]; d = s1s - sigma_min
        sig, s, mI, H, Hp = trm.profile(d, **kw)
        scale = 1.0 / float(np.interp(u_s, trm.U, H))
        fam = trm.MonotoneFamily(d, scale, kw=kw)
        self.tag, self.M = tag, 1.0
        self.R1, self.rho_c, self.Itot = fam.R1, fam.rho_c, float(mI[-1])
        self.ell = float(np.sqrt(3 / (8 * np.pi * self.rho_c)))
        self.mI = CubicSpline(trm.LU, mI)
        self.lns = CubicSpline(trm.LU, np.log(s))
        self.sig = CubicSpline(trm.LU, sig)
        self.A = self.M / self.Itot

    # unperturbed functions
    def m0(self, R, R1=None, A=None):
        R1 = self.R1 if R1 is None else R1; A = self.A if A is None else A
        return A * self.mI(np.log(R / R1))

    def m0p(self, R, R1=None, A=None):
        R1 = self.R1 if R1 is None else R1; A = self.A if A is None else A
        return A * self.mI(np.log(R / R1), 1) / R

    def rho(self, R):
        return self.rho_c * np.exp(self.lns(np.log(R / self.R1)))

    def f3(self, R):
        """f'''(R) analytically via rho, sigma, dsigma/dlnu (unperturbed profile)."""
        lu = np.log(R / self.R1)
        rho = self.rho(R); sig = self.sig(lu); sigp = self.sig(lu, 1)
        rho1 = -sig * rho / R
        rho2 = rho * (sig**2 + sig - sigp) / R**2
        m = self.m0(R); m1 = 4 * np.pi * R**2 * rho
        m2 = 4 * np.pi * (2 * R * rho + R**2 * rho1)
        m3 = 4 * np.pi * (2 * rho + 4 * R * rho1 + R**2 * rho2)
        g3 = m3 / R - 3 * m2 / R**2 + 6 * m1 / R**3 - 6 * m / R**4
        return -2 * g3

    # perturbed m, m'
    def m_eps(self, R, kind, eps, Rm=None):
        R = np.asarray(R, float)
        if kind == "a":
            return (1 + eps) * self.m0(R), (1 + eps) * self.m0p(R)
        if kind == "b":
            R1 = self.R1 * (1 + eps) ** (2 / 3)
            return self.m0(R, R1=R1), self.m0p(R, R1=R1)
        if kind == "g":
            R1 = self.R1 * (1 + eps) ** (1 / 3); A = self.A * (1 + eps)
            return self.m0(R, R1=R1, A=A), self.m0p(R, R1=R1, A=A)
        if kind in ("v", "v2"):
            lo, hi = (0.9, 1.3) if kind == "v" else (0.7 * Rm, 1.3 * Rm)
            t = np.clip((R - lo) / (hi - lo), 0.0, 1.0)
            S = t**3 * (10 - 15 * t + 6 * t**2)
            Sp = np.where((R > lo) & (R < hi), 30 * t**2 * (1 - t) ** 2 / (hi - lo), 0.0)
            return self.m0(R) + eps * self.M * S, self.m0p(R) + eps * self.M * Sp
        raise ValueError(kind)

    def f_eps(self, R, kind, eps, Rm=None):
        m, mp = self.m_eps(R, kind, eps, Rm)
        R = np.asarray(R, float)
        return 1 - 2 * m / R, 2 * m / R**2 - 2 * mp / R

    def g_of(self, R, kind, Rm=None):
        """g = df/deps as eps -> 0 (analytically)."""
        m, mp = self.m0(R), self.m0p(R)
        if kind == "a": return -2 * m / R
        if kind == "b": return (4 / 3) * mp
        if kind == "g": return -2 * m / R + (2 / 3) * mp
        lo, hi = (0.9, 1.3) if kind == "v" else (0.7 * Rm, 1.3 * Rm)
        t = np.clip((R - lo) / (hi - lo), 0.0, 1.0)
        return -2 * self.M * t**3 * (10 - 15 * t + 6 * t**2) / R


def roots_and_kappa(prof, kind, eps, Rm, Rp):
    R = np.geomspace(1e-3, Rp * 0.999, 400001)
    f, fp = prof.f_eps(R, kind, eps, Rm)
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    out = []
    for i in idx:
        r = brentq(lambda x: prof.f_eps(x, kind, eps, Rm)[0], R[i], R[i + 1], xtol=1e-15)
        out.append((r, float(prof.f_eps(r, kind, eps, Rm)[1]) / 2))
    return out


def fit_power(eps, kap):
    x, y = np.log(np.abs(eps)), np.log(np.abs(kap))
    p, c = np.polyfit(x, y, 1)
    return p, float(np.exp(c))


def main():
    say("T1: kappa_-(eps) law near the triple root")
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    profs = [Profile(base0, 1.0, "base λ=0.271"), Profile(dict(base0, u3=20.0), 1.0, "u3=20 λ=0.13"),
             Profile(dict(base0, s_end=8.0), 1.0, "s_end=8 λ=0.21"), Profile(dict(base0, s_end=48.0), 1.0, "s_end=48 λ=0.30")]
    eps_list = [10.0**k for k in range(-8, -1)]
    kinds = [("a", "density amplitude"), ("b", "ell shift at fixed M"), ("g", "M -> (1+eps)M at fixed ell"),
             ("v", "mass addition on [0.9, 1.3]"), ("v2", "mass addition on [0.7 R-, 1.3 R-]")]
    results = []
    for prof in profs:
        # unperturbed: horizons, R_-, a
        R = np.geomspace(1e-3, 3.0, 400001)
        f, fp = prof.f_eps(R, "a", 0.0)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        Rp = brentq(lambda x: prof.f_eps(x, "a", 0.0)[0], R[idx[-1]], R[idx[-1] + 1])
        inner = R < 0.9 * Rp
        j = int(np.argmin(np.abs(f[inner])))
        res = minimize_scalar(lambda x: abs(prof.f_eps(x, "a", 0.0)[0]), bracket=(R[j] * 0.999, R[j], R[j] * 1.001), tol=1e-14)
        Rm = float(res.x)
        f_m, fp_m = prof.f_eps(Rm, "a", 0.0)
        a = float(prof.f3(Rm)) / 6
        say(f"\n{prof.tag}: ell = {prof.ell:.4f}, R- = {Rm:.5f} (f = {f_m:.1e}, f' = {fp_m:.1e}, kappa0 = {fp_m/2:.1e}), R+ = {Rp:.5f}, a = f'''/6 = {a:.4f}, "
            f"8 pi R-^2 rho = {8*np.pi*Rm**2*prof.rho(Rm):.5f}")
        prow = dict(tag=prof.tag, ell=prof.ell, R_minus=Rm, R_plus=Rp, kappa0=fp_m / 2, a=a, kinds={})
        for kind, kname in kinds:
            gR = float(prof.g_of(Rm, kind, Rm))
            rows = []
            for sgn in (+1, -1):
                for e in eps_list:
                    eps = sgn * e
                    rk = roots_and_kappa(prof, kind, eps, Rm, Rp * (1 + 0.05))
                    inner_roots = [(r, k) for r, k in rk if r < 0.9 * Rp]
                    # R_- of the perturbed profile: the root closest to the unperturbed R_- (if three, the innermost one is taken as R_-; flagged)
                    if inner_roots:
                        r_sel, k_sel = min(inner_roots, key=lambda t: abs(t[0] - Rm))
                    else:
                        r_sel, k_sel = np.nan, np.nan
                    pred = -(1.5) * abs(a) ** (1 / 3) * abs(eps * gR) ** (2 / 3)
                    rows.append(dict(eps=eps, n_inner=len(inner_roots), R=r_sel, kappa=k_sel, pred=pred, kappas=[k for _, k in inner_roots]))
            # fit over 1e-8..1e-4 for each sign
            fits = {}
            for sgn in (+1, -1):
                sel = [r for r in rows if np.sign(r["eps"]) == sgn and abs(r["eps"]) <= 1e-4 and np.isfinite(r["kappa"]) and abs(r["kappa"]) > 0]
                if len(sel) >= 3:
                    p, C = fit_power(np.array([r["eps"] for r in sel]), np.array([r["kappa"] for r in sel]))
                    fits[sgn] = dict(p=p, C=C)
            Cpred = 1.5 * abs(a) ** (1 / 3) * abs(gR) ** (2 / 3)
            say(f"  ({kind}) {kname}: g(R-) = {gR:+.4f}; prediction |kappa| = {Cpred:.4f} |eps|^(2/3)")
            for r in rows:
                say(f"      eps = {r['eps']:+.0e}: roots inside {r['n_inner']}, R = {r['R']:.5f}, kappa = {r['kappa']:+.3e} (pred. {r['pred']:+.3e}, ratio {r['kappa']/r['pred'] if r['pred'] else np.nan:.3f})"
                    + (f"; all kappa: {['%+.2e' % k for k in r['kappas']]}" if r['n_inner'] > 1 else ""))
            for sgn, ft in fits.items():
                say(f"      fit eps {'>' if sgn > 0 else '<'} 0, |eps| <= 1e-4: |kappa| = {ft['C']:.4f} |eps|^{ft['p']:.4f}")
            prow["kinds"][kind] = dict(name=kname, g=gR, C_pred=Cpred, rows=rows, fits={str(k): v for k, v in fits.items()})
        results.append(prow)
    # controls
    say("\nControls:")
    b = profs[0]
    for kind, e, ref in (("a", 1e-3, "stability: 0.1 % level -> kappa_- = -0.04"), ("a", -1e-3, "same, other sign"), ("g", 1e-2, "accretion_tracking: M -> 1.01 at fixed ell -> kappa_- = -0.079")):
        r = [x for x in results[0]["kinds"][kind]["rows"] if x["eps"] == e][0]
        say(f"  ({kind}) eps = {e:+.0e}: kappa_- = {r['kappa']:+.4f} [{ref}]")
    # precision requirement
    say("\nPrecision requirement (N_e = |kappa_-| tau <= 10, tau = 1e10 yr):")
    yr = 3.15576e7; GM_c3_sun = 4.925490947e-6
    prec = {}
    for Msun in (10.0, 1e6):
        tau_M = 1e10 * yr / (GM_c3_sun * Msun)
        kmax = 10.0 / tau_M
        line = f"  M = {Msun:.0e} Msun: tau = {tau_M:.2e} M, |kappa_-| <= {kmax:.2e} / M"
        prec[str(Msun)] = dict(tau_M=tau_M, kappa_max=kmax, eps_max={})
        for prow in results:
            for kind in ("a", "b", "g", "v2"):
                ft = prow["kinds"][kind]["fits"].get("1") or prow["kinds"][kind]["fits"].get("-1")
                if ft:
                    emax = (kmax / ft["C"]) ** (1 / ft["p"])
                    prec[str(Msun)]["eps_max"][f"{prow['tag']}|{kind}"] = emax
        say(line)
        for k, v in prec[str(Msun)]["eps_max"].items():
            if k.startswith("base"):
                say(f"      {k}: |eps| <= {v:.1e}")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for kind, mk in (("a", "o"), ("b", "s"), ("g", "^"), ("v2", "d")):
            rows = results[0]["kinds"][kind]["rows"]
            for sgn, ls in ((1, "-"), (-1, "--")):
                sel = [r for r in rows if np.sign(r["eps"]) == sgn and np.isfinite(r["kappa"])]
                axs[0].loglog([abs(r["eps"]) for r in sel], [abs(r["kappa"]) for r in sel], ls, marker=mk, ms=4, label=f"({kind}) eps{'>' if sgn>0 else '<'}0")
        e = np.array(eps_list); a0 = results[0]["a"]
        axs[0].loglog(e, 1.5 * abs(a0) ** (1 / 3) * e ** (2 / 3), "k:", label="(3/2)|a|^(1/3) eps^(2/3), g=1")
        axs[0].set_xlabel("|eps|"); axs[0].set_ylabel("|kappa_-| M"); axs[0].set_title("base: perturbation types"); axs[0].legend(fontsize=7)
        for prow in results:
            rows = prow["kinds"]["a"]["rows"]
            sel = [r for r in rows if r["eps"] > 0 and np.isfinite(r["kappa"])]
            axs[1].loglog([r["eps"] for r in sel], [abs(r["kappa"]) for r in sel], "o-", ms=4, label=prow["tag"])
        axs[1].loglog(e, e ** (2 / 3), "k:", label="eps^(2/3)")
        axs[1].set_xlabel("eps (type a, eps>0)"); axs[1].set_ylabel("|kappa_-| M"); axs[1].set_title("profile family"); axs[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "T1_kappa_eps.png", dpi=130)
        say(f"plot: {OUT / 'T1_kappa_eps.png'}")
    except Exception as ex:  # noqa: BLE001
        say(f"plot not built: {ex}")
    json.dump(dict(results=results, precision=prec), open(OUT / "T1_kappa_law.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T1_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
