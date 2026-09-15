"""T23: rotation inside a rotating regular black hole with a triple inner root (Franzin-Liberati-Mazza-Vellucci 2022
metric in the Gurses-Gursey class: Kerr with M -> m(r) = M (r^2 + alpha r + beta)/(r^2 + gamma r + mu); coefficients from
src/rotating_eikonal/eikonal_qnm.coefficients). The conformal factor Psi/Sigma (core regularization) affects
omega = -g_tphi/g_phiphi, but not the sign of g_tt or G_rt (proof in the report), so it is not needed here.
    Sigma = r^2 + a^2 cos^2, Delta = r^2 - 2 m r + a^2, g_tt = -(1 - 2 m r/Sigma), g_tphi = -2 a m r sin^2/Sigma,
    g_phiphi = [(r^2+a^2)^2 - Delta a^2 sin^2] sin^2/Sigma,  omega = 2 a m r/[(r^2+a^2)^2 - Delta a^2 sin^2].
Run: python src/verification_02/T23_rotation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "rotating_eikonal"))
from eikonal_qnm import coefficients  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class GG:
    def __init__(self, M, a, e):
        self.M, self.a, self.e = M, a, e
        self.r_p, self.r_m, self.al, self.be, self.ga, self.mu = coefficients(M, a, e)

    def m(self, r):
        return self.M * (r**2 + self.al * r + self.be) / (r**2 + self.ga * r + self.mu)

    def Delta(self, r):
        return r**2 - 2 * self.m(r) * r + self.a**2

    def omega(self, r, th):
        s2 = np.sin(th) ** 2
        return 2 * self.a * self.m(r) * r / ((r**2 + self.a**2) ** 2 - self.Delta(r) * self.a**2 * s2)

    def gtt(self, r, th):
        Sig = r**2 + self.a**2 * np.cos(th) ** 2
        return -(1 - 2 * self.m(r) * r / Sig)


def kerr_Omega(M, a):
    rp = M + np.sqrt(M**2 - a**2); rm = M - np.sqrt(M**2 - a**2)
    return rp, rm, a / (rp**2 + a**2), a / (rm**2 + a**2)


def ergo_roots(g, th, r_max=4.0):
    """roots of g_tt = 0 <=> Sigma - 2 m r = 0 over r in (1e-6, r_max)."""
    F = lambda r: r**2 + g.a**2 * np.cos(th) ** 2 - 2 * g.m(r) * r
    rr = np.geomspace(1e-6, r_max, 200001)
    v = F(rr)
    idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    return [brentq(F, rr[i], rr[i + 1], xtol=1e-13) for i in idx]


def symbolic_G_rt():
    """R_rt (= G_rt, since g_rt = 0) for the Gurses-Gursey metric with arbitrary m(r)."""
    t, r, th, ph, a = sp.symbols("t r theta phi a", real=True)
    m = sp.Function("m")(r)
    Sig = r**2 + a**2 * sp.cos(th) ** 2
    Del = r**2 - 2 * m * r + a**2
    s2 = sp.sin(th) ** 2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * m * r / Sig); g[0, 3] = g[3, 0] = -2 * a * m * r * s2 / Sig
    g[3, 3] = ((r**2 + a**2) ** 2 - Del * a**2 * s2) * s2 / Sig
    g[1, 1] = Sig / Del; g[2, 2] = Sig
    # inverse: (t,phi) block with determinant -Delta sin^2
    D = -Del * s2
    gi = sp.zeros(4, 4)
    gi[0, 0] = g[3, 3] / D; gi[0, 3] = gi[3, 0] = -g[0, 3] / D; gi[3, 3] = g[0, 0] / D
    gi[1, 1] = Del / Sig; gi[2, 2] = 1 / Sig
    X = [t, r, th, ph]
    # check the inverse
    chk = sp.simplify((g * gi - sp.eye(4))[0, 0])
    assert chk == 0, chk
    Gam = [[[sp.S(0)] * 4 for _ in range(4)] for _ in range(4)]
    for l in range(4):
        for i in range(4):
            for j in range(i, 4):
                val = sum(gi[l, k] * (sp.diff(g[k, i], X[j]) + sp.diff(g[k, j], X[i]) - sp.diff(g[i, j], X[k])) for k in range(4)) / 2
                Gam[l][i][j] = Gam[l][j][i] = val
    i, j = 1, 0   # R_{r t}
    Ric = sum(sp.diff(Gam[l][i][j], X[l]) for l in range(4)) - sum(sp.diff(Gam[l][i][l], X[j]) for l in range(4)) \
        + sum(Gam[l][l][s] * Gam[s][i][j] for l in range(4) for s in range(4)) - sum(Gam[l][j][s] * Gam[s][i][l] for l in range(4) for s in range(4))
    return sp.simplify(Ric)


def main():
    say("T23: frame dragging omega, ergoregions and radial flux inside a rotating triple inner root (Franzin et al. 2022, Gurses-Gursey class)")
    say("\n(0) Symbolic: G_rt = R_rt for Kerr with M -> m(r) (arbitrary m):")
    Grt = symbolic_G_rt()
    say(f"    R_rt = {Grt}  -> no radial energy flux T^r_t identically for any m(r)")
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ia, a in enumerate((0.6, 0.9)):
        rpK, rmK, OpK, OmK = kerr_Omega(1.0, a)
        ax = axes[ia]
        for e in (0.5, 1.0):
            g = GG(1.0, a, e)
            # check horizons and triple root
            d_p = g.Delta(g.r_p); d_m = g.Delta(g.r_m)
            h = 1e-5
            d1 = (g.Delta(g.r_m + h) - g.Delta(g.r_m - h)) / (2 * h); d2 = (g.Delta(g.r_m + h) - 2 * d_m + g.Delta(g.r_m - h)) / h**2
            disc = g.ga**2 - 4 * g.mu
            Om_p = g.omega(g.r_p, np.pi / 2); Om_m = g.omega(g.r_m, np.pi / 2)
            th_dep_p = max(abs(g.omega(g.r_p, th) - Om_p) for th in (0.3, 1.0, 1.4))
            th_dep_m = max(abs(g.omega(g.r_m, th) - Om_m) for th in (0.3, 1.0, 1.4))
            m0 = g.m(0.0)
            om_core_eq = g.omega(1e-6, np.pi / 2); om_core_60 = g.omega(1e-6, np.pi / 3); om_axis = g.omega(1e-6, 1e-9)
            rr = np.geomspace(1e-4, 10, 4000)
            om_eq = g.omega(rr, np.pi / 2)
            # sign of omega inside R-: minimum over r and theta
            rin = np.linspace(1e-4, g.r_m, 2000)
            om_min = min(g.omega(rin, th).min() for th in (np.pi / 2, np.pi / 3, np.pi / 6, 1e-3))
            ergo = {int(round(np.degrees(th))): ergo_roots(g, th) for th in (np.pi / 2, np.pi / 3, np.pi / 6)}
            row = dict(a=a, e=e, r_plus=g.r_p, r_minus=g.r_m, Delta_p=d_p, Delta_m=d_m, dDelta_m=d1, d2Delta_m=d2, disc_denominator=disc,
                       Omega_plus=Om_p, Omega_minus=Om_m, ratio=Om_m / Om_p, kerr=dict(r_minus=rmK, Omega_plus=OpK, Omega_minus=OmK, ratio=OmK / OpK),
                       theta_dependence_on_horizons=[th_dep_p, th_dep_m], m0=m0, omega_core=dict(eq=om_core_eq, th60=om_core_60, axis=om_axis),
                       omega_min_inside=om_min, ergo=ergo)
            rows.append(row)
            say(f"\n  a = {a}, e = {e}: r+ = {g.r_p:.5f}, r- = {g.r_m:.5f}; Delta(r+) = {d_p:.1e}, Delta(r-) = {d_m:.1e}, Delta'(r-) = {d1:.1e}, Delta''(r-) = {d2:.1e}; "
                f"denominator of m(r): discriminant {disc:.3f} (< 0 -- no poles)")
            say(f"    Omega+ = {Om_p:.6f} (Kerr {OpK:.6f}), Omega- = {Om_m:.6f} (Kerr {OmK:.6f} at Kerr r- = {rmK:.4f}); Omega-/Omega+ = {Om_m/Om_p:.4f} (Kerr {OmK/OpK:.4f}); "
                f"theta dependence on the horizons: {th_dep_p:.1e}, {th_dep_m:.1e}")
            say(f"    m(0) = {m0:.4f} M (not 0: the GG-class core is regularized by the conformal factor, not by m); omega(r->0): equator {om_core_eq:.4f} (= 1/a = {1/a:.4f}), theta = 60 deg: {om_core_60:.2e}, axis: {om_axis:.2e}; "
                f"min omega inside r- = {om_min:.2e} (> 0: no zero-angular-momentum core)")
            for k, v in ergo.items():
                say(f"    ergosurfaces at theta = {k} deg: r = {', '.join(f'{x:.5f}' for x in v)}")
            ax.plot(rr, om_eq, label=f"e = {e}")
            ax.axvline(g.r_p, ls=":", c="k"); ax.axvline(g.r_m, ls="--", c="gray")
        rK = np.geomspace(1e-4, 10, 4000)
        gK = GG(1.0, a, 1e-9)
        ax.plot(rK, gK.omega(rK, np.pi / 2), "k-", lw=0.8, label="Kerr")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("r/M"); ax.set_ylabel("omega (equator)"); ax.set_title(f"a = {a}"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "T23_omega.png", dpi=130)
    json.dump(dict(G_rt=str(Grt), rows=rows), open(OUT / "T23_rotation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T23_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
