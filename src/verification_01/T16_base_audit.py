"""T16. Audit of the base profile lambda = 0.271 (experiments/stability/triple_root_monotone.py, base0, sigma_min = 1):
sign of rho', min(rho + p_perp) = min(-R rho'/2), rho outside R+, f, f', f'' at R-, tuning precision; export of m(r), rho(r), f(r), sigma(r) for external use.
Run: python src/verification_01/T16_base_audit.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

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


def main():
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    s1, u_s, info = trm.find_merge_first(sigma_min=1.0, base=base0)
    kw = info["kw"]; d = s1 - 1.0
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    fam = trm.MonotoneFamily(d, scale, kw=kw)
    ell = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    say(f"T16: base profile: s1* = {s1:.10f} (bisection 45 steps over s1 in [2.02, 3.5]: step {(3.5-2.02)/2**45:.1e}), u_s = {u_s:.6f}, scale = {scale:.8f}, R1 = {fam.R1:.6f}, rho_c = {fam.rho_c:.6f}, lambda = {ell:.5f}")
    R = np.geomspace(1e-3, 10.0, 200001)
    rho = fam.rho(R); drho = fam.drho(R)
    sigR = np.interp(np.log(R / fam.R1), trm.LU, fam.sig)
    m = np.interp(R, fam._Rg, fam._mg)
    f = 1 - 2 * m / R
    fp = 2 * m / R**2 - 8 * np.pi * R * rho
    fpp = -4 * m / R**3 - 8 * np.pi * R * drho      # f'' = -4m/R^3 + 4m'/R^2 - 2m''/R, m' = 4 pi R^2 rho, m'' = 8 pi R rho + 4 pi R^2 rho'
    nec = -R * drho / 2   # rho + p_perp
    say(f"  rho' <= 0 everywhere: {bool((drho <= 0).all())}; max rho' = {drho.max():.3e}; sigma = -R rho'/rho: min {sigR.min():.4f} (at R = {R[np.argmin(sigR)]:.3f}), max {sigR.max():.3f} (at R = {R[np.argmax(sigR)]:.3f})")
    say(f"  min(rho + p_perp) = min(-R rho'/2) = {nec.min():.3e} at R = {R[np.argmin(nec)]:.4f} (rho there {rho[np.argmin(nec)]:.3e}); rho > 0 everywhere: {bool((rho > 0).all())}; min rho = {rho.min():.3e} at R = 10")
    say(f"  min(rho + p_r) = 0 identically (p_r = -rho); SEC: min(rho + p_r + 2 p_perp) = min(-2 rho + sigma rho) = {(rho*(sigR-2)).min():.4f} (< 0 in the core)")
    # horizons
    roots = []
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    for i in idx:
        roots.append(brentq(lambda x: 1 - 2 * np.interp(x, fam._Rg, fam._mg) / x, R[i], R[i + 1], xtol=1e-14))
    j = np.argmin(np.abs(f[(R > 0.3) & (R < 1.5)])); Rin = R[(R > 0.3) & (R < 1.5)]
    R_touch = float(Rin[j])
    say(f"  roots of f (sign change): {[round(r, 6) for r in roots]}; minimum |f| on (0.3, 1.5): R = {R_touch:.6f}, f = {float(np.interp(R_touch, R, f)):.2e}, f' = {float(np.interp(R_touch, R, fp)):.2e}, f'' = {float(np.interp(R_touch, R, fpp)):.2e}")
    R_minus = R_touch if not any(abs(r - R_touch) < 0.05 for r in roots) else [r for r in roots if abs(r - R_touch) < 0.05][0]
    kappa_minus = float(np.interp(R_minus, R, fp)) / 2
    R_plus = max(roots); kappa_plus = float(np.interp(R_plus, R, fp)) / 2
    m_plus = float(np.interp(R_plus, fam._Rg, fam._mg))
    say(f"  R- = {R_minus:.5f}, kappa- = {kappa_minus:+.2e}, f''(R-) = {float(np.interp(R_minus, R, fpp)):+.2e}; sigma(R-) = {float(np.interp(R_minus, R, sigR)):.5f}, 8 pi R-^2 rho(R-) = {8*np.pi*R_minus**2*float(fam.rho(R_minus)):.6f}")
    say(f"  R+ = {R_plus:.5f}, kappa+ = {kappa_plus:.5f}, T_H/T_Schw = {4*kappa_plus:.4f}, m(R+)/M = {m_plus:.5f}, halo outside R+ = {1-m_plus:.5f}")
    for Rx in (3.0, 10.0):
        say(f"  rho({Rx:g} M)/rho_c = {float(fam.rho(Rx))/fam.rho_c:.2e}; 1 - m({Rx:g} M)/M = {1-float(np.interp(Rx, fam._Rg, fam._mg)):.2e}")
    # sensitivity: sigma_min on both sides of s1*
    for ds in (1e-6, 1e-4):
        for sgn in (+1, -1):
            sg2, s2, mI2, H2, Hp2 = trm.profile(d + sgn * ds, **kw)
            sc2 = 1.0 / float(np.interp(u_s, trm.U, H2))
            f2 = 1 - sc2 * H2
            fp2 = np.gradient(f2, trm.U)
            k = (trm.U > 0.3 * u_s) & (trm.U < 1.7 * u_s)
            idx2 = np.nonzero(np.sign(f2[k][:-1]) * np.sign(f2[k][1:]) < 0)[0]
            say(f"  shift of the dip depth d by {sgn*ds:+.0e} at the same normalization at u_s: f(u_s) = {float(np.interp(u_s, trm.U, f2)):+.1e}, number of roots on (0.3, 1.7) u_s: {idx2.size}, f'(u_s)/R1 = {float(np.interp(u_s, trm.U, fp2))/fam.R1:+.2e}")
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(R, drho / fam.rho_c, "k-"); ax[0].set_xlabel("R/M"); ax[0].set_ylabel("rho'(R) M/rho_c"); ax[0].set_title("rho' <= 0 everywhere"); ax[0].axvline(R_minus, ls="--", c="r"); ax[0].axvline(R_plus, ls="--", c="b")
    ax[1].semilogx(R, sigR, "k-"); ax[1].axhline(2, ls=":", c="gray"); ax[1].axvline(R_minus, ls="--", c="r", label="R-"); ax[1].axvline(R_plus, ls="--", c="b", label="R+"); ax[1].set_xlabel("R/M"); ax[1].set_ylabel("sigma = -d ln rho/d ln R"); ax[1].legend(); ax[1].set_title("slope: 0 -> %.2f -> %.2f -> %.0f" % (sigR.max(), sigR[(R > 0.5) & (R < 3)].min(), kw["s_end"]))
    fig.tight_layout(); fig.savefig(OUT / "T16_rho_prime.png", dpi=130)
    # export
    Re = np.geomspace(1e-3, 20.0, 4000)
    me = np.interp(Re, fam._Rg, fam._mg); re_ = fam.rho(Re); fe = 1 - 2 * me / Re; se = np.interp(np.log(Re / fam.R1), trm.LU, fam.sig)
    hdr = ("# Base profile of the inner-extremal family (triple root) (src/stability/triple_root_monotone.py, base0, sigma_min=1), units G=c=M=1, lambda=ell/M=%.5f, "
           "rho_c=%.6f, R-=%.5f, R+=%.5f; columns: R, m(R) (Misner-Sharp mass), rho(R) (= -T^t_t = -T^R_R), f = 1-2m/R, sigma = -d ln rho/d ln R; p_perp = -rho(1 - sigma/2)" % (ell, fam.rho_c, R_minus, R_plus))
    with open(OUT / "T16_base_profile_m_of_r.csv", "w", encoding="utf-8") as fh:
        fh.write(hdr + "\nR,m,rho,f,sigma\n")
        for row in np.column_stack([Re, me, re_, fe, se]):
            fh.write(",".join(f"{v:.12e}" for v in row) + "\n")
    json.dump(dict(s1=s1, u_s=u_s, scale=scale, R1=fam.R1, rho_c=fam.rho_c, lam=ell, R_minus=R_minus, kappa_minus=kappa_minus, R_plus=R_plus, kappa_plus=kappa_plus, m_plus=m_plus,
                   sigma_min=float(sigR.min()), sigma_max=float(sigR.max()), nec_min=float(nec.min()), roots=roots), open(OUT / "T16_base_audit.json", "w", encoding="utf-8"), indent=1, default=float)
    (LOGS / "T16_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
