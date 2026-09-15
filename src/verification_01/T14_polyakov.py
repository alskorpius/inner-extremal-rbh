"""T14: semiclassical stress tensor at R₋ in the 2D Polyakov approximation.
Metric ds² = −f du dv, u = t − r*, v = t + r*, e^{2ρ} = f, ∂_u = −(f/2)∂_r, ∂_v = +(f/2)∂_r.
Polyakov tensor (single massless scalar field, ħ = c = 1):
    T_uu = −(1/12π)[(∂_uρ)² − ∂_u²ρ] + t_u,  T_vv = −(1/12π)[(∂_vρ)² − ∂_v²ρ] + t_v,  T_uv = −(1/12π)∂_u∂_vρ.
With ρ = ½ ln f:  ∂_uρ = −f′/4, ∂_u²ρ = ff″/8, ∂_vρ = +f′/4, ∂_v²ρ = ff″/8, ∂_u∂_vρ = −ff″/8  ⇒
    T^B_uu = T^B_vv = −(f′² − 2ff″)/(192π),  T_uv = ff″/(96π);  trace T = −(4/f)T_uv = −f″/(24π) = R/(24π) (R = −f″ for ds² = −f dt² + dr²/f) ✓.
States: Unruh t_u = κ₊²/(48π), t_v = 0; Boulware t_u = t_v = 0; Hartle–Hawking t_u = t_v = κ₊²/(48π).
Run: python src/verification_01/T14_polyakov.py
"""
import json
import sys
from pathlib import Path

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
import qnm_band as qb  # noqa: E402  (import only, not run: build_family, FamilyBG)

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- backgrounds ----------------
class Metric:
    """f, f', f'' on a grid; f''' numerically."""

    def __init__(self, name, mfun, eps=0.0):
        self.name, self.mfun, self.eps = name, mfun, eps

    def f(self, r):
        r = np.asarray(r, float)
        m, mp, mpp = self.mfun(r)
        s = 1 + self.eps
        f = 1 - 2 * s * m / r
        fp = 2 * s * m / r**2 - 2 * s * mp / r
        fpp = -4 * s * m / r**3 + 4 * s * mp / r**2 - 2 * s * mpp / r
        return f, fp, fpp

    def fppp(self, r, h=1e-4):
        return (self.f(r + h)[2] - self.f(r - h)[2]) / (2 * h)

    def horizons(self):
        r = np.geomspace(0.05, 6.0, 600001)
        f = self.f(r)[0]
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        roots = [brentq(lambda x: self.f(x)[0], r[i], r[i + 1], xtol=1e-15) for i in idx]
        # touch (triple root): minimum |f| inside with no sign change
        inner = r < 1.2
        j = np.argmin(np.abs(f[inner]))
        touch = float(r[inner][j]) if np.abs(f[inner][j]) < 1e-6 else None
        return roots, touch


def scenario_m():
    base0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
    fam = qb.build_family(base0, 1.0)
    bg = qb.FamilyBG(fam)

    def mfun(r):
        d = bg.fields(r)
        return d["m"], d["mp"], d["mpp"]
    return mfun, fam


def hayward_m(ell, M=1.0):
    def mfun(r):
        D = r**3 + 2 * M * ell**2
        m = M * r**3 / D
        mp = 6 * M**2 * ell**2 * r**2 / D**2
        mpp = 12 * M**2 * ell**2 * r * (D - 3 * r**3) / D**3   # d/dr [6M²ℓ² r²/D²]
        return m, mp, mpp
    return mfun


def schw_m(r, M=1.0):
    z = np.zeros_like(np.asarray(r, float))
    return M + z, z, z


# ---------------- Polyakov tensor ----------------
def polyakov(met, r, kappa_p, state="unruh"):
    f, fp, fpp = met.f(r)
    TB = -(fp**2 - 2 * f * fpp) / (192 * np.pi)
    Tuv = f * fpp / (96 * np.pi)
    t_u = kappa_p**2 / (48 * np.pi) if state in ("unruh", "hh") else 0.0
    t_v = kappa_p**2 / (48 * np.pi) if state == "hh" else 0.0
    return TB + t_u, TB + t_v, Tuv, f


def rho_ff(met, r, kappa_p, E=1.0, state="unruh"):
    Tuu, Tvv, Tuv, f = polyakov(met, r, kappa_p, state)
    s = np.sqrt(E**2 - f)
    uu, uv = (E + s) / f, (E - s) / f
    return Tuu * uu**2 + Tvv * uv**2 + 2 * Tuv * uu * uv, Tuu, Tvv, Tuv, f


def conservation_residual(met, r, kappa_p):
    """Static case: −T_vv′ + T_uv′ − (f′/f) T_uv = 0 (and symmetrically for uu); relative residual."""
    Tuu, Tvv, Tuv, f = polyakov(met, r, kappa_p)
    fp = met.f(r)[1]
    dTvv = np.gradient(Tvv, r); dTuv = np.gradient(Tuv, r)
    res = -dTvv + dTuv - fp / f * Tuv
    scale = np.max(np.abs(dTvv) + np.abs(dTuv) + np.abs(fp / f * Tuv))   # normalized by the max of the terms (not pointwise)
    return np.max(np.abs(res)) / scale


def analyse(met, label):
    roots, touch = met.horizons()
    Rp = roots[-1]
    kp = met.f(Rp)[1][0] / 2 if np.ndim(met.f(Rp)[1]) else met.f(Rp)[1] / 2
    if touch is not None and (len(roots) == 1):
        Rm, km = touch, met.f(touch)[1] / 2
    else:
        Rm = roots[0]; km = met.f(Rm)[1] / 2
    kp = float(kp); km = float(km)
    say(f"\n{label}: R₋ = {Rm:.6f} (κ₋ = {km:+.3e}), R₊ = {Rp:.6f} (κ₊ = {kp:.5f}); roots of f: {len(roots)}" + (f", touch at {touch:.5f}" if touch else ""))
    # values at the horizons
    Tuu_m, Tvv_m, Tuv_m, _ = polyakov(met, np.array([Rm]), kp)
    Tuu_p, Tvv_p, Tuv_p, _ = polyakov(met, np.array([Rp]), kp)
    say(f"  T_uu(R₋) = {Tuu_m[0]:+.5e} vs (κ₊²−κ₋²)/48π = {(kp**2-km**2)/(48*np.pi):+.5e}; T_vv(R₋) = {Tvv_m[0]:+.5e} vs −κ₋²/48π = {-km**2/(48*np.pi):+.5e}; T_uv(R₋) = {Tuv_m[0]:+.2e}")
    say(f"  T_uu(R₊) = {Tuu_p[0]:+.2e} (regularity: 0), T_vv(R₊) = {Tvv_p[0]:+.5e} vs −κ₊²/48π = {-kp**2/(48*np.pi):+.5e}")
    r = np.concatenate([np.linspace(Rm * 1.0001, Rp * 0.9999, 4000), np.linspace(Rp * 1.0001, 40, 4000)])
    Tuu, Tvv, Tuv, f = polyakov(met, r, kp)
    flux = Tuu - Tvv
    say(f"  flux T_uu − T_vv: min {flux.min():.6e}, max {flux.max():.6e} (expected κ₊²/48π = {kp**2/(48*np.pi):.6e}); T_uu(40) = {Tuu[-1]:.5e}")
    say(f"  signs in the trapped region R₋<r<R₊: T_uu ∈ [{Tuu[r<Rp].min():+.2e}, {Tuu[r<Rp].max():+.2e}], T_vv ∈ [{Tvv[r<Rp].min():+.2e}, {Tvv[r<Rp].max():+.2e}], T_uv ∈ [{Tuv[r<Rp].min():+.2e}, {Tuv[r<Rp].max():+.2e}]")
    say(f"  conservation residual (relative, max): inside {conservation_residual(met, np.linspace(Rm*1.001, Rp*0.999, 20001), kp):.1e}, outside {conservation_residual(met, np.linspace(Rp*1.001, 40, 20001), kp):.1e}")
    # divergence of ρ_ff as r → R₋⁺
    d = np.geomspace(1e-5, 3e-2, 300)
    rr = Rm + d
    rho, *_ = rho_ff(met, rr, kp)
    a3 = -met.fppp(Rm) / 6
    fpp_m = float(met.f(Rm)[2])
    sl = {}
    for lo, hi in ((1e-5, 1e-4), (1e-4, 1e-3), (3e-3, 3e-2)):
        k = (d >= lo) & (d <= hi)
        sl[(lo, hi)] = float(np.polyfit(np.log(d[k]), np.log(np.abs(rho[k])), 1)[0])
    say(f"  ρ_ff (E = 1) as r → R₋⁺: sign {'+' if rho[-1] > 0 else '−'}; local exponent |ρ_ff| ∝ (r−R₋)^p: " + ", ".join(f"[{lo:g},{hi:g}]: {v:+.2f}" for (lo, hi), v in sl.items())
        + f"; f″(R₋) = {fpp_m:+.2e}, f‴(R₋)/6 = {-a3:+.4f} (f ≈ κ₋·2d + f″d²/2 − a₃d³); asymptotics 4E²T_uu(R₋)/f²: ratio to direct evaluation at d = 1e-3: {np.interp(1e-3, d, rho) / (4*Tuu_m[0]/met.f(Rm+1e-3)[0]**2):.4f}")
    sl = [sl[(3e-3, 3e-2)]]
    return dict(label=label, R_minus=Rm, kappa_minus=km, R_plus=Rp, kappa_plus=kp, Tuu_minus=float(Tuu_m[0]), Tvv_minus=float(Tvv_m[0]),
                slope=float(sl[0]), a3=float(a3), r=r, Tuu=Tuu, Tvv=Tvv, Tuv=Tuv, d=d, rho=rho)


def main():
    say("T14: Polyakov tensor (2D, Unruh state) for the scenario profile, Hayward, and the perturbed scenario")
    # Schwarzschild control
    S = Metric("Schwarzschild", schw_m)
    Tuu, Tvv, Tuv, _ = polyakov(S, np.array([2.0 + 1e-9, 1e4]), 0.25)
    say(f"Schwarzschild control: T_uu(∞) = {Tuu[1]:.6e} vs 1/(768π) = {1/(768*np.pi):.6e}; T_vv(R₊) = {Tvv[0]:.6e} vs −1/(768π) = {-1/(768*np.pi):.6e}; "
        f"T_uu(R₊) = {Tuu[0]:.1e}; conservation residual {conservation_residual(S, np.linspace(2.001, 40, 20001), 0.25):.1e}")
    mS, fam = scenario_m()
    scen = Metric("scenario (triple root)", mS)
    hay = Metric("Hayward ℓ = 0.271", hayward_m(0.271))
    # perturbed scenario: m → (1+ε)m, κ₋ ≈ −1e-2
    def km_of(eps):
        met = Metric("", mS, eps); roots, touch = met.horizons()
        return met.f(roots[0])[1] / 2
    eps = brentq(lambda e: km_of(e) + 1e-2, 1e-6, 1e-2, xtol=1e-12)
    pert = Metric(f"scenario, m→(1+ε)m, ε = {eps:.3e}", mS, eps)
    res = [analyse(scen, "Scenario (triple root)"), analyse(hay, "Hayward ℓ/M = 0.271"), analyse(pert, f"Perturbed scenario (ε = {eps:.2e})")]
    # dependence of the coefficient on κ₋
    say("\nDivergence coefficient T_uu(R₋) = (κ₊² − κ₋²)/48π across the family of perturbations m→(1+ε)m:")
    for e in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2):
        met = Metric("", mS, e); roots, touch = met.horizons()
        Rm = roots[0] if len(roots) > 1 else touch; Rp = roots[-1]
        km = float(met.f(Rm)[1] / 2); kp = float(met.f(Rp)[1] / 2)
        say(f"  ε = {e:.0e}: κ₋ = {km:+.4e}, κ₊ = {kp:.5f}, (κ₊²−κ₋²)/48π = {(kp**2-km**2)/(48*np.pi):+.4e}; κ₋ ∝ ε^(2/3): κ₋/ε^(2/3) = {km/e**(2/3) if e > 0 else float('nan'):+.3f}")
    say(f"  coefficient zero at |κ₋| = κ₊ = {res[0]['kappa_plus']:.4f} (for Hayward ℓ = 0.271: κ₋ = {res[1]['kappa_minus']:+.3f} — |κ₋| > κ₊, coefficient {(res[1]['kappa_plus']**2-res[1]['kappa_minus']**2)/(48*np.pi):+.3e} < 0)")
    # dimensional estimate of the layer
    say("\nDimensional estimate (ħ restored; 4D density = T_2D ħc/(4πR²), compared with ρ_c = 3c⁴/(8πGℓ²)):")
    G, c, Msun, hbar = 6.67430e-11, 2.99792458e8, 1.98847e30, 1.054571817e-34
    mP = np.sqrt(hbar * c / G); l_P = np.sqrt(hbar * G / c**3)
    r0 = res[0]; lam = fam_ell = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    rows = []
    for Ms in (3.0, 10.0, 4e6):
        M = Ms * Msun
        tau_need = 3 * r0["R_minus"]**2 / (2 * lam**2) * (M / mP)**2      # dimensionless ρ_ff at which the 4D density = ρ_c
        # ρ_ff ≈ 4 T_uu(R₋)/f², f ≈ −a₃ (r−R₋)³ ... use |f| = a d³ with a = |f'''|/6
        a = abs(r0["a3"])
        f_crit = np.sqrt(4 * r0["Tuu_minus"] / tau_need)
        d = (f_crit / a) ** (1 / 3)
        GM = G * M / c**2
        rows.append(dict(M=Ms, f_crit=f_crit, d_over_M=d, d_m=d * GM))
        say(f"  M = {Ms:g} M☉: required ρ_ff = {tau_need:.2e} (units M⁻²), f_crit = {f_crit:.2e}, layer |r−R₋| = {d:.2e} M = {d*GM:.2e} m; (l_P² GM/c²)^(1/3) = {(l_P**2*GM)**(1/3):.1e} m; project estimate (McMaken 58a): 3 M☉ 1.3e-10, 10 M☉ 2.8e-10, Sgr A* 1.5e-6 m")
    # dissipation channel
    say("\nDissipation channel:")
    kp = r0["kappa_plus"]
    for Ms in (10.0,):
        M = Ms * Msun; GM = G * M / c**2
        L = kp**2 / (48 * np.pi) * hbar * c / GM**2 * c        # W, single scalar field, no grey-body factors
        Mdot_edd = 2e-8 * Msun / 3.156e7                        # kg/s (SUMMARY: 2·10⁻⁸ M☉/yr for 10 M☉)
        L_need = 0.2 * Mdot_edd * c**2
        say(f"  10 M☉: semiclassical luminosity L = κ₊²ħc⁶/(48πG²M²) = {L:.2e} W (Schwarzschild: {0.25**2/(48*np.pi)*hbar*c**2/GM**2:.2e} W); required negative ingoing flux at Eddington accretion 0.2 Ṁc² = {L_need:.2e} W; ratio {L/L_need:.1e}")
        Tvv_p = -kp**2 / (48 * np.pi)
        say(f"  sign: T_vv(R₊) = {Tvv_p:+.3e} < 0 — ingoing negative-energy flux (partners of the Hawking quanta), same sign as required; T_vv in the trapped region: [{res[0]['Tvv'][res[0]['r'] < res[0]['R_plus']].min():+.2e}, {res[0]['Tvv'][res[0]['r'] < res[0]['R_plus']].max():+.2e}]")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for k, key, ttl in ((0, "Tuu", "T_uu (Unruh)"), (1, "Tvv", "T_vv (Unruh)"), (2, "Tuv", "T_uv")):
            a_ = ax.flat[k]
            for R in res:
                a_.plot(R["r"], R[key], label=R["label"])
            a_.set_xscale("log"); a_.set_yscale("symlog", linthresh=1e-5); a_.set_xlabel("r/M"); a_.set_title(ttl); a_.grid(alpha=.3)
        a_ = ax.flat[3]
        for R in res:
            a_.loglog(R["d"], np.abs(R["rho"]), label=f"{R['label']}: slope {R['slope']:+.2f}")
        a_.set_xlabel("r − R₋ (M)"); a_.set_title("|ρ_ff| of a freely falling observer (E = 1)"); a_.grid(alpha=.3); a_.legend(fontsize=7)
        ax.flat[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "T14_rset.png", dpi=120)
        say(f"\nPlot: {OUT / 'T14_rset.png'}")
    except Exception as e:  # noqa: BLE001
        say(f"plot not built: {e}")
    json.dump(dict(eps=eps, backgrounds=[{k: v for k, v in R.items() if not isinstance(v, np.ndarray)} for R in res], layer=rows),
              open(OUT / "T14_polyakov.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T14_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
