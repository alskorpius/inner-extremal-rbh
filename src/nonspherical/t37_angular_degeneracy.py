"""T37, item 2: how many conditions does degeneracy impose once R_- depends on the angle, and by how much
does the T1 tuning requirement (|eps| <= 3e-32 for 10 M_sun) get harder.

Pieces:
  A. COUNTING (analytic).  Spherical case: f = f' = f'' = 0 at one radius -> 3 equations, 1 unknown (R_-),
     codimension 2 in the space of profiles (this is also what T31_REPORT.md found for the rotating retuning).
     Non-spherical case: the inner horizon is a surface S, and degeneracy is theta = delta_n theta = delta_n^2 theta = 0
     AT EVERY POINT of S.  The free datum is the location of S, one function on S^2.  Net: 2 FUNCTIONAL conditions
     on the sphere -- a continuum.  Truncated at multipole L: N(L) = 2 (L+1)^2 real conditions (of which 3 are
     removable by translations for L >= 1, and 3 more, the l=1 odd modes, are the spin -- a genuine parameter).
  B. MEASURE.  With each mistuning amplitude held to the T1 tolerance eps_max, the fraction of source space that
     works falls from ~eps_max^2 to ~prod_lm eps_max,lm.  Per-mode tolerance in the sup norm:
     |c_lm| <= eps_max / max|Y_lm| = eps_max sqrt(4 pi/(2l+1)) for m = 0.
  C. SPIN (computed).  Gurses-Gursey Delta = r^2 - 2 m(r) r + a^2 with the branch-A m(r): the triple root is lifted,
     |kappa_-| = C a^{4/3}.  Invert for the largest spin compatible with N_e = |kappa_-| tau <= 10.
  D. ENVIRONMENT (derived + arithmetic).  A static external quadrupole gives eps ~ E R_-^2 = M_c R_-^2/d^3.
     Invert for the exclusion distance.  Same for a passing gravitational wave.
  E. RELAXATION (estimate).  Freely decaying multipoles are removed by ringdown + Price tail; time to fall below
     eps_max.  This is the part that works AGAINST the "non-sphericity makes it worse" expectation and is
     reported as such.

Run: python src/nonspherical/t37_angular_degeneracy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
FIG = OUT
sys.path.insert(0, str(ROOT / "src" / "inhomogeneous_collapse"))
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# T1 (experiments/verification_01/T1_REPORT.md): N_e = |kappa_-| tau <= 10 with tau = 1e10 yr.
GMSUN = 4.925490947e-6          # GM_sun/c^3 in seconds
MSUN_M = 1476.6250385           # GM_sun/c^2 in metres
YEAR = 3.1557e7
AU = 1.495978707e11
PC = 3.0856775814913673e16
EPS_MAX_10 = 3.0e-32            # T1, type (a), base profile, 10 M_sun
EPS_MAX_1e6 = 8.6e-25           # T1, 10^6 M_sun
KAPPA_MAX_10 = 1.6e-21          # T1, in units 1/M
KAPPA_MAX_1e6 = 1.6e-16


def main():
    res = {}

    # ------------------------------------------------------------------ A. counting
    say("T37 item 2 -- degeneracy under angular dependence: how many conditions, and how much harder")
    say("\n--- A. Counting (analytic)")
    say("  spherical:     3 equations (f = f' = f'' = 0) minus 1 unknown (R_-)  -> codimension 2")
    say("  non-spherical: 3 equations AT EVERY POINT of the inner-horizon surface S, minus one free function")
    say("                 (the location of S)                                   -> codimension 2 x dim C^inf(S^2)")
    rows = []
    for L in (0, 1, 2, 3, 4, 8, 16):
        n_modes = (L + 1) ** 2
        N = 2 * n_modes
        rows.append(dict(L=L, n_modes=n_modes, n_conditions=N, n_conditions_minus_gauge=max(N - 3, 0)))
        say(f"    multipole cutoff L = {L:2d}: (L+1)^2 = {n_modes:4d} angular modes -> {N:5d} real conditions "
            f"({max(N - 3, 0)} after removing the 3 translations)")
    res["counting"] = rows
    say("  L -> infinity: a continuum.  The degeneracy condition is no longer 'three numbers vanish at a point'")
    say("  but 'two functions on the sphere vanish identically'.")

    # ------------------------------------------------------------------ B. measure
    say("\n--- B. Tuning measure")
    say(f"  T1 tolerance (10 M_sun, type a, base profile): |eps| <= {EPS_MAX_10:.1e}; for 10^6 M_sun {EPS_MAX_1e6:.1e}")
    say("  per-mode tolerance in the sup norm: |c_lm| <= eps_max / max|Y_lm| = eps_max sqrt(4 pi/(2l+1)) (m = 0):")
    for l in (0, 2, 4, 10):
        fac = np.sqrt(4 * np.pi / (2 * l + 1))
        say(f"    l = {l:2d}: factor sqrt(4 pi/(2l+1)) = {fac:.3f}  ->  |c_l0| <= {EPS_MAX_10 * fac:.2e}")
    say("  (an O(1) factor: the per-mode precision stays ~1e-32; what explodes is the NUMBER of them)")

    meas = []
    for eps, tag in ((EPS_MAX_10, "10 M_sun"), (EPS_MAX_1e6, "1e6 M_sun")):
        for L in (0, 1, 2, 3, 4, 8):
            N = 2 * (L + 1) ** 2
            logP = N * np.log10(eps)
            logP_sph = 2 * np.log10(eps)
            meas.append(dict(M=tag, L=L, n_conditions=N, log10_fraction=float(logP),
                             log10_harder_than_spherical=float(logP_sph - logP)))
            if tag == "10 M_sun":
                say(f"    {tag}, L = {L}: fraction of source space ~ eps^{N} = 10^{logP:.0f} "
                    f"(spherical 10^{logP_sph:.0f}); harder by 10^{logP_sph - logP:.0f}")
    res["measure"] = meas
    say(f"  Quadrupole truncation alone (L = 2, 18 conditions): 10^{2 * np.log10(EPS_MAX_10) - 18 * np.log10(EPS_MAX_10):.0f}"
        f" times harder than the spherical requirement.")

    # ------------------------------------------------------------------ C. spin
    say("\n--- C. Spin (computed): Gurses-Gursey Delta = r^2 - 2 m(r) r + a^2 with the branch-A m(r)")
    scen = lb.ScenarioMass()
    hz = lb.static_horizons(scen, 1.0)
    Rm = hz[0][0]
    mfun = lambda r: scen.all(1.0, np.asarray(r, float))["m"]
    mpfun = lambda r: scen.all(1.0, np.asarray(r, float))["m_R"]

    def kappa_inner(a):
        """Inner root of Delta and its surface gravity kappa = Delta'(r_h)/(2(r_h^2+a^2))."""
        rg = np.linspace(0.3 * Rm, 1.6 * Rm, 400001)
        D = rg**2 - 2 * mfun(rg) * rg + a**2
        sc = np.nonzero(np.sign(D[:-1]) * np.sign(D[1:]) < 0)[0]
        if not len(sc):
            return None, None
        i = sc[0]
        rh = brentq(lambda r: r**2 - 2 * float(mfun(r)) * r + a**2, rg[i], rg[i + 1], xtol=1e-15, rtol=1e-15)
        Dp = 2 * rh - 2 * float(mfun(rh)) - 2 * rh * float(mpfun(rh))
        return rh, Dp / (2 * (rh**2 + a**2))

    say("  analytic law: near a triple root f = a3 (r - R_-)^3, Delta = r^2 f + a^2 = 0 gives")
    say("    kappa_- = -(3/2) |a3|^{1/3} (a/R_-)^{4/3}   [derived here; the exponent 4/3 reproduces T31_REPORT.md]")
    spin_rows = []
    for a in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3):
        rh, k = kappa_inner(a)
        if k is None:
            say(f"    a = {a:6g}: no horizon")
            continue
        spin_rows.append(dict(a=a, r_inner=rh, kappa=k, kappa_over_a43=k / a ** (4 / 3)))
        say(f"    a = {a:6g}: inner root r = {rh:.5f}, kappa_- = {k:+.5e}, kappa_-/a^(4/3) = {k / a**(4/3):+.4f}")
    res["spin_law"] = spin_rows
    # the asymptotic coefficient must be read off at the SMALLEST a (the 4/3 law is the a -> 0 limit; at
    # a >~ 0.01 the subleading -2a^2/R_- term already bends it, exactly as in T1 for large |eps|).
    # a = 1e-4 gives |kappa_-| = 1e-5, only twice the residual kappa_0 of the numerically tuned profile,
    # so it is excluded; a >= 1e-2 is already bent by the subleading term.
    small = [r for r in spin_rows if 3e-4 <= r["a"] <= 3e-3]
    aa = np.array([r["a"] for r in small])
    kk = np.abs([r["kappa"] for r in small])
    p_fit, lc = np.polyfit(np.log(aa), np.log(kk), 1)
    C_spin = float(np.mean(kk / aa ** (4 / 3)))
    p = 4 / 3
    say(f"  free fit on 3e-4 <= a <= 3e-3: exponent {p_fit:.4f} (analytic 4/3 = 1.3333); coefficient at fixed 4/3: "
        f"C = {C_spin:.4f} (spread {np.std(kk / aa**(4/3)) / C_spin * 100:.2f}%)")
    say(f"  the numerical profile has a residual kappa_0 = {hz[0][1]:.2e}, so a < ~1e-4 is not resolvable here.")
    res["spin_fit"] = dict(C=C_spin, exponent_used=p, exponent_free_fit=float(p_fit),
                           kappa_residual=float(hz[0][1]))

    spin_bounds = []
    for kmax, tag, Mtag in ((KAPPA_MAX_10, "10 M_sun", 10.0), (KAPPA_MAX_1e6, "1e6 M_sun", 1e6)):
        a_max = (kmax / C_spin) ** (1 / p)
        # tolerance on a for a hole already retuned at spin a0: delta(Delta) = 2 a0 delta a  <=>  eps = 2 a0 da/R_-^2
        da01 = EPS_MAX_10 * Rm**2 / (2 * 0.1) if tag == "10 M_sun" else EPS_MAX_1e6 * Rm**2 / (2 * 0.1)
        spin_bounds.append(dict(M=tag, kappa_max=kmax, a_max=float(a_max), delta_a_at_a0_0p1=float(da01)))
        say(f"    {tag}: |kappa_-| <= {kmax:.1e}/M  ->  |a| <= {a_max:.2e}   "
            f"(and, for a hole retuned at a0 = 0.1, |delta a| <= {da01:.2e})")
    res["spin_bounds"] = spin_bounds
    # what accretion does to a: delta J ~ l_ISCO delta M with l_ISCO ~ 3.46 M  ->  delta a ~ 3.46 delta M/M
    dM_over_M = float(spin_bounds[0]["a_max"] / 3.46)
    say(f"  accreting mass dM with ISCO specific angular momentum (l = 3.46 M) gives delta a ~ 3.46 dM/M,")
    say(f"  so |a| <= {spin_bounds[0]['a_max']:.1e} requires dM/M <= {dM_over_M:.1e}, i.e. dM <= "
        f"{dM_over_M * 10 * 1.989e30:.2e} kg for a 10 M_sun hole.")
    res["spin_accretion"] = dict(dM_over_M=dM_over_M, dM_kg_10Msun=float(dM_over_M * 10 * 1.989e30))

    # ------------------------------------------------------------------ D. environment
    say("\n--- D. Environment: an l = 2 tide cannot be tuned away")
    Rm_m_10 = Rm * 10 * MSUN_M
    say(f"  R_- = {Rm:.4f} M = {Rm_m_10:.4g} m for 10 M_sun; static external quadrupole eps ~ E R_-^2 = M_c R_-^2/d^3")
    env = []
    for name, Mc_msun, d_m in (("solar-mass companion at 1 AU", 1.0, AU),
                               ("solar-mass star at 1000 AU", 1.0, 1000 * AU),
                               ("nearest star at 1 pc", 1.0, PC),
                               ("Galaxy (1e11 M_sun at 8 kpc)", 1e11, 8000 * PC)):
        eps = Mc_msun * MSUN_M * Rm_m_10**2 / d_m**3
        env.append(dict(case=name, eps=float(eps), over_tolerance=float(eps / EPS_MAX_10)))
        say(f"    {name:32s}: eps = {eps:.2e}  ({eps / EPS_MAX_10:.1e} x the tolerance)")
    d_min = (1.0 * MSUN_M * Rm_m_10**2 / EPS_MAX_10) ** (1 / 3)
    say(f"  exclusion distance: no solar mass closer than d = {d_min:.3e} m = {d_min / AU:.0f} AU = {d_min / PC:.4f} pc")
    res["environment_static"] = dict(cases=env, d_min_m=float(d_min), d_min_AU=float(d_min / AU), d_min_pc=float(d_min / PC))

    say("  passing gravitational wave of strain h and wavelength lam: eps ~ h (R_-/lam)^2.")
    say("  T1 found kappa_- < 0 for BOTH signs of eps, so an oscillating eps does not average out: the e-folds")
    say("  accumulate at the rate |kappa_-| ~ C |eps|^{2/3} (C = 2.06, T1 base profile, type a) for as long as")
    say("  the wave is present.  N_e is quoted over the stated exposure time.")
    gw = []
    for name, h, lam, expo, expo_tag in (
            ("LIGO-band stochastic background, h ~ 1e-25 at 100 Hz", 1e-25, 2.998e8 / 100, 1e10 * YEAR, "1e10 yr"),
            ("loud binary merger at 100 Mpc, h ~ 1e-21 at 100 Hz", 1e-21, 2.998e8 / 100, 0.1, "0.1 s (transient)"),
            ("pulsar-timing background, h_c ~ 1e-15 at 1/yr", 1e-15, 2.998e8 * YEAR, 1e10 * YEAR, "1e10 yr")):
        eps = h * (Rm_m_10 / lam) ** 2
        kap = 2.06 * abs(eps) ** (2 / 3)
        Ne = kap * (expo / (10 * GMSUN))
        gw.append(dict(case=name, eps=float(eps), kappa=float(kap), exposure=expo_tag, N_efolds=float(Ne)))
        say(f"    {name:52s}: eps = {eps:.2e}, |kappa_-| = {kap:.2e}/M, N_e over {expo_tag} = {Ne:.2e}")
    res["environment_gw"] = gw

    # ------------------------------------------------------------------ E. relaxation (works the other way)
    say("\n--- E. Counter-argument: freely decaying multipoles DO relax below the tolerance")
    A0, wi, l = 0.1, 0.0890, 2                                 # l = 2 Schwarzschild QNM damping, in 1/M
    t_cross = 150.0                                            # ringdown -> tail crossover, in M (standard range 100-200 M)
    A_cross = A0 * np.exp(-wi * t_cross)
    n_tail = 2 * l + 3
    t_eps = t_cross * (A_cross / EPS_MAX_10) ** (1 / n_tail)
    say(f"  l = 2 ringdown (Im omega = -{wi}/M) from A0 = {A0} down to the tail at t = {t_cross} M: A = {A_cross:.2e}")
    say(f"  Price tail t^-{n_tail}: reaches eps_max = {EPS_MAX_10:.1e} at t = {t_eps:.3e} M "
        f"= {t_eps * 10 * GMSUN:.2f} s for 10 M_sun")
    say("  => a multipole left over from collapse is gone within seconds.  Non-sphericity therefore makes the")
    say("     tuning harder only through the channels that do NOT decay: spin, a permanent environmental tide,")
    say("     continuing accretion -- and through the interior, where the same tail is blueshifted at R_- instead")
    say("     of decaying.")
    res["relaxation"] = dict(A0=A0, imag_omega=wi, t_crossover_M=t_cross, A_crossover=float(A_cross),
                             tail_exponent=n_tail, t_to_eps_max_M=float(t_eps), t_to_eps_max_s_10Msun=float(t_eps * 10 * GMSUN))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    Ls = np.arange(0, 9)
    Ns = 2 * (Ls + 1) ** 2
    ax[0].semilogy(Ls, [10 ** (n * np.log10(EPS_MAX_10)) for n in Ns], "o-")
    ax[0].set_xlabel("multipole cutoff L"); ax[0].set_ylabel("fraction of source space")
    ax[0].set_title(r"tuned fraction $\sim \epsilon^{2(L+1)^2}$, $\epsilon=3\times10^{-32}$")
    ax[0].grid(alpha=0.3)
    aa2 = np.geomspace(1e-3, 0.3, 60)
    kk2 = [abs(kappa_inner(a)[1] or np.nan) for a in aa2]
    ax[1].loglog(aa2, kk2, "C3.", label="computed")
    ax[1].loglog(aa2, C_spin * aa2 ** (4 / 3), "k--", label=r"$C a^{4/3}$")
    ax[1].set_xlabel("spin a/M"); ax[1].set_ylabel(r"$|\kappa_-|$ [1/M]")
    ax[1].set_title("spin lifts the triple root"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "angular_degeneracy.png", dpi=130)
    plt.close(fig)

    (OUT / "t37_angular_degeneracy.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (LOGS / "t37_angular_degeneracy_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
