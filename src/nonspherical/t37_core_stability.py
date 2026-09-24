"""T37, item 1: linear stability of the de Sitter plateau (the CORE, 0 < r < R_-) to l >= 2 perturbations.

What is new relative to src/polar_qnm/interior_layer.py (which this reuses):
  * domain: that script treated the TRAPPED LAYER R_- < r < R_+ (N < 0, r timelike) and could therefore only
    measure a transfer-matrix amplification ln||T|| across a finite stretch of "time".  Here the domain is the
    CORE 0 < r < R_-, where N = f > 0: the region is static, t is timelike, and the perturbation problem is a
    genuine eigenvalue problem.  omega^2 < 0 there is a real exponentially growing mode, not an amplification.
  * boundary conditions: regularity at the centre (u ~ r^{l+1}) and decay into the infinite tortoise throat at
    the triple root (r* -> +infinity as r -> R_-^-, with r* ~ 1/(2|a|(R_- - r)^2) because f ~ a(r - R_-)^3).
  * axial sector added (interior_layer.py was polar/NED only).
  * kappa_MS is re-derived here as the TANGENTIAL SOUND SPEED SQUARED of the equivalent anisotropic medium,
    c_t^2 = dp_t/drho, so the polar conclusion no longer depends on the NED realisation.
  * kappa is evaluated as  kappa = sigma/2 - 1 - (1/2) dln(sigma)/dln(r), which stays well conditioned as
    sigma -> 0 in the plateau; polar_qnm.potential_matrix falls back to kappa = 1 there (guard sigma > 1e-12),
    which is a numerical artefact and must not be used inside the core.

Sectors:
  AXIAL.  V_ax = f [ l(l+1) - 6 m/r + 2 m' ] / r^2   (same potential as src/polar_qnm/axial_td.py,
    cross-checked there against Regge-Wheeler; equals the Chandrasekhar-Ferrari fluid form
    f[l(l+1)/r^2 + 4 pi (rho - p_r) - 6m/r^3] because 8 pi rho = 2m'/r^2 and p_r = -rho).
    Inside the inner horizon 2m/r < 1 and m' >= 0 (rho >= 0), hence B := l(l+1) - 6m/r + 2m' > l(l+1) - 3 >= 3
    for l >= 2:  V_ax > 0 everywhere in the core, for ANY regular profile.  Positive potential + vanishing
    boundary terms => omega^2 > 0 => no growing axial mode.  Checked numerically as well.
  POLAR.  The coefficient of l(l+1) in the polar potential is f c_t^2 / r^2 with c_t^2 = kappa_MS.  In a
    de Sitter plateau rho = rho_c (1 - (r/L)^p + ...) gives sigma = p (r/L)^p and c_t^2 -> -1 - p/2 < -1 at
    the centre (<= -2 only once p >= 2; every profile tried here has p >= 2), so the radial operator acquires
    -(1 + p/2) l(l+1)/r^2 plus the l-independent static term (p+2)(p+4)/(4 r^2) of the matter channel
    (H_P Q^2 = 2 pi r^4 rho sigma ~ r^(p+4)), i.e. -(p+2)/4 [2 l(l+1) - (p+4)] in total. This is below the
    critical -1/4 when 2 l(l+1) > p + 4 + 1/(p+2) -- every l >= 2 for p <= 7, which covers every profile here,
    never l = 1 -- and there "fall to the centre": spectrum unbounded below, growth rate unbounded in l and as
    r -> 0 (Hadamard ill-posed). For larger p the lowest multipoles have bounded growth instead
    (t37_large_p.py: p = 10, l = 2 saturates at 26.6/M). Coefficient checked in t37_centre_limit.py.
    Confirmed by (a) the eikonal rate, (b) a two-channel Moreno-Sarbach eigenvalue solve, (c) time evolution
    whose measured growth rate keeps rising as the grid is refined.

Run: python src/nonspherical/t37_core_stability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
FIG = OUT
sys.path.insert(0, str(ROOT / "src" / "polar_qnm"))
sys.path.insert(0, str(ROOT / "src" / "inhomogeneous_collapse"))
sys.argv = [sys.argv[0], "0.05"]
import polar_qnm as pq  # noqa: E402
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------- backgrounds
class ScenarioCore:
    """Branch-A triple-root profile (src/stability/triple_root_monotone.py via ltb_bounce.ScenarioMass)."""

    name = "scenario (branch A, triple root)"

    def __init__(self, M=1.0):
        self.scen = lb.ScenarioMass()
        self.M = M
        self.bg = pq.Background(self.scen, M)
        hz = lb.static_horizons(self.scen, M)
        self.Rm, self.kappa_res = hz[0][0], hz[0][1]
        self.Rp = hz[-1][0]
        self.ell = self.scen.ell
        # dln(sigma)/dln(r) on the profile's own log grid (stable as sigma -> 0)
        s = np.maximum(self.scen.sig, 1e-300)
        self._dlnsig = np.gradient(np.log(s), self.scen.LU)
        self._R1 = (2 * M * self.scen.ell**2 / (3 * self.scen.Itot)) ** (1 / 3)

    def fields(self, r):
        return self.bg.fields(np.asarray(r, float))

    def dln_sigma(self, r):
        return np.interp(np.log(np.asarray(r, float) / self._R1), self.scen.LU, self._dlnsig)


class AnalyticCore:
    """Closed-form regular profiles, for the p-law control of c_t^2(0) = -1 - p/2."""

    def __init__(self, name, m, mp, mpp, sigma, dlnsigma, ell, M=1.0):
        self.name, self.M, self.ell = name, M, ell
        self._m, self._mp, self._mpp, self._sig, self._dls = m, mp, mpp, sigma, dlnsigma
        rg = np.geomspace(1e-6, 6.0, 400001)
        f = 1 - 2 * self._m(rg) / rg
        sc = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        self.Rm = float(rg[sc[0]]) if len(sc) else float("nan")
        self.Rp = float(rg[sc[-1]]) if len(sc) else float("nan")
        self.kappa_res = float("nan")

    def fields(self, r):
        r = np.asarray(r, float)
        return dict(m=self._m(r), mp=self._mp(r), mpp=self._mpp(r), sigma=self._sig(r), sigp=self._dls(r) * self._sig(r))

    def dln_sigma(self, r):
        return self._dls(np.asarray(r, float))


def hayward_core(ell=0.271, M=1.0):
    D = lambda r: r**3 + 2 * M * ell**2
    return AnalyticCore(
        f"Hayward ell={ell}",
        lambda r: M * r**3 / D(r),
        lambda r: 6 * M**2 * ell**2 * r**2 / D(r) ** 2,
        lambda r: 12 * M**2 * ell**2 * r * (2 * M * ell**2 - 2 * r**3) / D(r) ** 3,
        lambda r: 6 * r**3 / D(r),
        lambda r: 3 * (2 * M * ell**2) / D(r),      # dln(sigma)/dln r  -> 3 at r -> 0
        ell, M)


def bardeen_core(g=0.6, M=1.0):
    D = lambda r: (r**2 + g**2) ** 1.5
    return AnalyticCore(
        f"Bardeen g={g}",
        lambda r: M * r**3 / D(r),
        lambda r: 3 * M * g**2 * r**2 / (r**2 + g**2) ** 2.5,
        lambda r: 3 * M * g**2 * r * (2 * g**2 - 3 * r**2) / (r**2 + g**2) ** 3.5,
        lambda r: 5 * r**2 / (r**2 + g**2),
        lambda r: 2 * g**2 / (r**2 + g**2),         # -> 2 at r -> 0
        g, M)


def dymnikova_core(rstar=0.9615, M=1.0):
    """rho = rho_c exp(-r^3/rstar^3), m = M (1 - exp(-r^3/rstar^3)) with M = (4pi/3) rho_c rstar^3."""
    E = lambda r: np.exp(-(r**3) / rstar**3)
    return AnalyticCore(
        f"Dymnikova r*={rstar}",
        lambda r: M * (1 - E(r)),
        lambda r: 3 * M * r**2 / rstar**3 * E(r),
        lambda r: 3 * M * r / rstar**3 * E(r) * (2 - 3 * r**3 / rstar**3),
        lambda r: 3 * r**3 / rstar**3,
        lambda r: 3.0 + 0.0 * r,                    # sigma ~ r^3 exactly => dln sigma/dln r = 3
        rstar, M)


# ----------------------------------------------------------------- core quantities
def core_quantities(core, r):
    """f, axial bracket B, kappa_MS = c_t^2, and the direct fluid c_t^2 = dp_t/drho."""
    r = np.asarray(r, float)
    fl = core.fields(r)
    m, mp, mpp, sig = fl["m"], fl["mp"], fl["mpp"], fl["sigma"]
    f = 1 - 2 * m / r
    kappa = sig / 2 - 1 - 0.5 * core.dln_sigma(r)
    # direct fluid route: 8 pi rho = 2 m'/r^2, p_r = -rho, p_t = -rho - (r/2) rho'
    rho = mp / (4 * np.pi * r**2)
    rhop = (mpp * r - 2 * mp) / (4 * np.pi * r**3)
    pt = -rho - 0.5 * r * rhop
    ptp = np.gradient(pt, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ct2_direct = ptp / np.gradient(rho, r)
    return dict(f=f, m=m, mp=mp, mpp=mpp, sigma=sig, kappa=kappa, ct2_direct=ct2_direct, rho=rho)


def axial_potential(core, r, l):
    fl = core.fields(np.asarray(r, float))
    m, mp = fl["m"], fl["mp"]
    r = np.asarray(r, float)
    f = 1 - 2 * m / r
    B = l * (l + 1) - 6 * m / r + 2 * mp
    return f * B / r**2, B, f


def tortoise(core, r):
    f = 1 - 2 * core.fields(r)["m"] / r
    return cumulative_trapezoid(1 / f, r, initial=0.0)


# ----------------------------------------------------------------- spectral solvers
# The master equation d^2u/dr*^2 = (f W - omega^2) u with dr* = dr/f is, in Sturm-Liouville form on r,
#     -(f u')' + W u = omega^2 (1/f) u,     W = V_ax/f (axial) or the Moreno-Sarbach matrix V (polar).
# A uniform grid in r* cannot be used here: the tortoise span is dominated by the throat at R_-, so the
# short-scale well near the centre would be unresolved.  Working directly on a uniform r grid fixes that.
def _sl_matrices(core, r, W_diag_blocks):
    """Assemble A (block tridiagonal) and B = diag(1/f) for one or two coupled channels."""
    h = r[1] - r[0]
    fh = 1 - 2 * core.fields(0.5 * (r[1:] + r[:-1]))["m"] / (0.5 * (r[1:] + r[:-1]))   # f at half-steps
    f = 1 - 2 * core.fields(r)["m"] / r
    k = len(r) - 2
    main = (fh[1:] + fh[:-1]) / h**2
    off = -fh[1:-1] / h**2
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    nch = len(W_diag_blocks)
    A = np.zeros((nch * k, nch * k))
    for i in range(nch):
        for j in range(nch):
            blk = np.diag(W_diag_blocks[i][j][1:-1])
            A[i * k:(i + 1) * k, j * k:(j + 1) * k] = (lap + blk) if i == j else blk
    s = np.sqrt(f[1:-1])
    S = np.tile(s, nch)
    return (A * S[:, None]) * S[None, :]        # B^{-1/2} A B^{-1/2}, symmetric


def axial_sl_min(core, r_lo, r_hi, l, n=4001):
    r = np.linspace(r_lo, r_hi, n)
    W = axial_potential(core, r, l)[1] / r**2          # V_ax / f = B / r^2
    C = _sl_matrices(core, r, [[W]])
    w = np.linalg.eigvalsh(C)
    return float(w[0]), float(w[1])


def ms_two_channel_min(core, r_lo, r_hi, l, n=2001):
    """Lowest omega^2 of the 2x2 Moreno-Sarbach system.  polar_qnm.potential_matrix supplies V; its kappa
    guard (kappa := 1 where sigma < 1e-12) is replaced by the well-conditioned core expression."""
    r = np.linspace(r_lo, r_hi, n)
    N, V11, V12, V22, info = pq.potential_matrix(_BgAdapter(core), r, l)
    kap = core_quantities(core, r)["kappa"]
    V22 = V22 + (kap - info["kappa"]) * l * (l + 1) / r**2
    C = _sl_matrices(core, r, [[V11, V12], [V12, V22]])
    w, vec = np.linalg.eigh(C)
    k = len(r) - 2
    v = vec[:, 0]
    wPhi = float(np.sum(v[k:] ** 2) / np.sum(v**2))          # weight of the matter channel in the mode
    rpeak = float(r[1:-1][np.argmax(v[k:] ** 2 + v[:k] ** 2)])
    return float(w[0]), float(np.min(N * V22)), wPhi, rpeak


class _BgAdapter:
    """polar_qnm.potential_matrix expects .fields() and .scen."""

    def __init__(self, core):
        self._c = core
        self.scen = getattr(core, "scen", "analytic")

    def fields(self, r):
        return self._c.fields(r)


# ----------------------------------------------------------------- time domain
def evolve_core(core, l, r_lo, r_hi, drs, t_end, channels="axial"):
    """Leapfrog in (t, r*) on the core, Dirichlet at both ends.  Returns t, max|u| history."""
    r = np.linspace(r_lo, r_hi, 400001)
    rs = tortoise(core, r)
    grid = np.arange(rs[0], rs[-1], drs)
    rr = np.interp(grid, rs, r)
    if channels == "axial":
        U11, _, _ = axial_potential(core, rr, l)
        U12 = U22 = np.zeros_like(U11)
    else:
        N, V11, V12, V22, info = pq.potential_matrix(_BgAdapter(core), rr, l)
        kap = core_quantities(core, rr)["kappa"]
        V22 = V22 + (kap - info["kappa"]) * l * (l + 1) / rr**2
        U11, U12, U22 = N * V11, N * V12, N * V22
    dt = 0.4 * drs
    nst = int(t_end / dt)
    x0 = grid[0] + 0.25 * (grid[-1] - grid[0])
    wid = 0.06 * (grid[-1] - grid[0])
    psi = np.exp(-((grid - x0) ** 2) / (2 * wid**2))
    phi = psi.copy() if channels != "axial" else np.zeros_like(psi)
    pp, fp = psi.copy(), phi.copy()
    lap = lambda u: (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / drs**2
    ts, amp = [], []
    for nstep in range(nst):
        pn = 2 * psi - pp + dt**2 * (lap(psi) - U11 * psi - U12 * phi)
        fn = 2 * phi - fp + dt**2 * (lap(phi) - U12 * psi - U22 * phi)
        pn[0] = pn[-1] = fn[0] = fn[-1] = 0.0
        pp, psi, fp, phi = psi, pn, phi, fn
        ts.append((nstep + 1) * dt)
        amp.append(float(max(np.max(np.abs(psi)), np.max(np.abs(phi)))))
        if amp[-1] > 1e80 or not np.isfinite(amp[-1]):
            break
    return np.array(ts), np.array(amp), len(grid)


def growth_fit(t, a, frac=(0.4, 0.9)):
    i0, i1 = int(frac[0] * len(t)), int(frac[1] * len(t))
    if i1 - i0 < 10:
        return float("nan")
    y = np.log(np.maximum(a[i0:i1], 1e-300))
    return float(np.polyfit(t[i0:i1], y, 1)[0])


# ----------------------------------------------------------------- main
def main():
    res = {}
    core = ScenarioCore()
    say("T37 item 1 -- stability of the de Sitter core (0 < r < R_-) to l >= 2 perturbations")
    say(f"Background: {core.name}; ell/M = {core.ell:.6f}; R_- = {core.Rm:.6f} (residual kappa_- = {core.kappa_res:.2e}); R_+ = {core.Rp:.6f}")
    res["background"] = dict(name=core.name, ell=core.ell, R_minus=core.Rm, R_plus=core.Rp, kappa_residual=core.kappa_res)

    # ---------------- 1. AXIAL
    say("\n--- AXIAL sector: V_ax = f [ l(l+1) - 6m/r + 2m' ] / r^2")
    r = np.linspace(1e-4, core.Rm * (1 - 1e-4), 200001)
    q = core_quantities(core, r)
    say(f"  core checks: max 2m/r = {np.max(2 * q['m'] / r):.8f} (<= 1 by definition of the inner horizon); "
        f"min m' = {q['mp'].min():.3e} (>= 0 <=> rho >= 0); min f = {q['f'].min():.3e}")
    ax_rows = []
    for l in (2, 3, 4, 5, 10, 20, 50):
        V, B, f = axial_potential(core, r, l)
        # V_ax and f share a sign; the meaningful test is B > 0 (= V_ax/f * r^2).  Where the numerically tuned
        # f dips below zero (|f| < 5e-9, the residual mistuning kappa_- = -4.9e-6) V_ax is negative only
        # because f is: that is a tuning artefact, not a feature of the potential.
        ax_rows.append(dict(l=l, B_min=float(B.min()), r_at_B_min=float(r[np.argmin(B)]),
                            bound=float(l * (l + 1) - 3), B_positive=bool(np.all(B > 0)),
                            V_min_where_f_pos=float(V[f > 0].min())))
        say(f"  l={l:3d}: min B = {B.min():.6f} at r = {r[np.argmin(B)]:.4f} (analytic bound l(l+1)-3 = {l*(l+1)-3}); "
            f"B > 0 everywhere: {bool(np.all(B > 0))}; min V_ax where f > 0: {V[f > 0].min():+.3e}")
    res["axial_potential"] = ax_rows
    say("  => B >= l(l+1) - 3 >= 3 for l >= 2 because 2m/r < 1 (so 6m/r < 3) and m' >= 0.  This uses nothing")
    say("     about the profile beyond rho >= 0 and the absence of trapped surfaces inside R_-.")

    say("\n  lowest two eigenvalues omega^2 of -(f u')' + (B/r^2) u = omega^2 u/f (Dirichlet on [r_lo, r_hi]):")
    ax_spec = []
    for l in (2, 5, 10):
        for r_hi in (0.5, 0.6, 0.65):
            w0, w1 = axial_sl_min(core, 1e-3, r_hi, l)
            ax_spec.append(dict(l=l, r_hi=r_hi, omega2_0=w0, omega2_1=w1))
            say(f"    l={l:3d}, r_hi = {r_hi}: omega^2 = {w0:+.6e}, {w1:+.6e};  growing mode: {w0 < 0}")
    res["axial_spectrum"] = ax_spec

    say("\n  time evolution in the core (axial), l = 2, domain [0.01, 0.6] in r (r* span 30.3):")
    ax_td = []
    for drs in (0.04, 0.02, 0.01):
        t, a, npts = evolve_core(core, 2, 0.01, 0.6, drs, 200.0, "axial")
        g = growth_fit(t, a)
        ax_td.append(dict(drs=drs, npoints=npts, t_end=200.0, rate=g, amp_ratio=float(a[-1] / a[0])))
        say(f"    dr* = {drs}: N = {npts}, fitted d ln|u|/dt = {g:+.3e}, |u|_end/|u|_0 = {a[-1] / a[0]:.3e}")
    t, a, npts = evolve_core(core, 2, 0.01, 0.6, 0.04, 4000.0, "axial")
    g = growth_fit(t, a)
    ax_td.append(dict(drs=0.04, npoints=npts, t_end=4000.0, rate=g, amp_ratio=float(a[-1] / a[0])))
    say(f"    long run to t = 4000 M (dr* = 0.04): d ln|u|/dt = {g:+.3e}, |u|_end/|u|_0 = {a[-1] / a[0]:.3e}")
    res["axial_timedomain"] = ax_td

    # ---------------- 2. POLAR: c_t^2 = kappa_MS
    say("\n--- POLAR sector: identification kappa_MS = c_t^2 = dp_t/drho")
    rr = np.linspace(2e-2, core.Rm * (1 - 1e-3), 40001)
    qq = core_quantities(core, rr)
    ok = np.isfinite(qq["ct2_direct"])
    dev = np.max(np.abs(qq["ct2_direct"][ok] - qq["kappa"][ok]))
    say(f"  max |c_t^2(direct, from p_t = -rho - (r/2) rho') - kappa_MS(sigma)| over 0.02 < r < R_- : {dev:.3e}")
    res["ct2_identity_max_dev"] = float(dev)

    say("\n  c_t^2 at the centre for regular cores (analytic: rho = rho_c(1 - (r/L)^p + ...) => c_t^2 -> -1 - p/2):")
    tab = []
    for c, p in ((core, 4), (hayward_core(), 3), (bardeen_core(), 2), (dymnikova_core(), 3)):
        rs = np.geomspace(1e-5, 1e-3, 2001)
        k = core_quantities(c, rs)["kappa"]
        tab.append(dict(name=c.name, p=p, ct2_centre_numeric=float(k[0]), ct2_centre_analytic=-1 - p / 2,
                        R_minus=float(c.Rm)))
        say(f"    {c.name:34s}: p = {p}, c_t^2(0) numeric = {k[0]:+.6f}, analytic -1 - p/2 = {-1 - p/2:+.3f}")
    res["ct2_centre"] = tab
    say("  => every de Sitter plateau has c_t^2(0) = -1 - p/2 < -1: the tangential sound speed squared is")
    say("     negative (and |c_t| > 1) at the centre for ANY regular profile, not only for this one.")
    say("     The sharper bound c_t^2(0) <= -2 needs p >= 2 (p = 1 gives -1.5); every profile above has")
    say("     p >= 2.  The polar sector needs the full centre coefficient -(p+2)/4 [2l(l+1) - (p+4)] < -1/4,")
    say("     which holds for every l >= 2 when p <= 7 (all profiles here); see t37_centre_limit.py.")

    rfull = np.linspace(1e-3, core.Rm * (1 - 1e-4), 200001)
    kap = core_quantities(core, rfull)["kappa"]
    neg = rfull[kap < 0]
    say(f"\n  c_t^2 < 0 on r in [{neg.min():.4f}, {neg.max():.4f}] = {(kap < 0).mean() * 100:.1f}% of the core; "
        f"min c_t^2 = {kap.min():.4f}; c_t^2(R_-^-) = {kap[-1]:+.4f}")
    res["ct2_negative_region"] = dict(r_lo=float(neg.min()), r_hi=float(neg.max()), fraction=float((kap < 0).mean()),
                                      ct2_min=float(kap.min()), ct2_at_Rminus=float(kap[-1]))

    # eikonal growth rate
    say("\n  eikonal growth rate  nu = sqrt( f |c_t^2| l(l+1) ) / r  (Killing time), maximised over r > r_cut:")
    eik = []
    fful = 1 - 2 * core.fields(rfull)["m"] / rfull
    for l in (2, 3, 5, 10, 20, 50):
        row = dict(l=l)
        for rc in (0.2, 0.1, 0.05, 0.02):
            msk = (rfull > rc) & (kap < 0)
            nu = np.sqrt(fful[msk] * (-kap[msk]) * l * (l + 1)) / rfull[msk]
            row[f"nu_rcut_{rc}"] = float(nu.max())
        eik.append(row)
        say(f"    l={l:3d}: nu_max = " + ", ".join(f"{row[f'nu_rcut_{rc}']:9.3f} (r>{rc})" for rc in (0.2, 0.1, 0.05, 0.02)) + "  [1/M]")
    res["eikonal_rate"] = eik
    say("    nu grows like sqrt(l(l+1)) and like 1/r_cut: there is no fastest mode (Hadamard ill-posed).")

    # two-channel Moreno-Sarbach eigenvalue problem
    say("\n  two-channel Moreno-Sarbach eigenvalue solve on [r_cut, 0.6], Dirichlet, lowest omega^2:")
    ms = []
    for l in (2, 3, 5, 10, 20):
        for rc in (0.2, 0.1, 0.05, 0.02):
            w0, u22min, wPhi, rpk = ms_two_channel_min(core, rc, 0.6, l, n=2001)
            ms.append(dict(l=l, r_cut=rc, omega2_min=w0, rate=float(np.sqrt(-w0)) if w0 < 0 else 0.0,
                           NV22_min=u22min, matter_channel_weight=wPhi, r_peak=rpk))
            say(f"    l={l:3d}, r_cut = {rc:4g}: omega^2_min = {w0:+.5e} -> growth rate {np.sqrt(max(-w0, 0)):9.3f}/M; "
                f"min N V22 = {u22min:.3e}; matter-channel weight {wPhi:.4f}; mode peaks at r = {rpk:.4f}")
    res["ms_eigen"] = ms

    say("\n  time evolution of the same two-channel system in the core (l = 2), inner cut moved inwards:")
    pol_td = []
    for rc, drs in ((0.2, 0.004), (0.1, 0.002), (0.05, 0.001), (0.02, 0.0004)):
        t, a, npts = evolve_core(core, 2, rc, 0.45, drs, 6.0, "polar")
        g = growth_fit(t, a)
        pol_td.append(dict(r_cut=rc, drs=drs, npoints=npts, rate=g, t_end=float(t[-1]), amp_ratio=float(a[-1] / a[0])))
        say(f"    r_cut = {rc:4g}, dr* = {drs:g}: N = {npts}, fitted d ln|u|/dt = {g:+.3f}/M, ran to t = {t[-1]:.3g}M, |u|_end/|u|_0 = {a[-1] / a[0]:.3e}")
    res["polar_timedomain"] = pol_td

    # physical numbers for 10 solar masses
    GMsun_c3 = 4.925490947e-6
    say("\n  physical scales for M = 10 M_sun (GM_sun/c^3 = 4.9255e-6 s, M = 1.477e4 m), from the MS eigenvalues:")
    phys = []
    for row in ms:
        if row["rate"] <= 0:
            continue
        nu = row["rate"]
        te = 1 / nu * 10 * GMsun_c3
        if (row["l"], row["r_cut"]) in ((2, 0.1), (2, 0.02), (10, 0.1), (20, 0.02)):
            phys.append(dict(l=row["l"], r_cut=row["r_cut"], nu_per_M=nu, e_fold_M=1 / nu, e_fold_seconds=te,
                             length_scale_m=row["r_cut"] * 1.477e4))
            say(f"    l={row['l']:2d}, r_cut = {row['r_cut']:4g} M ({row['r_cut'] * 1.477e4 / 1e3:.2f} km): nu = {nu:.2f}/M "
                f"-> e-folding {1 / nu:.4f} M = {te * 1e6:.3f} microseconds")
    res["physical_10Msun"] = phys

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for l in (2, 5, 10):
        V, B, f = axial_potential(core, rfull, l)
        ax[0].plot(rfull, V, label=f"l={l}")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("r / M"); ax[0].set_ylabel(r"$V_{\rm ax}$"); ax[0].set_yscale("symlog", linthresh=1e-2)
    ax[0].set_title("axial potential in the core (positive everywhere)"); ax[0].legend()
    ax[1].plot(rfull, kap, "C3")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axhline(-3, color="C0", ls=":", label=r"$-1-p/2$, $p=4$")
    ax[1].set_xlabel("r / M"); ax[1].set_ylabel(r"$c_t^2 = \kappa_{\rm MS}$")
    ax[1].set_title("tangential sound speed squared"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "core_potentials.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in pol_td:
        ax.plot([], [])
    ls = np.array([r["l"] for r in ms if r["r_cut"] == 0.1])
    rates = np.array([r["rate"] for r in ms if r["r_cut"] == 0.1])
    ax.plot(np.sqrt(ls * (ls + 1)), rates, "o-", label="MS eigenvalue, r_cut = 0.1")
    eik01 = [row["nu_rcut_0.1"] for row in eik if row["l"] in list(ls)]
    ax.plot(np.sqrt(ls * (ls + 1)), eik01, "s--", label="eikonal, r > 0.1")
    ax.set_xlabel(r"$\sqrt{l(l+1)}$"); ax.set_ylabel(r"growth rate $\nu$ [1/M]")
    ax.set_title("polar growth rate is linear in l (no cutoff)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "polar_growth_vs_l.png", dpi=130)
    plt.close(fig)

    (OUT / "t37_core_stability.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (LOGS / "t37_core_stability_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
