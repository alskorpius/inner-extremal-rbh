"""Inhomogeneous spherical collapse with curvature regularization (LTB class, synchronous comoving coordinates).

ds^2 = -dtau^2 + R'^2/(1+2E(r)) dr^2 + R^2(tau,r) dOmega^2,  G = c = 1, M_tot = 1.
Each shell r moves according to Rdot^2 = 2 m(M(r), R)/R + 2E(r) with the effective Misner-Sharp mass m(M, R):
  Hayward:   m = M R^3/(R^3 + 2 M ell^2)   (ell is the global scale: ell = 0.271 M_tot per the scenario, or "fundamental");
  local:     ell = lam*M(r)                  (the shell "sees" its own interior mass);
  scenario:  branch-A profile (triple root) with fixed ell and mass M(r) (tabulated, experiments/stability/triple_root_monotone.py);
  ell = 0 -- classical LTB (control: cycloid).
Shell equation of motion: Rddot = F(R, M) = -m/R^2 + m_R/R; variation Y = dR/dr: Yddot = F_R Y + F_M M'(r).
Effective source (from the Einstein equations, verified in verify_ltb.py): T^tau_r = 0,
  rho = (m_M M' + m_R Y)/(4 pi R^2 Y) = rho_d + rho_v,  p_r = -rho_v,  p_perp = p_r + R p_r'/(2Y),
  K = 48 Psi2^2 + 2 R_ab R^ab - Rs^2/3,  Psi2 = -m/R^3 + (4 pi/3)(rho - p_r + p_perp).
Apparent horizons: Rdot^2 = 1 + 2E (2m/R = 1); Hayward-Kodama dynamical surface gravity kappa = Box_2 R/2.
Shell crossing: Y = 0 (density and K diverge) -- outside the scope of the model.
Mass profile: M(r) = M_tot x^3 (1 + eps (1 - x^2)), x = r/r0 (eps > 0 -- dense center, eps < 0 -- dense periphery); outside M = M_tot.
Run: python src/inhomogeneous_collapse/ltb_bounce.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "stability"))
sys.path.insert(0, str(HERE.parents[0] / "approach_map"))
import triple_root_monotone as trm  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- effective mass models m(M, R)
class SympyMass:
    def __init__(self, name, expr, Msym, Rsym, ell_label):
        self.name, self.ell_label = name, ell_label
        d = dict(m=expr, m_R=sp.diff(expr, Rsym), m_M=sp.diff(expr, Msym), m_RR=sp.diff(expr, Rsym, 2), m_RM=sp.diff(expr, Rsym, Msym))
        self.f = {k: sp.lambdify((Msym, Rsym), v, "numpy") for k, v in d.items()}

    def all(self, M, R):
        return {k: np.asarray(fn(M, R), float) + 0.0 * R for k, fn in self.f.items()}


def hayward_mass(ell):
    M, R = sp.symbols("M R", positive=True)
    return SympyMass("hayward", M * R**3 / (R**3 + 2 * M * ell**2) if ell > 0 else M, M, R, f"ell={ell}")


def local_mass(lam):
    M, R = sp.symbols("M R", positive=True)
    return SympyMass("local", M * R**3 / (R**3 + 2 * lam**2 * M**3), M, R, f"ell={lam}*M(r)")


class ScenarioMass:
    """Branch-A profile with fixed ell: R1(M) = (2 M ell^2/(3 Itot))^{1/3}, m = (M/Itot) mI(R/R1)."""

    def __init__(self):
        base = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
        s1s, u_s, info = trm.find_merge_first(sigma_min=1.0, base=base)
        kw = info["kw"]
        d = s1s - 1.0
        sig, s, mI, H, Hp = trm.profile(d, **kw)
        scale = 1.0 / float(np.interp(u_s, trm.U, H))
        fam = trm.MonotoneFamily(d, scale, kw=kw)
        self.fam = fam
        self.ell = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
        self.rho_c = 3 / (8 * np.pi * self.ell**2)
        self.LU, self.ln_s, self.sig, self.mI, self.Itot = trm.LU, np.log(s), sig, mI, float(mI[-1])
        self.name, self.ell_label = "scenario", f"ell={self.ell:.4f} (branch A, triple root at M=1)"

    def all(self, M, R):
        M = np.asarray(M, float) + 0.0 * np.asarray(R, float)
        R = np.asarray(R, float) + 0.0 * M
        R1 = (2 * M * self.ell**2 / (3 * self.Itot)) ** (1 / 3)
        lu = np.log(R / R1)
        s = np.exp(np.interp(lu, self.LU, self.ln_s, left=0.0, right=-np.inf))
        sig = np.interp(lu, self.LU, self.sig, left=0.0, right=self.sig[-1])
        u = R / R1
        mI = np.where(lu < self.LU[0], u**3 / 3, np.interp(lu, self.LU, self.mI, right=self.Itot))
        m = (M / self.Itot) * mI
        m_R = 4 * np.pi * R**2 * self.rho_c * s
        m_M = mI / self.Itot - s * u**3 / (3 * self.Itot)
        m_RR = 4 * np.pi * R * self.rho_c * s * (2 - sig)
        m_RM = 4 * np.pi * R**2 * self.rho_c * sig * s / (3 * M)
        return dict(m=m, m_R=m_R, m_M=m_M, m_RR=m_RR, m_RM=m_RM)


# ---------------------------------------------------------------- mass profile M(r)
class MassProfile:
    def __init__(self, eps, r0=20.0, Mtot=1.0):
        self.eps, self.r0, self.Mtot = eps, r0, Mtot

    def M(self, r):
        x = np.minimum(np.asarray(r, float) / self.r0, 1.0)
        return self.Mtot * x**3 * (1 + self.eps * (1 - x**2))

    def Mp(self, r):
        r = np.asarray(r, float)
        x = r / self.r0
        inside = x < 1.0
        val = self.Mtot / self.r0 * (3 * (1 + self.eps) * x**2 - 5 * self.eps * x**4)
        return np.where(inside, val, 0.0)

    def rho0(self, r):
        return self.Mp(r) / (4 * np.pi * np.asarray(r, float) ** 2)


# ---------------------------------------------------------------- shell dynamics
def shell_rhs_factory(mm, M, Mp):
    def rhs(t, y):
        R, V, Y, W = y
        d = mm.all(M, R)
        m, mR, mM, mRR, mRM = d["m"], d["m_R"], d["m_M"], d["m_RR"], d["m_RM"]
        F = -m / R**2 + mR / R
        FR = 2 * m / R**3 - 2 * mR / R**2 + mRR / R
        FM = -mM / R**2 + mRM / R
        return [V, float(F), W, float(FR * Y + FM * Mp)]
    return rhs


def integrate_shell(mm, prof, r, tau_grid, rtol=1e-11, atol=1e-14, E_override=None):
    """E_override: set the shell energy (e.g. 0 -- marginally bound); then start with V0 = -sqrt(2m/r + 2E)."""
    M, Mp = float(prof.M(r)), float(prof.Mp(r))
    d0 = mm.all(M, r)
    if E_override is None:
        E = -float(d0["m"]) / r
        V0, W0 = 0.0, 0.0
    else:
        E = float(E_override)
        V0 = -np.sqrt(2 * float(d0["m"]) / r + 2 * E)
        W0 = ((float(d0["m_M"]) * Mp + float(d0["m_R"])) / r - float(d0["m"]) / r**2) / V0
    rhs = shell_rhs_factory(mm, M, Mp)

    def ev_bounce(t, y):
        return y[1]
    ev_bounce.direction = 1

    def ev_cross(t, y):
        return y[2]

    def ev_horizon(t, y):
        return y[1] ** 2 - (1 + 2 * E)

    def ev_threshold(t, y):
        return float(mm.all(M, y[0])["m"]) - 0.5 * M
    ev_threshold.direction = -1

    def ev_crush(t, y):
        return y[0] - 1e-9
    ev_crush.terminal = True
    ev_crush.direction = -1

    sol = solve_ivp(rhs, (0.0, tau_grid[-1]), [r, V0, 1.0, W0], method="DOP853", rtol=rtol, atol=atol,
                    t_eval=tau_grid, events=[ev_bounce, ev_cross, ev_horizon, ev_threshold, ev_crush], dense_output=False)
    def pad(a):
        b = np.full(len(tau_grid), np.nan)
        b[: len(a)] = a
        return b
    out = dict(r=r, M=M, Mp=Mp, E=E, t=tau_grid, R=pad(sol.y[0]), V=pad(sol.y[1]), Y=pad(sol.y[2]), W=pad(sol.y[3]), status=sol.status, t_last=float(sol.t[-1]),
               t_bounce=sol.t_events[0], t_cross=sol.t_events[1], t_horizon=sol.t_events[2], y_horizon=sol.y_events[2],
               t_thr=sol.t_events[3], crushed=(len(sol.t_events[4]) > 0) or (sol.status == -1), t_crush=sol.t_events[4])
    # energy integral as an accuracy check
    d = mm.all(M, sol.y[0])
    out["energy_residual"] = float(np.nanmax(np.abs(sol.y[1] ** 2 - 2 * d["m"] / sol.y[0] - 2 * E)))
    return out


def fields(mm, prof, r, R, V, Y, W):
    """Effective source and invariants on the shell grid (vectorized over tau)."""
    M, Mp = float(prof.M(r)), float(prof.Mp(r))
    d = mm.all(M, R)
    m, mR, mM, mRR, mRM = d["m"], d["m_R"], d["m_M"], d["m_RR"], d["m_RM"]
    rho_v = mR / (4 * np.pi * R**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_d = mM * Mp / (4 * np.pi * R**2 * Y)
        drho_v_dR = (mRR - 2 * mR / R) / (4 * np.pi * R**2)
        drho_v_dM = mRM / (4 * np.pi * R**2)
        p_r = -rho_v
        p_perp = p_r - R * (drho_v_dM * Mp + drho_v_dR * Y) / (2 * Y)
        rho = rho_d + rho_v
        Psi2 = -m / R**3 + (4 * np.pi / 3) * (rho - p_r + p_perp)
        T = -rho + p_r + 2 * p_perp
        Ric = 8 * np.pi * np.array([-rho - T / 2, p_r - T / 2, p_perp - T / 2, p_perp - T / 2])
        RabRab = np.sum(Ric**2, axis=0)
        Rs = -8 * np.pi * T
        K = 48 * Psi2**2 + 2 * RabRab - Rs**2 / 3
        F = -m / R**2 + mR / R
        E = -float(mm.all(M, r)["m"]) / r
        d0 = mm.all(M, r)
        Ep = -(d0["m_M"] * Mp + d0["m_R"]) / r + d0["m"] / r**2
        kappa = 0.5 * (-F - V * W / Y + Ep / Y)
    return dict(m=m, rho=rho, rho_d=rho_d, rho_v=rho_v, p_r=p_r, p_perp=p_perp, K=K, kappa=kappa, E=E, Ep=Ep,
                nec_perp=rho + p_perp, nec_r=rho + p_r, sec=rho + p_r + 2 * p_perp)


def static_K(mm, M, R):
    """Kretschmann invariant of the static metric f = 1 - 2m(R)/R (exterior region, M' = 0) -- independent check."""
    d = mm.all(M, R)
    m, mR, mRR = d["m"], d["m_R"], d["m_RR"]
    f = 1 - 2 * m / R
    fp = 2 * m / R**2 - 2 * mR / R
    fpp = -4 * m / R**3 + 4 * mR / R**2 - 2 * mRR / R
    return fpp**2 + 4 * fp**2 / R**2 + 4 * (1 - f) ** 2 / R**4


def static_horizons(mm, M, Rmax=4.0):
    Rg = np.geomspace(1e-4, Rmax, 40001)
    h = 2 * mm.all(M, Rg)["m"] / Rg - 1
    idx = np.nonzero(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]
    res = []
    for i in idx:
        Rr = brentq(lambda R: 2 * float(mm.all(M, R)["m"]) / R - 1, Rg[i], Rg[i + 1], xtol=1e-14)
        d = mm.all(M, Rr)
        fp = 2 * d["m"] / Rr**2 - 2 * d["m_R"] / Rr
        res.append((Rr, 0.5 * float(fp)))
    return res


def cycloid_R(r, M, tau):
    """Classical LTB from rest: R = (r/2)(1 + cos eta), tau = r^{3/2}/(2 sqrt(2M)) (eta + sin eta)."""
    out = np.empty_like(tau)
    for i, tt in enumerate(tau):
        g = lambda e: r**1.5 / (2 * np.sqrt(2 * M)) * (e + np.sin(e)) - tt
        if g(np.pi) < 0:
            out[i] = np.nan
        else:
            e = brentq(g, 0, np.pi, xtol=1e-14)
            out[i] = r / 2 * (1 + np.cos(e))
    return out


# ---------------------------------------------------------------- single case
def run_case(mm, eps, r0=20.0, n_in=160, n_out=24, r_max=30.0, tau_end=None, label=""):
    prof = MassProfile(eps, r0)
    r_in = r0 * np.linspace(0.0, 1.0, n_in + 1)[1:]
    r_out = np.linspace(r0, r_max, n_out + 2)[1:-1]
    rs = np.concatenate([r_in, r_out])
    tau_c0 = np.pi * r0**1.5 / (2**1.5 * np.sqrt(prof.Mtot))       # classical collapse time of the surface
    tau_end = tau_end or 2.3 * tau_c0
    tau = np.linspace(0.0, tau_end, 2301)
    t0 = time.time()
    shells = [integrate_shell(mm, prof, r, tau) for r in rs]
    R = np.array([s["R"] for s in shells]); V = np.array([s["V"] for s in shells])
    Y = np.array([s["Y"] for s in shells]); W = np.array([s["W"] for s in shells])
    fl = [fields(mm, prof, s["r"], s["R"], s["V"], s["Y"], s["W"]) for s in shells]
    K = np.array([f["K"] for f in fl]); rho = np.array([f["rho"] for f in fl]); rho_v = np.array([f["rho_v"] for f in fl])
    rho_d = np.array([f["rho_d"] for f in fl]); p_perp = np.array([f["p_perp"] for f in fl]); nec = np.array([f["nec_perp"] for f in fl])
    res = dict(label=label, model=mm.name, ell_label=mm.ell_label, eps=eps, r0=r0, n_in=n_in, n_out=n_out, tau_end=tau_end, tau_c_classical=tau_c0,
               energy_residual_max=max(s["energy_residual"] for s in shells), runtime_s=time.time() - t0)
    # --- bounce, threshold, crossings
    tb = np.array([s["t_bounce"][0] if len(s["t_bounce"]) else np.nan for s in shells])
    tthr = np.array([s["t_thr"][0] if len(s["t_thr"]) else np.nan for s in shells])
    tx = np.array([s["t_cross"][0] if len(s["t_cross"]) else np.nan for s in shells])
    crushed = np.array([s["crushed"] for s in shells])
    res["n_crushed"] = int(crushed.sum())
    res["bounce"] = dict(tau_min=float(np.nanmin(tb)) if np.isfinite(tb).any() else None, tau_max=float(np.nanmax(tb)) if np.isfinite(tb).any() else None,
                         n_bounced=int(np.isfinite(tb).sum()), n_shells=len(rs))
    Rb = np.array([np.interp(t, s["t"], s["R"]) if np.isfinite(t) else np.nan for t, s in zip(tb, shells)])
    Kb = np.array([np.interp(t, s["t"], f["K"]) if np.isfinite(t) else np.nan for t, s, f in zip(tb, shells, fl)])
    res["bounce"].update(R_b_over_r=[float(v) for v in (Rb / rs)[::20]], r_sample=[float(v) for v in rs[::20]],
                         K_b_sample=[float(v) for v in Kb[::20]], tau_b_sample=[float(v) for v in tb[::20]], tau_thr_sample=[float(v) for v in tthr[::20]])
    # first shell crossing (over the whole set)
    if np.isfinite(tx).any():
        i = int(np.nanargmin(tx))
        s = shells[i]
        Rx = float(np.interp(tx[i], s["t"], s["R"]))
        after_bounce = bool(np.isfinite(tb[i]) and tx[i] > tb[i])
        d = mm.all(s["M"], Rx)
        trapped = bool(2 * float(d["m"]) / Rx > 1)
        # K just before crossing: at |Y| = 0.1 and 0.01
        Ky = {}
        for yv in (0.1, 0.01):
            j = np.nonzero((s["t"] < tx[i]) & (np.abs(s["Y"]) > yv))[0]
            Ky[str(yv)] = float(fl[i]["K"][j[-1]]) if len(j) else None
        res["first_crossing"] = dict(tau=float(tx[i]), r=float(rs[i]), R=Rx, R_over_ell=None, after_bounce=after_bounce,
                                     tau_after_bounce=float(tx[i] - tb[i]) if np.isfinite(tb[i]) else None, inside_trapped=trapped, K_at_Y=Ky,
                                     n_shells_crossing_by_end=int(np.isfinite(tx).sum()))
    else:
        res["first_crossing"] = None
    # --- horizons: per-shell events, R and kappa
    hz = []
    for s, f in zip(shells, fl):
        for te, ye in zip(s["t_horizon"], s["y_horizon"]):
            Rh, Vh, Yh, Wh = ye
            fh = fields(mm, prof, s["r"], np.array([Rh]), np.array([Vh]), np.array([Yh]), np.array([Wh]))
            hz.append(dict(tau=float(te), r=float(s["r"]), R=float(Rh), V=float(Vh), kappa=float(fh["kappa"][0]), exterior=bool(s["r"] > r0)))
    res["horizon_events"] = hz
    # static horizons of the exterior region, for comparison
    sh = static_horizons(mm, prof.Mtot)
    res["static_horizons_exterior"] = [dict(R=Rr, kappa=k) for Rr, k in sh]
    ext = [h for h in hz if h["exterior"] and h["V"] < 0]
    if ext and sh:
        chk = []
        for h in ext[:6]:
            Rr, k = min(sh, key=lambda p: abs(p[0] - h["R"]))
            chk.append(dict(R_event=h["R"], R_static=Rr, kappa_event=h["kappa"], kappa_static=k))
        res["exterior_horizon_check"] = chk
    # --- check K in the exterior region against the static formula
    ext_idx = np.nonzero(rs > r0)[0]
    if len(ext_idx):
        i = ext_idx[len(ext_idx) // 2]
        Ks = static_K(mm, prof.Mtot, R[i])
        ok = np.isfinite(K[i]) & np.isfinite(Ks) & (Ks > 0)
        res["exterior_K_check_max_rel"] = float(np.max(np.abs(K[i][ok] / Ks[ok] - 1)))
    # --- maximum K in the regular region (before the first crossing of each shell)
    valid = np.ones_like(K, bool)
    for i, s in enumerate(shells):
        if np.isfinite(tx[i]):
            valid[i, tau >= tx[i]] = False
    Kv = np.where(valid, K, np.nan)
    res["K_max_regular"] = float(np.nanmax(Kv))
    res["K_max_regular_at"] = dict(tau=float(tau[np.nanargmax(np.nanmax(Kv, axis=0))]), r=float(rs[np.nanargmax(np.nanmax(Kv, axis=1))]))
    res["min_nec_perp_regular"] = float(np.nanmin(np.where(valid, nec, np.nan)))
    res["min_rho_regular"] = float(np.nanmin(np.where(valid, rho, np.nan)))
    res["max_p_perp_regular"] = float(np.nanmax(np.where(valid, p_perp, np.nan)))
    # --- fraction of the ball's Misner-Sharp mass in the vacuum-like component: int m_R Y dr / m(r0)
    i0 = n_in - 1
    mRY = np.array([mm.all(s["M"], s["R"])["m_R"] * s["Y"] for s in shells[:n_in]])
    m_vac = np.trapezoid(mRY, rs[:n_in], axis=0)
    m_ball = np.array([float(mm.all(shells[i0]["M"], Rv)["m"]) for Rv in R[i0]])
    frac = m_vac / m_ball
    res["vacuum_fraction"] = dict(tau=[float(v) for v in tau[::230]], frac=[float(v) for v in frac[::230]],
                                  frac_at_bounce=float(np.interp(tb[i0], tau, frac)) if np.isfinite(tb[i0]) else None)
    np.savez_compressed(OUT / f"grid_{label}.npz", tau=tau, r=rs, R=R, V=V, Y=Y, K=K, rho=rho, rho_v=rho_v, rho_d=rho_d, p_perp=p_perp)
    return res, dict(tau=tau, rs=rs, R=R, V=V, Y=Y, K=K, shells=shells, tb=tb, tx=tx, hz=hz, prof=prof)


def plot_case(data, res, mm, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tau, rs, R, Y, K, hz = data["tau"], data["rs"], data["R"], data["Y"], data["K"], data["hz"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    sel = list(range(0, len(rs), max(1, len(rs) // 12)))
    for i in sel:
        ax[0, 0].plot(tau, R[i], lw=0.8, color="C0" if rs[i] <= data["prof"].r0 else "C3")
    if hz:
        ax[0, 0].scatter([h["tau"] for h in hz], [h["R"] for h in hz], s=4, c=["k" if h["V"] < 0 else "m" for h in hz], zorder=5)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("tau"); ax[0, 0].set_ylabel("R(tau, r)"); ax[0, 0].set_title(f"{label}: shells (blue -- ball, red -- exterior); dots -- apparent horizons")
    for i in sel:
        ax[0, 1].plot(tau, Y[i], lw=0.8)
    ax[0, 1].axhline(0, color="k", lw=0.5); ax[0, 1].set_xlabel("tau"); ax[0, 1].set_ylabel("Y = dR/dr"); ax[0, 1].set_title("variation: Y = 0 -- shell crossing")
    ax[0, 1].set_ylim(-2, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1, 0].pcolormesh(tau, rs, np.log10(np.abs(K)), shading="auto", vmin=-6, vmax=np.log10(res["K_max_regular"]) + 1)
    if data["tx"] is not None:
        ok = np.isfinite(data["tx"])
        ax[1, 0].plot(data["tx"][ok], rs[ok], "w.", ms=2)
        okb = np.isfinite(data["tb"])
        ax[1, 0].plot(data["tb"][okb], rs[okb], "r.", ms=2)
    ax[1, 0].set_xlabel("tau"); ax[1, 0].set_ylabel("r"); ax[1, 0].set_title("log10 |K|; red -- bounce, white -- shell crossing")
    for i in sel:
        ax[1, 1].plot(tau, np.abs(K[i]), lw=0.8)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("tau"); ax[1, 1].set_ylabel("|K|"); ax[1, 1].set_title("Kretschmann invariant along the shells")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_{label}.png", dpi=110)
    plt.close(fig)


def main():
    say("Inhomogeneous collapse with curvature regularization (LTB class). M_tot = 1, ball r0 = 20 (R0 = 10 R_s), start from rest.")
    results = []
    # ---------- control 0: classical LTB vs. the cycloid
    mm0 = hayward_mass(0.0)
    prof = MassProfile(0.0)
    tau = np.linspace(0, 95, 951)
    s = integrate_shell(mm0, prof, 20.0, tau)
    Rc = cycloid_R(20.0, 1.0, tau)
    ok = np.isfinite(Rc) & (Rc > 1e-3)
    err = np.max(np.abs(s["R"][ok] / Rc[ok] - 1))
    say(f"C0. Classical (ell=0), surface r0=20: max relative deviation from the cycloid {err:.2e}; integrator reached tau={s['t_last']:.6f} (status {s['status']}) (cycloid: singularity at {np.pi*20**1.5/2**1.5:.6f})")
    # classical crossing for eps<0
    for eps in (-0.1, 0.1):
        r_, ok_ = run_case(mm0, eps, n_in=80, n_out=4, tau_end=110, label=f"classical_eps{eps}")
        fc = r_["first_crossing"]
        say(f"C0. Classical eps={eps}: first shell crossing: {fc and dict(tau=round(fc['tau'],3), r=round(fc['r'],3), R=fc['R'])}; crushed shells: {r_['n_crushed']} out of {len(ok_['rs'])}")
        results.append(r_)
    # ---------- models
    scen = ScenarioMass()
    say(f"\nScenario: ell = {scen.ell:.5f} M_tot, rho_c = {scen.rho_c:.4f}, Itot = {scen.Itot:.5f}")
    # check tabulated derivatives against finite differences, and agreement of m(1,R) with the branch-A backbone
    Rt = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
    d = scen.all(1.0, Rt); h = 1e-6
    fd_R = (scen.all(1.0, Rt + h)["m"] - scen.all(1.0, Rt - h)["m"]) / (2 * h)
    fd_M = (scen.all(1.0 + h, Rt)["m"] - scen.all(1.0 - h, Rt)["m"]) / (2 * h)
    fd_RR = (scen.all(1.0, Rt + h)["m_R"] - scen.all(1.0, Rt - h)["m_R"]) / (2 * h)
    fd_RM = (scen.all(1.0 + h, Rt)["m_R"] - scen.all(1.0 - h, Rt)["m_R"]) / (2 * h)
    m_fam = np.interp(Rt, scen.fam._Rg, scen.fam._mg)
    say(f"  derivative check (max relative difference vs finite differences): m_R {np.max(np.abs(fd_R/d['m_R']-1)):.1e}, m_M {np.max(np.abs(fd_M/d['m_M']-1)):.1e}, "
        f"m_RR {np.max(np.abs(fd_RR/d['m_RR']-1)):.1e}, m_RM {np.max(np.abs(fd_RM/d['m_RM']-1)):.1e}; m(1,R) vs branch-A backbone: {np.max(np.abs(d['m']/m_fam-1)):.1e}")
    say(f"  static horizons at M=1: {[(round(R,5), round(k,6)) for R, k in static_horizons(scen, 1.0)]}")
    ellA = scen.ell
    models = [("hayward_ellA", hayward_mass(ellA)), ("hayward_ell0.05", hayward_mass(0.05)), ("scenario", scen), ("local_lamA", local_mass(ellA))]
    cases = {"hayward_ellA": [0.0, 0.01, 0.1, 0.3, -0.01, -0.1, -0.3], "hayward_ell0.05": [0.0, 0.1, -0.1], "scenario": [0.0, 0.1, -0.1], "local_lamA": [0.0, 0.1]}
    for name, mm in models:
        say(f"\n=== model {name}: {mm.ell_label}; static horizons of the exterior region (R, kappa): {[(round(R,5), round(k,5)) for R, k in static_horizons(mm, 1.0)]}")
        for eps in cases[name]:
            label = f"{name}_eps{eps}"
            r_, data = run_case(mm, eps, label=label)
            results.append(r_)
            plot_case(data, r_, mm, label)
            b, fc = r_["bounce"], r_["first_crossing"]
            say(f"-- eps={eps:+.2f}: bounce for {b['n_bounced']}/{b['n_shells']} shells, tau_b from {b['tau_min']} to {b['tau_max']} (classical {r_['tau_c_classical']:.2f}); crushed: {r_['n_crushed']}")
            say(f"   R_b/r by shell (every 20th): {[f'{v:.3g}' for v in b['R_b_over_r']]}; K at bounce: {[f'{v:.4g}' for v in b['K_b_sample']]}  (24/ell^4 = {24/ellA**4 if 'ellA' in name or name=='scenario' else 24/0.05**4:.4g})")
            say(f"   K_max in the regular region = {r_['K_max_regular']:.4g} at tau={r_['K_max_regular_at']['tau']:.2f}, r={r_['K_max_regular_at']['r']:.3f}; min(rho+p_perp)={r_['min_nec_perp_regular']:.3g}, min rho={r_['min_rho_regular']:.3g}, max p_perp={r_['max_p_perp_regular']:.3g}")
            say(f"   fraction of the vacuum-like component in the ball mass at surface bounce: {r_['vacuum_fraction']['frac_at_bounce']}")
            if fc:
                say(f"   SHELL CROSSING: first at tau={fc['tau']:.3f} (r={fc['r']:.3f}, R={fc['R']:.4g}, R/ell={fc['R']/ellA:.3g}), after this shell's bounce: {fc['after_bounce']} "
                    f"(after {fc['tau_after_bounce']}), inside the trapped region: {fc['inside_trapped']}; K at |Y|=0.1: {fc['K_at_Y']['0.1']}, at 0.01: {fc['K_at_Y']['0.01']}; shells crossed by the end: {fc['n_shells_crossing_by_end']}")
            else:
                say("   no shell crossings over the whole integration time")
            if "exterior_horizon_check" in r_:
                c = r_["exterior_horizon_check"]
                say(f"   exterior-region horizons (event/static): " + "; ".join(f"R {x['R_event']:.5f}/{x['R_static']:.5f}, kappa {x['kappa_event']:+.5f}/{x['kappa_static']:+.5f}" for x in c[:4]))
            say(f"   check of K in the exterior region against the static formula: max relative difference {r_.get('exterior_K_check_max_rel')}; energy integral: {r_['energy_residual_max']:.1e}; time {r_['runtime_s']:.0f} s")
            hzi = [h for h in r_["horizon_events"] if not h["exterior"]]
            if hzi:
                tr = [h for h in hzi if h["V"] < 0]
                say(f"   horizons inside the ball: {len(hzi)} events; during contraction R from {min(h['R'] for h in tr):.4g} to {max(h['R'] for h in tr):.4g}, kappa from {min(h['kappa'] for h in tr):+.4g} to {max(h['kappa'] for h in tr):+.4g}")
    (OUT / "ltb_bounce.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
