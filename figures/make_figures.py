"""Figures for the paper: inner-extremal regular black holes with NEC-satisfying anisotropic sources.
All panels are rebuilt from the project results (data/*/*.json) or recomputed with the project code
(imports from src/...).
Run from the repository root:  python figures/make_figures.py  [fig numbers]
Output: figures/fig1_family.pdf ... fig7_residual_kappa.pdf (+ .png previews) and figures_log.txt.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
EXP = ROOT / "src"
DATA = ROOT / "data"
for sub in ("polar_qnm", "stability", "verification_01", "verification_02", "verification_03"):
    sys.path.insert(0, str(EXP / sub))
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a.isdigit()]
WANT = {int(a) for a in sys.argv[1:]} or set(range(1, 8))
_argv_backup = sys.argv
sys.argv = [sys.argv[0], "0.05"]          # qnm_band/polar_qnm read argv[1] as dr*
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import qnm_band as qb  # noqa: E402
import triple_root_monotone as trm  # noqa: E402

sys.argv = _argv_backup
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "lines.linewidth": 1.2, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
R1 = DATA / "verification_01"
R2 = DATA / "verification_02"
R3 = DATA / "verification_03"
RQ = DATA / "polar_qnm"
LOGS = ROOT / "logs"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=200)
    plt.close(fig)
    say(f"  -> {name}.pdf")


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def build_family_ext(base, smin, s1_hi=30.0):
    """As qnm_band.build_family, but with the extended s1 range of T15 (needed near the extremal end of the family)."""
    fam = qb.build_family(base, smin)
    if fam is not None:
        return fam
    import T15_lambda_bound as t15
    s1, u_s, kw, reason = t15.find_merge(base, smin, s1_hi=s1_hi)
    if s1 is None:
        raise RuntimeError(reason)
    d = s1 - smin
    sig, s, mI, H, Hp = trm.profile(d, **kw)
    scale = 1.0 / float(np.interp(u_s, trm.U, H))
    return trm.MonotoneFamily(d, scale, kw=kw)


def profile_data(base, smin):
    fam = build_family_ext(base, smin)
    lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
    R = np.geomspace(0.03, 4.0, 4000)
    rho = fam.rho(R) / fam.rho_c
    sig = np.interp(np.log(R / fam.R1), trm.LU, fam.sig)
    m_of = lambda r: np.interp(np.asarray(r, float), fam._Rg, fam._mg)
    f = 1 - 2 * m_of(R) / R
    rg = np.geomspace(0.05, 4.0, 400001)
    fg = 1 - 2 * m_of(rg) / rg
    idx = np.nonzero(np.sign(fg[:-1]) * np.sign(fg[1:]) < 0)[0]
    roots = [brentq(lambda x: 1 - 2 * m_of(x) / x, rg[i], rg[i + 1], xtol=1e-13) for i in idx]
    if len(roots) < 2:                      # near-extremal: f touches zero without sign change on the grid
        j = int(np.argmin(fg))
        roots = [rg[j], rg[j]] if not roots else [roots[0], rg[j]]
    return dict(fam=fam, lam=lam, R=R, rho=rho, sig=sig, f=f, Rm=roots[0], Rp=roots[-1], roots=roots)


# ------------------------------------------------------------------ fig 1
def fig1():
    say("fig1: family profiles")
    t15 = jload(R1 / "T15_qnm.json")
    cases = [("u3=20", dict(BASE0, u3=20.0), 1.0), ("base", BASE0, 1.0), ("limit", t15["base_opt"], t15["sigma_min_opt"])]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    colors = ["C0", "C3", "C2"]
    for (tag, base, smin), c in zip(cases, colors):
        d = profile_data(base, smin)
        lab = rf"$\lambda = {d['lam']:.3f}$"
        ax[0].plot(d["R"], d["rho"], color=c, label=lab)
        ax[1].plot(d["R"], d["sig"], color=c, label=lab)
        ax[2].plot(d["R"], d["f"], color=c, label=lab)
        for r in (d["Rm"], d["Rp"]):
            ax[2].plot([r], [0], marker="o", ms=3, color=c)
        say(f"  {tag}: lambda = {d['lam']:.4f}, R- = {d['Rm']:.4f}, R+ = {d['Rp']:.4f}")
    ax[0].set(xscale="log", yscale="log", xlabel=r"$R/M$", ylabel=r"$\rho/\rho_c$", ylim=(1e-9, 2), xlim=(0.03, 4))
    ax[1].set(xscale="log", xlabel=r"$R/M$", ylabel=r"$\sigma = -\,d\ln\rho/d\ln R$", xlim=(0.03, 4))
    ax[1].axhline(2, color="k", lw=0.6, ls=":")
    ax[2].set(xscale="log", xlabel=r"$R/M$", ylabel=r"$f(R) = 1 - 2m(R)/R$", xlim=(0.03, 4), ylim=(-0.5, 1.05))
    ax[2].axhline(0, color="k", lw=0.6, ls=":")
    ax[2].legend(loc="lower left", frameon=False)
    for a, t in zip(ax, "abc"):
        a.text(0.02, 0.95, f"({t})", transform=a.transAxes, va="top")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig1_family")


# ------------------------------------------------------------------ fig 2
def fig2():
    say("fig2: kappa_-(eps)")
    d = jload(R1 / "T1_kappa_law.json")
    want = {"base λ=0.271": (r"$\lambda = 0.271$ (base)", "C3"), "u3=20 λ=0.13": (r"$\lambda = 0.13$", "C0"), "s_end=48 λ=0.30": (r"$\lambda = 0.30$", "C2")}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for res in d["results"]:
        if res["tag"] not in want:
            continue
        lab, c = want[res["tag"]]
        k = res["kinds"]["a"]
        rows = k["rows"]
        eps = np.array([r["eps"] for r in rows]); kap = np.array([r["kappa"] for r in rows])
        pos, neg = eps > 0, eps < 0
        ax.loglog(np.abs(eps[pos]), np.abs(kap[pos]), "o", ms=3.5, color=c, label=lab + r", $\epsilon>0$")
        ax.loglog(np.abs(eps[neg]), np.abs(kap[neg]), "s", ms=3.5, mfc="none", color=c, label=lab + r", $\epsilon<0$")
        e = np.geomspace(1e-8, 1e-2, 50)
        ax.loglog(e, k["C_pred"] * e ** (2 / 3), "-", lw=0.8, color=c)
        say(f"  {res['tag']}: C_pred = {k['C_pred']:.3f}, fits p(+) = {k['fits']['1']['p']:.3f}, p(-) = {k['fits']['-1']['p']:.3f}")
    e = np.geomspace(1e-8, 1e-2, 5)
    ax.loglog(e, 0.3 * e ** (2 / 3), "k--", lw=0.8, label=r"slope $2/3$")
    ax.set(xlabel=r"$|\epsilon|$ (density amplitude)", ylabel=r"$|\kappa_-|\,M$", xlim=(5e-9, 2e-2))
    ax.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_kappa_eps")


# ------------------------------------------------------------------ fig 3
def fig3():
    say("fig3: KS phase portrait")
    import T26_ks as ks
    bg = qb.FamilyBG(qb.build_family(BASE0, 1.0))
    t26 = jload(R2 / "T26_ks.json")
    fig, (ax, ins) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for eps, c, ls in [(0.0, "k", "-"), (1e-4, "C3", "-"), (-1e-4, "C0", "-"), (1e-2, "C3", "--"), (-1e-2, "C0", "--"), (1e-1, "C3", ":"), (-1e-1, "C0", ":")]:
        P = ks.Profile(bg, eps)
        roots = P.roots()
        Rm, Rp = roots[0], roots[-1]
        b = np.linspace(Rm * (1 + 1e-9), Rp * (1 - 1e-9), 3000)
        f = P.f(b)
        v = -np.sqrt(np.maximum(-f, 0.0))
        lab = "triple root (separatrix)" if eps == 0 else rf"$\epsilon = {eps:+.0e}$".replace("e-0", "e-").replace("e+0", "e+")
        ax.plot(b, v, color=c, ls=ls, lw=1.3 if eps == 0 else 1.0, label=lab)
        say(f"  eps = {eps:+.0e}: R- = {Rm:.5f}, R+ = {Rp:.5f}, roots = {len(roots)}")
    ax.set(xlabel=r"$b = r$  [$M$]", ylabel=r"$\dot b = dr/d\tau$", xlim=(0.55, 2.05))
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=6)
    ax.text(0.02, 0.05, "(a)", transform=ax.transAxes)
    # (b): proper time to reach R- + x along the separatrix
    x = np.array(t26["tau"]["x"]); tau = np.array(t26["tau"]["tau"])
    ins.loglog(x, tau, "ko", ms=4, label="numerical (T26)")
    xx = np.geomspace(x.min(), x.max(), 20)
    ins.loglog(xx, tau[-1] * (xx / x[-1]) ** (-0.5), "k--", lw=0.8, label=r"$\tau\propto x^{-1/2}$")
    ins.set(xlabel=r"$x = b - R_-$  [$M$]", ylabel=r"$\tau(R_+ \to R_- + x)$  [$M$]")
    ins.legend(frameon=False, fontsize=6)
    ins.text(0.02, 0.05, "(b)", transform=ins.transAxes)
    say(f"  tau slope (T26): {t26['tau']['slope']:.3f}")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_ks_phase")


# ------------------------------------------------------------------ fig 4
def fig4():
    say("fig4: T2 iterations")
    d = jload(R1 / "T2_feedback.json")["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sel = [("kappa~+1e-03, N=3", r"$\kappa_-^{(0)}=-10^{-3}$, heavier ($\epsilon>0$)", "C3", "-"),
           ("kappa~-1e-03, N=3", r"$\kappa_-^{(0)}=-10^{-3}$, lighter ($\epsilon<0$)", "C0", "-"),
           ("kappa~+8e-02, N=3", r"$\kappa_-^{(0)}=-0.08$, heavier", "C3", "--"),
           ("kappa~-8e-02, N=3", r"$\kappa_-^{(0)}=-0.08$, lighter", "C0", "--"),
           ("kappa~+1e-03, N=3, P1s", r"$-10^{-3}$, heavier, smoothed over $\ell$", "C3", ":"),
           ("kappa~-1e-03, N=3, P1s", r"$-10^{-3}$, lighter, smoothed over $\ell$", "C0", ":")]
    for key, lab, c, ls in sel:
        h = d[key]["hist"]
        ne = np.array([x["Ne_cum"] for x in h]); kap = np.array([x["kappa"] for x in h])
        ax.semilogy(ne, np.abs(kap), marker="o", ms=3, color=c, ls=ls, label=lab)
        say(f"  {key}: kappa {kap[0]:+.3e} -> {kap[-1]:+.3e} over N_e = {ne[-1]:.0f}")
    ax.set(xlabel=r"accumulated e-folds of mass inflation $N_e$", ylabel=r"$|\kappa_-|\,M$", ylim=(5e-4, 4))
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    save(fig, "fig4_t2_iterations")


# ------------------------------------------------------------------ fig 5
def t30_trajectory(src, rho0_frac, dE_frac=0.0, b0=1.9, tau_max=400.0):
    import T30_phase_transition as t30
    PI = np.pi
    eps_c = src.eps_c
    rho0 = rho0_frac * eps_c
    bd0 = -np.sqrt(2.0 / b0 - 1.0)
    y0 = [t30.h_from_constraint(b0, bd0, rho0), b0, bd0, rho0, 0.0]
    ev_trig = lambda t, y: y[3] - eps_c; ev_trig.terminal = True; ev_trig.direction = 1
    ev_turn = lambda t, y: y[2] + 1e-7; ev_turn.terminal = True; ev_turn.direction = 1
    ev_sing = lambda t, y: y[1] - 1e-3; ev_sing.terminal = True
    s1 = solve_ivp(t30.rhs_factory(src, 0.0, False), (0, tau_max), y0, events=[ev_trig, ev_turn, ev_sing], rtol=1e-10, atol=1e-14, max_step=0.02)
    B, V = [s1.y[1]], [s1.y[2]]
    outcome = "no transition"
    if s1.t_events[0].size:
        y = s1.y[:, -1].copy()
        rho_v = eps_c * (1 + dE_frac)
        y[3] = rho_v; y[0] = t30.h_from_constraint(y[1], y[2], rho_v); y[4] = 1.0
        s2 = solve_ivp(t30.rhs_factory(src, 0.0, True), (s1.t[-1], s1.t[-1] + tau_max), y, events=[ev_turn, ev_sing], rtol=1e-10, atol=1e-14, max_step=0.02)
        B.append(s2.y[1]); V.append(s2.y[2])
        m0 = y[1] * (1 + y[2] ** 2) / 2 - (4 * PI / 3) * rho_v * y[1] ** 3
        if s2.t_events[0].size:
            b, rho = s2.y[1, -1], s2.y[3, -1]
            kappa = (1 - 8 * PI * rho * b**2) / (2 * b)
            outcome = f"turning point b = {b:.3f}, kappa = {kappa:+.3g}"
        elif s2.t_events[1].size:
            outcome = "singularity b -> 0"
        outcome += f" (m0 = {m0:+.3f})"
    return np.concatenate(B), np.concatenate(V), outcome, (s1.y[1, -1] if s1.t_events[0].size else None)


def fig5():
    say("fig5: T29 / T30 portraits")
    import T29_viscosity as t29
    import T30_phase_transition as t30
    bg = qb.FamilyBG(qb.build_family(BASE0, 1.0))
    med = t29.Medium(bg)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    # (a) viscosity
    runs = [(0.0, 0, 0.0, "k", "-", r"$\zeta = 0$ (separatrix)"),
            (0.3, 1, 1e-2, "C1", "-", r"$\zeta = 0.3\rho$, $\epsilon = +10^{-2}$"),
            (0.3, 1, -1e-2, "C1", "--", r"$\zeta = 0.3\rho$, $\epsilon = -10^{-2}$"),
            (1.0, 1, 1e-2, "C4", "-", r"$\zeta = \rho$, $\epsilon = +10^{-2}$"),
            (-0.2, 0, 1e-2, "C2", "-", r"$\zeta = -0.2$, $\epsilon = +10^{-2}$"),
            (-0.2, 0, -1e-2, "C2", "--", r"$\zeta = -0.2$, $\epsilon = -10^{-2}$"),
            (-0.2, 0, 1e-1, "C2", ":", r"$\zeta = -0.2$, $\epsilon = +10^{-1}$")]
    for z0, n, eps, c, ls, lab in runs:
        r = t29.run_I(med, t29.zeta_fn(z0, n), eps)
        b, v = r["traj"]
        ax[0].plot(b, v, color=c, ls=ls, label=lab)
        say(f"  T29 zeta0 = {z0}, n = {n}, eps = {eps:+.0e}: fate = {r['fate']}, b_end = {r['b_end']:.4f}, kappa_exit = {r['kappa_exit']:+.3e}, nec_min = {r['nec_min']:+.3f}")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel=r"$b$  [$M$]", ylabel=r"$\dot b$", xlim=(0.3, 2.05))
    ax[0].legend(frameon=False, fontsize=6, loc="lower left")
    ax[0].text(0.02, 0.95, "(a)", transform=ax[0].transAxes, va="top")
    # (b) first-order transition
    for ell, rho0, dE, c, ls, lab in [(0.27, 1e-2, 0.0, "C0", "-", r"$\ell/M = 0.27$, $\rho_0 = 10^{-2}\epsilon_c$, $\Delta E = 0$"),
                                      (0.27, 0.3, 0.0, "C3", "-", r"$\ell/M = 0.27$, $\rho_0 = 0.3\,\epsilon_c$, $\Delta E = 0$"),
                                      (0.27, 1e-2, -0.4530562409715838, "C2", "--", r"$\ell/M = 0.27$, tuned $\Delta E^* = -0.453\,\epsilon_c$"),
                                      (0.27, 1e-2, -0.45, "C2", ":", r"$\ell/M = 0.27$, $\Delta E = -0.450\,\epsilon_c$"),
                                      (0.10, 1e-4, 0.0, "C1", "-", r"$\ell/M = 0.10$, $\rho_0 = 10^{-4}\epsilon_c$, $\Delta E = 0$")]:
        eps_c = 3 / (8 * np.pi * ell**2)
        src = t30.Source(eps_c, vac="dS")
        b, v, out, bc = t30_trajectory(src, rho0, dE)
        ax[1].plot(b, v, color=c, ls=ls, label=lab)
        if bc is not None:
            ax[1].plot([bc], [np.interp(bc, b[::-1], v[::-1])], marker="x", color=c, ms=4)
        say(f"  T30 ell = {ell}, rho0 = {rho0}, dE = {dE}: {out}; transition at b_c = {bc}")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel=r"$b$  [$M$]", ylabel=r"$\dot b$", xscale="log", xlim=(8e-4, 2.1))
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_ylim(-80, 0.15)
    ax[1].legend(frameon=False, fontsize=6, loc="lower left")
    ax[1].text(0.02, 0.95, "(b)", transform=ax[1].transAxes, va="top")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig5_t29_t30")


# ------------------------------------------------------------------ fig 6
def parse_fd_stdout():
    pts = []
    for name in ("axial_fd_fix_stdout.txt", "axial_fd_fix2_stdout.txt"):
        path = LOGS / "polar_qnm" / name
        if not path.exists():      # a single combined run covers every tag
            continue
        for line in open(path, encoding="utf-8"):
            m = re.match(r"(.+?) l=2: frequency domain dRe/Re = ([-+0-9.e]+), dIm/Im = ([-+0-9.e]+)", line)
            if m:
                pts.append((m.group(1).strip(), float(m.group(2)), float(m.group(3))))
    return pts


def fig6():
    say("fig6: halo -> QNM shift")
    band = jload(RQ / "qnm_band.json")["rows"]
    t18 = jload(R1 / "T18_halo.json")
    fd = parse_fd_stdout()
    tagmap = {"base (scenario)": "base (scenario)", "u2=3": "u2 = 3", "u3=12": "u3 = 12", "u3=20": "u3 = 20", "s_end=8": "s_end = 8",
              "s_end=48": "s_end = 48", "sigma_min=0.5": "sigma_min = 0.5", "w2=0.6": "w2 = 0.6"}
    halo_by_tag = {r["tag"]: 1 - r["m_plus"] for r in band}
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    def put(h, dre, dim, kind):
        if dre is None or dim is None:
            return
        h = max(h, 1e-6)
        style = dict(band=dict(marker="o", color="C3"), near=dict(marker="s", color="C0"), t15=dict(marker="^", color="C2"))[kind]
        ax.loglog([h], [abs(dre)], ls="none", ms=4.5, mfc=style["color"] if dre <= 0 else "none", **style)
        ax.loglog([h], [abs(dim)], ls="none", marker="x", ms=4, color=style["color"], alpha=0.7)
    for r in band:
        put(1 - r["m_plus"], r["ax_l2_dRe"], r["ax_l2_dIm"], "band")
    for r in t18["C"]:
        put(r["halo"], r["qnm_dRe"], r["qnm_dIm"], "near")
    for r in t18["D"]:
        if r["tag"] != "base":
            put(r["halo"], r["dRe"], r["dIm"], "t15")
    for tag, dre, dim in fd:
        h = halo_by_tag[tagmap[tag]]
        ax.loglog([h], [abs(dre)], ls="none", marker="D", ms=6, mfc="none", color="k", lw=0.8)
    for r in t18["A"]:
        if "qnm_dRe" in r:
            put(1e-6, r["qnm_dRe"], r.get("qnm_dIm", r["qnm_dRe"]), "near")
            say(f"  compact cut Rc = {r['Rc']}: dRe = {r['qnm_dRe']:+.1e}")
    hh = np.geomspace(1e-5, 0.3, 50)
    ax.loglog(hh, 0.2 * hh, "k--", lw=0.9, label=r"$|\delta\omega_R/\omega_R| = 0.2\,m_{\rm halo}/M$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="C3", ls="none", label="family scan (T = time domain)"),
               Line2D([], [], marker="s", color="C0", ls="none", label="near-compact profiles (T18)"),
               Line2D([], [], marker="^", color="C2", ls="none", label=r"extended family, $\lambda = 0.44$–$0.84$ (T15)"),
               Line2D([], [], marker="D", color="k", mfc="none", ls="none", label="frequency-domain check"),
               Line2D([], [], marker="x", color="gray", ls="none", label=r"$|\delta\omega_I/\omega_I|$"),
               Line2D([], [], ls="--", color="k", label=r"$0.2\,m_{\rm halo}/M$")]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper left")
    ax.set(xlabel=r"mass outside $R_+$, $m_{\rm halo}/M$", ylabel=r"$|\delta\omega/\omega|$ ($l = 2$, axial)", xlim=(5e-6, 0.4), ylim=(1e-6, 0.2))
    ax.text(0.98, 0.03, "open symbols: positive frequency shift", transform=ax.transAxes, ha="right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig6_halo_qnm")


# ------------------------------------------------------------------ fig 7
def fig7():
    say("fig7: residual kappa_- for ell^2 = lambda^2 M^2 + ell0^2")
    import T1_kappa_law as t1
    prof = t1.Profile(BASE0, 1.0, "base")
    lam = prof.ell
    R = np.geomspace(1e-3, 3.0, 400001)
    f0, _ = prof.f_eps(R, "a", 0.0)
    idx = np.nonzero(np.sign(f0[:-1]) * np.sign(f0[1:]) < 0)[0]
    Rm, Rp = R[idx[0]], R[idx[-1]]
    l0 = np.geomspace(1e-4, 0.3, 25)
    kap = []
    for x in l0:
        eps = np.sqrt(1 + (x / lam) ** 2) - 1
        rk = t1.roots_and_kappa(prof, "b", eps, Rm, Rp)
        kap.append(rk[0][1] if rk else np.nan)
    kap = np.array(kap)
    k = (l0 <= 1e-2) & np.isfinite(kap)
    p, c = np.polyfit(np.log(l0[k]), np.log(np.abs(kap[k])), 1)
    C = float(np.exp(c))
    say(f"  slope d ln|kappa|/d ln(ell0/M) for ell0/M <= 0.01 (eps_eff <= 7e-4): {p:.4f} (expected 4/3); coefficient |kappa_-| ≈ {C:.3f} (ell0/M)^{p:.3f}")
    k2 = (l0 <= 3e-2) & np.isfinite(kap)
    p2 = np.polyfit(np.log(l0[k2]), np.log(np.abs(kap[k2])), 1)[0]
    say(f"  slope for ell0/M <= 0.03: {p2:.4f} (coefficient drifts at eps ~ 1e-2, as in T1)")
    # analytic: kappa = C_b eps^{2/3}, eps ≈ (l0/lam)^2/2  =>  C = C_b (1/(2 lam^2))^{2/3}
    Cb = jload(R1 / "T1_kappa_law.json")["results"][0]["kinds"]["b"]["C_pred"]
    Can = Cb * (0.5 / lam**2) ** (2 / 3)
    say(f"  analytic coefficient from T1 (kind b, C_pred = {Cb:.3f}): {Can:.3f} (ell0/M)^(4/3)")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(l0, np.abs(kap), "o", ms=3.5, color="C3", label="numerical (base profile, fixed shape and $M$)")
    ax.loglog(l0, Can * l0 ** (4 / 3), "k--", lw=0.9, label=rf"${Can:.2f}\,(\ell_0/M)^{{4/3}}$ (from $\kappa_-\propto\epsilon^{{2/3}}$)")
    ax.set(xlabel=r"$\ell_0/M$", ylabel=r"$|\kappa_-|\,M$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig7_residual_kappa")
    return p, C, Can


def main():
    t0 = time.time()
    out = {}
    for n, fn in [(1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5), (6, fig6), (7, fig7)]:
        if n in WANT:
            t = time.time()
            out[n] = fn()
            say(f"  [{time.time()-t:.0f} s]")
    say(f"total {time.time()-t0:.0f} s")
    (HERE / "figures_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
