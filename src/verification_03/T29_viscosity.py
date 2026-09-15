"""T29. Bulk viscosity in the T-region (KS minisuperspace T26) as a possible attractor of a degenerate exit.
KS equations (T26): (C) 2hḃ/b + ḃ²/b² + 1/b² = 8πρ; (X) 2b̈/b + ḃ²/b² + 1/b² = -8πp_x; (Θ) ḣ + h² + b̈/b + hḃ/b = -8πp_⊥;
(S) ρ̇ + h(ρ + p_x) + 2(ḃ/b)(ρ + p_⊥) = 0;  h = ȧ/a,  Θ = h + 2ḃ/b — expansion of the congruence (in the T-region ḃ < 0: the sphere contracts).
Viscosity: p_eff = p − ζ(ρ)Θ, ζ = ζ₀ρⁿ.
Variant I (within the class T^t_t = T^r_r): viscosity only in p_⊥; p_x = −ρ ⇒ (C)=(X) ⇒ h = b̈/ḃ, a ∝ |ḃ| = √(−f): the metric stays of the form g_tt g_rr = −1,
    system (b, ḃ, ρ):  b̈ = 4πbρ − (ḃ² + 1)/(2b),   ρ̇ = −(2ḃ/b)ρ(1 + w(ρ)) + (2ζ/b)(b̈ + 2ḃ²/b)   [regular as ḃ → 0].
Variant II (the class is broken): p_x = −ρ − ζΘ, p_⊥ = wρ − ζΘ; system (h, b, ḃ, ρ) with h ∝ 1/ḃ at the horizon — the ζΘ term is singular (Eckart);
    causal Israel-Stewart version: τ_IS Π̇ + Π = −ζΘ, p_eff = p + Π.
Medium: EOS p_⊥ = w(ρ)ρ reproducing the base profile (σ = 2(1 + w)); start from R₊ with family data m → (1 + ε)m.
Criterion for a degenerate exit: at the moment ḃ = 0 (or at τ_max) |b̈| < 1e-3 (κ_exit = −b̈ → 0).
Run: python src/verification_03/T29_viscosity.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "polar_qnm"))
sys.path.insert(0, str(HERE.parents[0] / "stability"))
import qnm_band as qb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)


class Medium:
    """base-profile EOS w(ρ): w = p_⊥/ρ = σ/2 − 1, tabulated over ln ρ (ρ is monotonic in b)."""

    def __init__(self, bg):
        self.bg = bg
        r = np.geomspace(1e-3, 6.0, 40001)
        d = bg.fields(r)
        rho = d["mp"] / (4 * np.pi * r**2)
        m = d["m"]
        self.r, self.rho_b, self.m_b = r, rho, m
        sig = d["sigma"]
        w = sig / 2 - 1
        k = np.argsort(rho)
        self.lnrho, self.w_tab = np.log(rho[k]), w[k]
        self.rho_c = float(rho[0])
        self.w_min, self.w_max = float(w.min()), float(w.max())

    def w(self, rho):
        lr = np.log(np.maximum(rho, 1e-300))
        return np.interp(lr, self.lnrho, self.w_tab, left=self.w_max, right=-1.0)   # ρ > ρ_c: plateau (w = −1)

    def rho_of_b(self, b):
        return np.interp(b, self.r, self.rho_b)

    def m_of_b(self, b):
        return np.interp(b, self.r, self.m_b)


def zeta_fn(z0, n):
    return lambda rho: z0 * rho**n


# ---------------------------------------------------------------- variant I
def run_I(med, zeta, eps, tau_max=400.0, IS_tau=None, b_stop=0.05):
    P = med
    roots = np.array(P.r)[np.nonzero(np.diff(np.sign(1 - 2 * (1 + eps) * P.m_b / P.r)) != 0)[0]]
    Rp = roots[-1] if roots.size else 1.985
    b0 = Rp * (1 - 1e-5)
    m0 = (1 + eps) * P.m_of_b(b0)
    rho0 = (1 + eps) * P.rho_of_b(b0)
    v0 = -np.sqrt(max(2 * m0 / b0 - 1, 1e-14))

    def rhs(t, y):
        b, v, rho = y[0], y[1], y[2]
        bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
        w = P.w(rho)
        if IS_tau is None:
            z = zeta(rho)
            rhod = -(2 * v / b) * rho * (1 + w) + (2 * z / b) * (bdd + 2 * v**2 / b)
            return [v, bdd, rhod]
        Pi = y[3]
        rhod = -(2 * v / b) * (rho * (1 + w) + Pi)
        Theta = (bdd / v if abs(v) > 1e-12 else 0.0) + 2 * v / b
        Pid = -(Pi + zeta(rho) * Theta) / IS_tau
        return [v, bdd, rhod, Pid]

    ev_cross = lambda t, y: y[1]; ev_cross.terminal = True; ev_cross.direction = 1
    ev_coll = lambda t, y: y[0] - b_stop; ev_coll.terminal = True
    y0 = [b0, v0, rho0] + ([0.0] if IS_tau is not None else [])
    s = solve_ivp(rhs, (0, tau_max), y0, rtol=1e-10, atol=1e-13, events=[ev_cross, ev_coll], max_step=0.02, dense_output=True)
    b, v, rho = s.y[0], s.y[1], s.y[2]
    bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
    w = P.w(rho)
    if IS_tau is None:
        z = zeta(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            Theta = np.where(np.abs(v) > 1e-14, bdd / v, np.nan) + 2 * v / b
        p_perp = w * rho - z * Theta
    else:
        p_perp = w * rho + s.y[3]
    nec = (rho + p_perp) / rho
    fate = "cross" if s.t_events[0].size else ("collapse" if s.t_events[1].size else "asymptotic")
    kappa_exit = -bdd[-1]
    return dict(fate=fate, tau=float(s.t[-1]), b_end=float(b[-1]), v_end=float(v[-1]), bdd_end=float(bdd[-1]), kappa_exit=float(kappa_exit),
                q_end=float(8 * np.pi * b[-1] ** 2 * rho[-1] - 1), nec_min=float(np.nanmin(nec[np.isfinite(nec)])),
                nec_viol_frac=float(np.mean(nec[np.isfinite(nec)] < 0)), rho_end=float(rho[-1]), w_end=float(w[-1]),
                traj=(b, v))


# ---------------------------------------------------------------- variant II (Eckart and IS)
def run_II(med, zeta, eps, tau_max=400.0, IS_tau=None):
    P = med
    roots = np.array(P.r)[np.nonzero(np.diff(np.sign(1 - 2 * (1 + eps) * P.m_b / P.r)) != 0)[0]]
    Rp = roots[-1] if roots.size else 1.985
    b0 = Rp * (1 - 1e-3)
    m0 = (1 + eps) * P.m_of_b(b0)
    rho0 = (1 + eps) * P.rho_of_b(b0)
    v0 = -np.sqrt(max(2 * m0 / b0 - 1, 1e-14))
    h0 = (8 * np.pi * rho0 - v0**2 / b0**2 - 1 / b0**2) * b0 / (2 * v0)

    def rhs(t, y):
        h, b, v, rho = y[0], y[1], y[2], y[3]
        w = P.w(rho)
        Theta = h + 2 * v / b
        if IS_tau is None:
            Pi = -zeta(rho) * Theta
            px, pp = -rho + Pi, w * rho + Pi
        else:
            Pi = y[4]
            px, pp = -rho + Pi, w * rho + Pi
        bdd = (b / 2) * (-8 * np.pi * px - v**2 / b**2 - 1 / b**2)
        hd = -8 * np.pi * pp - bdd / b - h * v / b - h**2
        rhod = -h * (rho + px) - 2 * (v / b) * (rho + pp)
        out = [hd, v, bdd, rhod]
        if IS_tau is not None:
            out.append(-(Pi + zeta(rho) * Theta) / IS_tau)
        return out

    ev_cross = lambda t, y: y[2] + 1e-5; ev_cross.terminal = True; ev_cross.direction = 1
    ev_coll = lambda t, y: y[1] - 0.05; ev_coll.terminal = True
    ev_blow = lambda t, y: 1e6 - abs(y[0]); ev_blow.terminal = True
    y0 = [h0, b0, v0, rho0] + ([0.0] if IS_tau is not None else [])
    s = solve_ivp(rhs, (0, tau_max), y0, rtol=1e-9, atol=1e-12, events=[ev_cross, ev_coll, ev_blow], max_step=0.02)
    h, b, v, rho = s.y[0], s.y[1], s.y[2], s.y[3]
    C = 2 * h * v / b + v**2 / b**2 + 1 / b**2 - 8 * np.pi * rho
    Theta = h + 2 * v / b
    Pi = -zeta(rho) * Theta if IS_tau is None else s.y[4]
    px = -rho + Pi
    bdd = (b / 2) * (-8 * np.pi * px - v**2 / b**2 - 1 / b**2)
    fate = "cross" if s.t_events[0].size else ("collapse" if s.t_events[1].size else ("blowup" if s.t_events[2].size else "asymptotic"))
    return dict(fate=fate, tau=float(s.t[-1]), b_end=float(b[-1]), v_end=float(v[-1]), bdd_end=float(bdd[-1]), Pi_end=float(Pi[-1]),
                C_max=float(np.max(np.abs(C))), nec_x_min=float(np.min(rho + px)), status=s.status)


def main():
    bg = qb.FamilyBG(qb.build_family(BASE0, 1.0))
    med = Medium(bg)
    say("T29. Bulk viscosity in the T-region: KS dynamics with p_eff = p − ζ(ρ)Θ, base-profile EOS p_⊥ = w(ρ)ρ")
    say(f"  EOS: w from {med.w_min:+.3f} (core) to {med.w_max:+.2f} (cutoff); ρ_c = {med.rho_c:.4f}; in the T-region ḃ < 0 (the sphere contracts), Θ = h + 2ḃ/b, h = ȧ/a = b̈/ḃ (variant I)")
    # sign of Θ along the inviscid trajectory
    r0 = run_I(med, zeta_fn(0.0, 0), 0.0, tau_max=60.0)
    b, v = r0["traj"]
    bdd = 4 * np.pi * b * med.rho_of_b(b) - (v**2 + 1) / (2 * b)
    Th = bdd / v + 2 * v / b
    say(f"  ζ = 0, ε = 0: Θ near R₊ = {Th[5]:+.2f} (h → +∞: a grows from zero), Θ midway (b = {b[len(b)//2]:.3f}) = {Th[len(b)//2]:+.2f}, near the end (b = {b[-1]:.4f}) = {Th[-1]:+.2e}: "
        f"Θ → −∞ as ḃ → 0⁻ with b̈ > 0 (simple exit) — the Eckart ζΘ term diverges at any non-degenerate horizon; in variant I the −ζΘ term enters ρ̇ through (2ζ/b)(b̈ + 2ḃ²/b) — finite.")
    # linearization analytics near the exit: (v = −ḃ, q = 8πb²ρ − 1): v̇ = −q/(2b), q̇ = 16πbρ w v + 8πζ q  =>  s² − 8πζ s + 8πρ w = 0
    say("\n  Linearization near the exit (v = −ḃ → 0, q = 8πb²ρ − 1 → 0): v̇ = −q/(2b), q̇ = 16πbρw·v + 8πζq ⇒ s² − 8πζs + 8πρw = 0:"
        " sink (Re s < 0) ⇔ ζ < 0 and w > 0; node (degenerate exit without crossing) when |ζ| ≥ √(ρw/(2π)); for ζ > 0 — source/saddle: physical viscosity repels from degeneracy.")
    eps_list = [0.0, 1e-4, -1e-4, 1e-2, -1e-2, 1e-1, -1e-1]
    rows = []
    portraits = {}
    say("\n(I) Variant I (within the class): scan over ζ₀, n; fraction of initial ε with a degenerate exit (|b̈| < 1e-3 at ḃ = 0 or asymptotic)")
    say("  n | ζ₀ | fates vs ε (0, +1e-4, −1e-4, +1e-2, −1e-2, +1e-1, −1e-1) | κ_exit | min NEC_⊥ | degenerate fraction")
    for n in (0.0, 0.5, 1.0):
        for z0 in (0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, -1e-3, -1e-2, -3e-2, -1e-1, -3e-1, -1.0, -3.0):
            res = [run_I(med, zeta_fn(z0, n), e) for e in eps_list]
            degen = [(abs(r["bdd_end"]) < 1e-3) and r["fate"] != "collapse" for r in res]
            frac = np.mean(degen)
            fates = "".join({"cross": "x", "collapse": "c", "asymptotic": "a"}[r["fate"]] for r in res)
            kap = " ".join(f"{r['kappa_exit']:+.1e}" for r in res)
            necm = min(r["nec_min"] for r in res)
            rows.append(dict(variant="I", n=n, zeta0=z0, fates=fates, kappa_exit=[r["kappa_exit"] for r in res], bdd_end=[r["bdd_end"] for r in res],
                             q_end=[r["q_end"] for r in res], nec_min=[r["nec_min"] for r in res], degenerate_frac=float(frac), tau=[r["tau"] for r in res], b_end=[r["b_end"] for r in res]))
            say(f"  {n:3.1f} | {z0:+6.0e} | {fates} | {kap} | {necm:+.2e} | {frac:.2f}")
            if z0 in (0.0, 3e-1, -3e-1, -1.0):
                portraits[(n, z0)] = [(e, r["traj"]) for e, r in zip(eps_list, res)]
    # T1 test for the best negative ζ: κ_exit(ε) on a fine ε grid
    say("\n  T1 test (variant I, n = 1): κ_exit as a function of ε at ζ₀ = 0, −0.3, −1, −3, +0.3")
    t1 = {}
    eps_fine = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, -1e-5, -1e-4, -1e-3, -1e-2, -3e-2, -1e-1]
    for z0 in (0.0, -0.3, -1.0, -3.0, 0.3):
        rr = [run_I(med, zeta_fn(z0, 1.0), e) for e in eps_fine]
        t1[z0] = dict(eps=eps_fine, kappa=[r["kappa_exit"] for r in rr], fate=[r["fate"] for r in rr], b_end=[r["b_end"] for r in rr], q_end=[r["q_end"] for r in rr], nec=[r["nec_min"] for r in rr])
        say(f"  ζ₀ = {z0:+.1f}: " + "; ".join(f"ε={e:+.0e}: {r['fate'][0]} κ={r['kappa_exit']:+.1e} b={r['b_end']:.3f} q={r['q_end']:+.1e}" for e, r in zip(eps_fine, rr)))
    # Israel-Stewart, variant I, n = 1
    say("\n  Israel-Stewart (variant I, n = 1, τ_IS = 0.1 and 1.0): ζ₀ = ±0.3, ±1")
    is_rows = []
    for tau_is in (0.1, 1.0):
        for z0 in (0.3, 1.0, -0.3, -1.0):
            res = [run_I(med, zeta_fn(z0, 1.0), e, IS_tau=tau_is) for e in eps_list]
            fates = "".join({"cross": "x", "collapse": "c", "asymptotic": "a"}[r["fate"]] for r in res)
            kap = " ".join(f"{r['kappa_exit']:+.1e}" for r in res)
            degen = np.mean([(abs(r["bdd_end"]) < 1e-3) and r["fate"] != "collapse" for r in res])
            is_rows.append(dict(variant="I-IS", tau_IS=tau_is, zeta0=z0, fates=fates, kappa_exit=[r["kappa_exit"] for r in res], nec_min=[r["nec_min"] for r in res], degenerate_frac=float(degen)))
            say(f"  τ_IS = {tau_is}, ζ₀ = {z0:+.1f}: {fates} | κ_exit {kap} | min NEC_⊥ {min(r['nec_min'] for r in res):+.2e} | degenerate fraction {degen:.2f}")
    # variant II
    say("\n(II) Variant II (viscosity in both pressures; class g_tt g_rr = −1 broken): Eckart and Israel-Stewart, n = 1, ε = 0 and ±1e-2")
    ii_rows = []
    for tag, tau_is in (("Eckart", None), ("IS τ=0.1", 0.1), ("IS τ=1", 1.0)):
        for z0 in (1e-2, 1e-1, 1.0, -1e-2, -1e-1, -1.0):
            res = [run_II(med, zeta_fn(z0, 1.0), e, IS_tau=tau_is) for e in (0.0, 1e-2, -1e-2)]
            fates = "".join({"cross": "x", "collapse": "c", "asymptotic": "a", "blowup": "B"}[r["fate"]] for r in res)
            ii_rows.append(dict(variant="II-" + tag, zeta0=z0, fates=fates, bdd_end=[r["bdd_end"] for r in res], Pi_end=[r["Pi_end"] for r in res], C_max=[r["C_max"] for r in res], nec_x_min=[r["nec_x_min"] for r in res], b_end=[r["b_end"] for r in res]))
            say(f"  {tag:9s} ζ₀ = {z0:+.0e}: fates {fates}; b̈ at exit {[f'{r['bdd_end']:+.1e}' for r in res]}; Π at exit {[f'{r['Pi_end']:+.1e}' for r in res]}; |C|max {max(r['C_max'] for r in res):.1e}; min(ρ+p_x) {min(r['nec_x_min'] for r in res):+.1e}; b_end {[f'{r['b_end']:.3f}' for r in res]}")
    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n in (0.0, 0.5, 1.0):
            fig, axs = plt.subplots(1, 4, figsize=(17, 4))
            for ax, z0 in zip(axs, (0.0, 3e-1, -3e-1, -1.0)):
                for e, (b, v) in portraits[(n, z0)]:
                    ax.plot(b, v, lw=2 if e == 0 else 1, label=f"ε={e:+.0e}")
                ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("b"); ax.set_ylabel("ḃ"); ax.set_title(f"n = {n}, ζ₀ = {z0:+.1f}"); ax.set_xlim(0.3, 2.05)
            axs[0].legend(fontsize=6)
            fig.suptitle("T29: T-region phase portraits with viscosity (variant I)")
            fig.tight_layout(); fig.savefig(OUT / f"T29_phase_n{str(n).replace('.', '')}.png", dpi=120); plt.close(fig)
        say(f"\n  plots: results/T29_phase_n{{0,05,1}}.png")
    except Exception as e:  # noqa: BLE001
        say(f"  plot not built: {e}")
    json.dump(dict(scan_I=rows, t1=t1, IS=is_rows, II=ii_rows), open(OUT / "T29_viscosity.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T29_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
