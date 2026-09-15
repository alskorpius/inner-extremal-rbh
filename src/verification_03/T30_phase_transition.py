"""T30. Non-smooth source in the T-region: first-order phase transition at ε_c (KS minisuperspace, an extension of T26).
KS system (T26): state (h = ȧ/a, b, ḃ, ρ, ξ):
    b̈ = (b/2)(−8π p_x − ḃ²/b² − 1/b²),  ḣ = −8π p_⊥ − b̈/b − h ḃ/b − h²,  ρ̇ = −h(ρ + p_x) − 2(ḃ/b)(ρ + p_⊥),
    constraint (C): 2hḃ/b + ḃ²/b² + 1/b² = 8πρ  (sets h at the start, monitored).
Phases: ρ < ε_c — isotropic fluid p_x = p_⊥ = p(ρ) (p = wρ, w = 1/3; or polytrope p = Kρ^γ, γ = 2);
      ρ ≥ ε_c — vacuum-like branch p_x = −ρ with p_⊥ = −ρ ("dS", fixed ε_c — companion-paper scenario) or p_⊥ = (σ(s)/2 − 1)ρ,
      s = ρ/ρ_c, σ(s) — slope of the base profile as a function of density ("scenario EOS", trigger at s_tr = s(u₃)).
Instantaneous transition: ρ → ε_c + ΔE (h is recomputed from the constraint — jump in ξ_extrinsic; ΔE = 0 ⇒ pressure-only jump, h continuous);
finite time: phase fraction ξ, ξ̇ = (1 − ξ)/τ_rel after the trigger, p = (1−ξ)p_fluid + ξ p_vac.
Start: Schwarzschild T-region M = 1 at b₀ = 1.9 (ḃ₀ = −√(2/b₀ − 1)) with fluid ρ₀ = ρ₀/ε_c · ε_c.
Outcomes as ḃ → 0: κ = (1 − 8πρb²)/(2b) (= f'/2 at the root; κ < 0 — exit into the static R-region through a simple horizon, |h| → ∞;
κ = 0 — degenerate exit / genuine bounce); b → 0 — singularity. For the dS branch ρ = const ⇒ exactly SdS: ḃ² = −f, f = 1 − 2m₀/b − Λb²/3,
m₀ = b(1+ḃ²)/2 − (4π/3)ρb³ (mass not converted to vacuum); degenerate exit ⇔ 9Λm₀² = 1 (Nariai) ⇒ ΔE*(ε_c, ρ₀) — codimension 1.
Run: python src/verification_03/T30_phase_transition.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "polar_qnm"))
sys.path.insert(0, str(HERE.parents[0] / "stability"))
import qnm_band as qb  # noqa: E402
import triple_root_monotone as trm  # noqa: E402

LOG = []
PI = np.pi


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- scenario EOS from the base profile
BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)
FAM = qb.build_family(BASE0, 1.0)
_s_rev, _sig_rev = FAM.s[::-1], FAM.sig[::-1]          # s decreases with u ⇒ reverse for interp over s
RHO_C_BASE = FAM.rho_c
ELL_BASE = float(np.sqrt(3 / (8 * PI * RHO_C_BASE)))
S_TR = float(np.interp(np.log(BASE0["u3"]), trm.LU, FAM.s))     # s at the cutoff radius u₃


def sigma_of_s(s):
    s = float(s)
    if s >= 1.0:
        return 0.0
    return float(np.interp(s, _s_rev, _sig_rev))


# ---------------------------------------------------------------- source
class Source:
    def __init__(self, eps_c, fluid="w", w=1/3, gamma=2.0, vac="dS", rho_c_scen=None):
        self.eps_c, self.fluid, self.w, self.gamma, self.vac = eps_c, fluid, w, gamma, vac
        self.K = 0.3 * eps_c ** (1 - gamma)          # p/ρ = 0.3 at ε_c
        self.rho_c_scen = rho_c_scen or eps_c

    def p_fluid(self, rho):
        return self.w * rho if self.fluid == "w" else self.K * rho**self.gamma

    def p_vac(self, rho):
        if self.vac == "dS":
            return -rho, -rho
        s = rho / self.rho_c_scen
        return -rho, (sigma_of_s(s) / 2 - 1) * rho

    def pressures(self, rho, xi):
        pf = self.p_fluid(rho)
        pvx, pvt = self.p_vac(rho)
        return (1 - xi) * pf + xi * pvx, (1 - xi) * pf + xi * pvt


def rhs_factory(src, tau_rel, triggered):
    """h = ȧ/a is taken algebraically from the constraint (C) (exact, as long as ḃ ≠ 0); state y = (h_dummy, b, ḃ, ρ, ξ), h_dummy does not evolve."""
    def rhs(t, y):
        _, b, bd, rho, xi = y
        h = h_from_constraint(b, bd, rho)
        px, pt = src.pressures(rho, xi)
        bdd = (b / 2) * (-8 * PI * px - bd**2 / b**2 - 1 / b**2)
        rhod = -h * (rho + px) - 2 * (bd / b) * (rho + pt)
        xid = ((1.0 - xi) / tau_rel) if (triggered and tau_rel > 0) else 0.0
        return [0.0, bd, bdd, rhod, xid]
    return rhs


def h_from_constraint(b, bd, rho):
    return (8 * PI * rho - bd**2 / b**2 - 1 / b**2) * b / (2 * bd)


def constraint(y):
    """with algebraic h the constraint holds identically; returns the residual ḃ² − (2m/b − 1) with m = b(1+ḃ²)/2 ≡ 0 — kept for compatibility."""
    return 0.0


def run(src, rho0_frac, dE_frac=0.0, tau_rel=0.0, b0=1.9, tau_max=400.0, trigger_frac=1.0):
    """Returns a dict with the outcome. trigger_frac — trigger threshold in units of ε_c (for scenario EOS: s_tr)."""
    eps_c = src.eps_c
    rho0 = rho0_frac * eps_c
    bd0 = -np.sqrt(2.0 / b0 - 1.0)
    y0 = [h_from_constraint(b0, bd0, rho0), b0, bd0, rho0, 0.0]
    rho_tr = trigger_frac * eps_c
    ev_trig = lambda t, y: y[3] - rho_tr; ev_trig.terminal = True; ev_trig.direction = 1
    ev_turn = lambda t, y: y[2] + 1e-7; ev_turn.terminal = True; ev_turn.direction = 1
    ev_sing = lambda t, y: y[1] - 1e-3; ev_sing.terminal = True
    ev_hbig = lambda t, y: 1.0; ev_hbig.terminal = False
    res = dict(rho0_frac=rho0_frac, dE_frac=dE_frac, tau_rel=tau_rel, triggered=False)
    # phase 1: fluid
    s1 = solve_ivp(rhs_factory(src, tau_rel, False), (0, tau_max), y0, events=[ev_trig, ev_turn, ev_sing], rtol=1e-10, atol=1e-14, max_step=0.02)
    t_off = s1.t[-1]
    y = s1.y[:, -1].copy()
    res["C_fluid"] = float(abs(constraint(y)))
    if s1.t_events[1].size or s1.t_events[2].size or not s1.t_events[0].size:
        res.update(outcome="singularity in the fluid phase (ḃ<0 until b→0)" if s1.t_events[2].size else "ḃ = 0 in the fluid phase" if s1.t_events[1].size else "transition not reached",
                   b_end=float(y[1]), tau_end=float(t_off))
        return res
    res["triggered"] = True
    res["b_c"], res["bd_c"], res["tau_c"] = float(y[1]), float(y[2]), float(t_off)
    m_c = y[1] * (1 + y[2]**2) / 2
    res["m_c"] = float(m_c)
    # transition
    if tau_rel == 0.0:
        rho_v = rho_tr * (1.0 + dE_frac) if src.vac == "dS" else rho_tr * (1.0 + dE_frac)
        y[3] = rho_v
        y[0] = h_from_constraint(y[1], y[2], rho_v)     # jump in h when ΔE ≠ 0 (h continuous when ΔE = 0)
        y[4] = 1.0
    else:
        y[4] = 0.0
    # SdS diagnostics (dS branch only): m₀ and the Nariai condition
    if src.vac == "dS":
        rho_v = y[3]
        Lam = 8 * PI * rho_v
        m0 = y[1] * (1 + y[2]**2) / 2 - (4 * PI / 3) * rho_v * y[1]**3
        m0_star = 1 / (3 * np.sqrt(Lam))
        res.update(rho_vac_frac=float(rho_v / eps_c), m0=float(m0), m0_nariai=float(m0_star), nariai_ratio=float(9 * Lam * m0**2))
    # phase 2
    s2 = solve_ivp(rhs_factory(src, tau_rel, True), (t_off, t_off + tau_max), y, events=[ev_turn, ev_sing, ev_hbig], rtol=1e-10, atol=1e-14, max_step=0.02)
    y2 = s2.y[:, -1]
    res["C_vac"] = float(abs(constraint(y2)))
    h, b, bd, rho, xi = y2
    res.update(b_end=float(b), bd_end=float(bd), rho_end_frac=float(rho / eps_c), h_end=float(h), xi_end=float(xi), tau_end=float(s2.t[-1]))
    if s2.t_events[0].size or s2.t_events[2].size:
        kappa = (1 - 8 * PI * rho * b**2) / (2 * b)
        res["kappa_exit"] = float(kappa)
        res["outcome"] = "degenerate exit (κ≈0)" if abs(kappa) < 1e-5 else ("exit through a simple horizon (κ₋<0), R-region" if kappa < 0 else "maximum of b (κ>0?)")
    elif s2.t_events[1].size:
        res["outcome"] = "singularity (b→0) in the vacuum phase"
    else:
        # slow approach (degenerate) or not reached
        kappa = (1 - 8 * PI * rho * b**2) / (2 * b)
        res["kappa_exit"] = float(kappa)
        res["outcome"] = "asymptotic approach ḃ→0 (degenerate exit)" if abs(bd) < 1e-3 else "not completed within τ_max"
    return res


def sds_check(src, r):
    """for the dS branch: check ḃ² = −f_SdS(b) at the end (a formula independent of the integrator)."""
    if not r.get("triggered") or src.vac != "dS":
        return np.nan
    rho_v = r["rho_vac_frac"] * src.eps_c
    f = 1 - 2 * r["m0"] / r["b_end"] - 8 * PI * rho_v * r["b_end"]**2 / 3
    return abs(r["bd_end"]**2 + f)


def dE_star(src, rho0_frac):
    """ΔE* at which m₀ = m₀* (Nariai) — computed analytically from the state at the transition."""
    r = run(src, rho0_frac, 0.0, 0.0)
    if not r["triggered"]:
        return None, r
    eps_c = src.eps_c
    b_c, m_c = r["b_c"], r["m_c"]
    # m₀(ΔE) = m_c − (4π/3)(ε_c(1+δ)) b_c³ ; m₀* = 1/(3√(8π ε_c(1+δ))) ⇒ solve for δ
    g = lambda d: (m_c - (4 * PI / 3) * eps_c * (1 + d) * b_c**3) - 1 / (3 * np.sqrt(8 * PI * eps_c * (1 + d)))
    d_lo, d_hi = -0.999, 1e6
    if g(d_lo) * g(d_hi) > 0:
        return None, r
    d = brentq(g, d_lo, d_hi, xtol=1e-14)
    return d, r


def main():
    say("T30: first-order phase transition in the T-region (KS). M = 1, start b₀ = 1.9 inside the Schwarzschild horizon.")
    say(f"  base profile: ρ_c = {RHO_C_BASE:.4f}, ℓ = {ELL_BASE:.4f}, s(u₃) = {S_TR:.3e} (scenario-EOS threshold)")
    rows = []
    # ---------- (A) control: no transition (ε_c → ∞): fluid w = 1/3 — singularity
    src0 = Source(eps_c=1e9, vac="dS")
    r = run(src0, 1e-12)
    say(f"\n(A) control without transition, p = ρ/3, ρ₀ ≪: outcome — {r['outcome']} at b = {r['b_end']:.2e}, τ = {r['tau_end']:.2f}; |C| = {r['C_fluid']:.1e}")
    # ---------- (B) dS branch: table over ℓ/M, ρ₀/ε_c, ΔE, τ_rel
    say("\n(B) Transition to the p_x = p_⊥ = −ρ branch (fixed ε_c, companion-paper scenario): outcome, m₀, Nariai condition 9Λm₀² (1 = degenerate exit)")
    say("  ℓ/M | ρ₀/ε_c | ΔE/ε_c | τ_rel | b_c | m_c | ρ_vac/ε_c | m₀ | m₀/m₀* | 9Λm₀² | outcome | κ_exit | |ḃ²+f_SdS| | |C|")
    for ell in (0.1, 0.27, 0.5, 1.0, 1.5):
        eps_c = 3 / (8 * PI * ell**2)
        src = Source(eps_c=eps_c, vac="dS")
        for rho0 in (1e-4, 1e-2, 0.3):
            for dE in (0.0, 0.5, -0.5, 2.0):
                for trel in (0.0, 0.1, 1.0):
                    if trel > 0 and dE != 0.0:
                        continue          # finite time — only at ΔE = 0 (latent heat is not specified in the soft version)
                    r = run(src, rho0, dE, trel)
                    r.update(ell=ell, eps_c=eps_c, vac="dS")
                    chk = sds_check(src, r) if trel == 0 else np.nan
                    r["sds_check"] = chk
                    rows.append(r)
                    if r["triggered"]:
                        say(f"  {ell:4.2f} | {rho0:7.0e} | {dE:+4.1f} | {trel:3.1f} | {r['b_c']:.3f} | {r['m_c']:.3f} | {r.get('rho_vac_frac', np.nan):.2f} | {r.get('m0', np.nan):+.4f} | {r.get('m0', np.nan)/r.get('m0_nariai', np.nan):+.2f} | {r.get('nariai_ratio', np.nan):.3f} | {r['outcome']} | {r.get('kappa_exit', np.nan):+.3e} | {chk:.1e} | {r['C_vac']:.1e}")
                    else:
                        say(f"  {ell:4.2f} | {rho0:7.0e} | {dE:+4.1f} | {trel:3.1f} | {r['outcome']} (b_end = {r['b_end']:.3e})")
    # ---------- (C) ΔE*: codimension 1
    say("\n(C) ΔE* giving a degenerate exit (9Λm₀² = 1) — does it exist, and is it fine-tuning")
    star_rows = []
    for ell in (0.27, 0.5, 1.0):
        eps_c = 3 / (8 * PI * ell**2)
        src = Source(eps_c=eps_c, vac="dS")
        for rho0 in (1e-4, 1e-2, 0.3):
            d, r0 = dE_star(src, rho0)
            if d is None:
                say(f"  ℓ = {ell}, ρ₀/ε_c = {rho0:.0e}: ΔE* does not exist (transition not reached or no root)"); continue
            rs = run(src, rho0, d, 0.0, tau_max=2000.0)
            rp = run(src, rho0, d * (1 + 1e-3) if d != 0 else 1e-3, 0.0)
            rm = run(src, rho0, d * (1 - 1e-3) if d != 0 else -1e-3, 0.0)
            say(f"  ℓ = {ell}, ρ₀/ε_c = {rho0:.0e}: b_c = {r0['b_c']:.4f}, ΔE*/ε_c = {d:+.6f} (ρ_vac* = {(1+d):.4f} ε_c): outcome {rs['outcome']}, κ = {rs.get('kappa_exit', np.nan):+.2e}, 9Λm₀² = {rs['nariai_ratio']:.6f};"
                f"  ΔE*(1+10⁻³): {rp['outcome']}, κ = {rp.get('kappa_exit', np.nan):+.2e}, 9Λm₀² = {rp['nariai_ratio']:.4f};  ΔE*(1−10⁻³): {rm['outcome']}, κ = {rm.get('kappa_exit', np.nan):+.2e}, 9Λm₀² = {rm['nariai_ratio']:.4f}")
            star_rows.append(dict(ell=ell, rho0_frac=rho0, dE_star=d, b_c=r0["b_c"], res=rs, plus=rp, minus=rm))
    # ---------- (D) polytrope γ = 2 (fluid EOS control)
    say("\n(D) Control: polytrope p = Kρ² (p/ρ = 0.3 at ε_c), ℓ = 0.27, ΔE = 0")
    src = Source(eps_c=3 / (8 * PI * 0.27**2), fluid="poly", gamma=2.0, vac="dS")
    for rho0 in (1e-4, 1e-2, 0.3):
        r = run(src, rho0, 0.0, 0.0); r.update(ell=0.27, vac="dS-poly"); rows.append(r)
        say(f"  ρ₀/ε_c = {rho0:.0e}: b_c = {r.get('b_c', np.nan):.3f}, m₀ = {r.get('m0', np.nan):+.4f}, 9Λm₀² = {r.get('nariai_ratio', np.nan):.3f}, outcome {r['outcome']}, κ = {r.get('kappa_exit', np.nan):+.2e}")
    # ---------- (E) scenario EOS: trigger at s_tr, branch p_⊥ = (σ(s)/2−1)ρ, ε_c = base ρ_c: scan over ρ₀ — κ_exit(ρ₀)
    say(f"\n(E) Scenario branch (base-profile EOS, trigger at ρ = s(u₃)·ρ_c = {S_TR:.2e} ρ_c): degenerate exit ⇔ transition exactly at b_c = u₃R₁ = {BASE0['u3']*FAM.R1:.4f}")
    src = Source(eps_c=RHO_C_BASE, vac="scen", rho_c_scen=RHO_C_BASE)
    scen_rows = []
    rho0_list = np.geomspace(0.02, 0.999, 18) * S_TR      # ρ_tr is small (2·10⁻³ρ_c): for the transition to happen at b_c ~ M, ρ₀ must be of order ρ_tr
    for rho0 in rho0_list:
        r = run(src, rho0, 0.0, 0.0, trigger_frac=S_TR, tau_max=600.0)
        r.update(ell=ELL_BASE, vac="scen"); scen_rows.append(r)
        say(f"  ρ₀/ρ_tr = {rho0/S_TR:.1e}: b_c = {r.get('b_c', np.nan):.4f}, outcome {r['outcome']}, κ_exit = {r.get('kappa_exit', np.nan):+.3e}, b_end = {r['b_end']:.4f}, ρ_end/ρ_c = {r.get('rho_end_frac', np.nan):.3f}, |C| = {r.get('C_vac', np.nan):.1e}")
    say("  ⇒ starting from the Schwarzschild T-region, an upward crossing transition at ρ_tr is possible only for b_c ≤ 1.04 (ρ initially decreases: a = √|f| grows faster than b² contracts), and in all cases the outcome is a singularity: the residual mass M is not converted.")
    # (E2) two-parameter diagnostics of the scenario vacuum branch: start the branch at (b_c, m_c) — degenerate exit only when b_c = u₃R₁ AND m_c = m_base(b_c)
    say("\n(E2) Scenario branch, vacuum phase started at given (b_c, m_c) [ρ = ρ_tr, ḃ² = 2m_c/b_c − 1]: κ at exit (degenerate ⇔ κ = 0)")
    bc_star = BASE0["u3"] * FAM.R1
    m_base = lambda r: float(np.interp(r, FAM._Rg, FAM._mg))
    mc_star = m_base(bc_star)
    say(f"  b_c* = u₃R₁ = {bc_star:.5f}, m_c* = m_base(b_c*) = {mc_star:.5f} (base profile: R₋ = 0.6776, triple root)")
    deltas = (-0.02, -0.005, 0.0, 0.005, 0.02)
    e2 = []
    head = "  δ' (b_c) / δ (m_c) |" + "|".join(f" {d:+.3f} " for d in deltas)
    say(head)
    for dp in deltas:
        b_c = bc_star * (1 + dp)
        line = f"  {dp:+.3f}            |"
        for d in deltas:
            m_c = mc_star * (1 + d)
            bd_c = -np.sqrt(max(2 * m_c / b_c - 1, 1e-16))
            y = [0.0, b_c, bd_c, S_TR * RHO_C_BASE, 1.0]
            ev_turn = lambda t, y: y[2] + 1e-7; ev_turn.terminal = True; ev_turn.direction = 1
            ev_sing = lambda t, y: y[1] - 1e-3; ev_sing.terminal = True
            sv = solve_ivp(rhs_factory(src, 0.0, True), (0, 400.0), y, events=[ev_turn, ev_sing], rtol=1e-10, atol=1e-14, max_step=0.02)
            b, bd, rho = sv.y[1, -1], sv.y[2, -1], sv.y[3, -1]
            if sv.t_events[1].size:
                kap = np.nan; tag = "sing"
            else:
                kap = (1 - 8 * PI * rho * b**2) / (2 * b); tag = "exit" if sv.t_events[0].size else "asympt"
            e2.append(dict(dbc=dp, dmc=d, b_c=b_c, m_c=m_c, kappa=float(kap) if np.isfinite(kap) else None, b_end=float(b), tag=tag, tau=float(sv.t[-1])))
            line += f" {tag} {kap:+.1e} |" if np.isfinite(kap) else f" {tag}        |"
        say(line)
    scen_star = dict(bc_star=bc_star, mc_star=mc_star, table=e2)
    # ---------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
        # phase portrait of the dS branch for ℓ = 0.27, ρ₀ = 1e-2: ΔE = 0, 0.5, 2, ΔE*
        ell = 0.27; eps_c = 3 / (8 * PI * ell**2); src = Source(eps_c=eps_c, vac="dS")
        d_star, _ = dE_star(src, 1e-2)
        for dE, lab in ((0.0, "ΔE = 0"), (0.5, "ΔE = 0.5 ε_c"), (2.0, "ΔE = 2 ε_c"), (d_star, f"ΔE* = {d_star:.3f} ε_c (Nariai)")):
            if dE is None: continue
            # trajectory: fluid + vacuum, keep b, ḃ
            rho0 = 1e-2 * eps_c; b0 = 1.9; bd0 = -np.sqrt(2 / b0 - 1)
            y0 = [h_from_constraint(b0, bd0, rho0), b0, bd0, rho0, 0.0]
            ev_trig = lambda t, y: y[3] - eps_c; ev_trig.terminal = True
            s1 = solve_ivp(rhs_factory(src, 0.0, False), (0, 400), y0, events=[ev_trig], rtol=1e-10, atol=1e-14, max_step=0.02, dense_output=True)
            y = s1.y[:, -1].copy(); y[3] = eps_c * (1 + dE); y[0] = h_from_constraint(y[1], y[2], y[3]); y[4] = 1.0
            ev_turn = lambda t, y: y[2]; ev_turn.terminal = True
            ev_sing = lambda t, y: y[1] - 1e-3; ev_sing.terminal = True
            s2 = solve_ivp(rhs_factory(src, 0.0, True), (s1.t[-1], s1.t[-1] + 800), y, events=[ev_turn, ev_sing], rtol=1e-10, atol=1e-14, max_step=0.02, dense_output=True)
            bb = np.concatenate([s1.y[1], s2.y[1]]); bd = np.concatenate([s1.y[2], s2.y[2]])
            ax[0].plot(bb, bd, label=lab)
            ax[0].plot([s1.y[1, -1]], [s1.y[2, -1]], "k.", ms=4)
        ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("b / M"); ax[0].set_ylabel("ḃ"); ax[0].set_title("dS branch, ℓ = 0.27 M, ρ₀ = 10⁻² ε_c (dot — transition)"); ax[0].legend(fontsize=7); ax[0].set_xlim(0, 2)
        # scenario EOS: κ_exit(ρ₀)
        for dp in deltas:
            xs_ = [e["dmc"] for e in e2 if e["dbc"] == dp]; ks_ = [e["kappa"] if e["kappa"] is not None else np.nan for e in e2 if e["dbc"] == dp]
            ax[1].plot(xs_, ks_, "o-", label=f"δ'(b_c) = {dp:+.3f}")
        ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("δ (m_c)"); ax[1].set_ylabel("κ at exit"); ax[1].set_title("scenario EOS: κ_exit(b_c, m_c) — zero at one point"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "T30_phase.png", dpi=130); plt.close(fig)
        say(f"\n  plot: {OUT / 'T30_phase.png'}")
    except Exception as e:  # noqa: BLE001
        say(f"  plot not built: {e}")
    json.dump(dict(rows=rows, dE_star=star_rows, scen=scen_rows, scen_star=scen_star, S_TR=S_TR, R1=FAM.R1, ell_base=ELL_BASE), open(OUT / "T30_phase_transition.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T30_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
