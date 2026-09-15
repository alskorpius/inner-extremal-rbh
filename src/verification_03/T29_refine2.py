"""T29 (refinement 2): (a) sensitivity of the landing to the starting offset δ from R₊ (Eckart singularity at the horizon); (b) energy balance
of the "viscous" term at the attractor; (c) Israel-Stewart at n = 0 and 1; (d) T1 test in the window. Run: python src/verification_03/T29_refine2.py [abcd]."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
PARTS = sys.argv[1] if len(sys.argv) > 1 else "abcd"
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))
import T29_viscosity as T  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def run(med, zeta, eps, tau_max=300.0, IS_tau=None, delta=1e-5):
    P = med
    roots = np.array(P.r)[np.nonzero(np.diff(np.sign(1 - 2 * (1 + eps) * P.m_b / P.r)) != 0)[0]]
    if roots.size == 0:
        return dict(fate="no-horizon", b_end=np.nan, v_end=np.nan, bdd_end=np.nan, m_end=np.nan, rho_end=np.nan, W_visc=np.nan, W_adiab=np.nan, frac_Theta_pos=np.nan, Pi_max=np.nan, nec_min=np.nan, tau=0.0)
    Rp = roots[-1]; b0 = Rp * (1 - delta)
    m0 = (1 + eps) * P.m_of_b(b0); rho0 = (1 + eps) * P.rho_of_b(b0); v0 = -np.sqrt(max(2 * m0 / b0 - 1, 1e-14))

    def rhs(t, y):
        b, v, rho = y[0], y[1], y[2]
        bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
        w = P.w(rho)
        if IS_tau is None:
            return [v, bdd, -(2 * v / b) * rho * (1 + w) + (2 * zeta(rho) / b) * (bdd + 2 * v**2 / b)]
        Pi = y[3]
        Theta = (bdd / v if abs(v) > 1e-12 else 0.0) + 2 * v / b
        return [v, bdd, -(2 * v / b) * (rho * (1 + w) + Pi), -(Pi + zeta(rho) * Theta) / IS_tau]

    ev_coll = lambda t, y: y[0] - 0.05; ev_coll.terminal = True
    ev_neg = lambda t, y: y[2]; ev_neg.terminal = True
    ev_pos = lambda t, y: y[1] - 1e-3; ev_pos.terminal = True
    y0 = [b0, v0, rho0] + ([0.0] if IS_tau is not None else [])
    s = solve_ivp(rhs, (0, tau_max), y0, rtol=1e-10, atol=1e-13, events=[ev_coll, ev_neg, ev_pos], max_step=0.02, method="LSODA" if IS_tau is not None else "RK45")
    b, v, rho = s.y[0], s.y[1], s.y[2]
    bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
    fate = "collapse" if s.t_events[0].size else ("rho<0" if s.t_events[1].size else ("recross" if s.t_events[2].size else "asymptotic"))
    with np.errstate(divide="ignore", invalid="ignore"):
        Theta = np.where(np.abs(v) > 1e-14, bdd / v, np.nan) + 2 * v / b
    z = zeta(rho)
    Pi = -z * Theta if IS_tau is None else s.y[3]
    dW = (2 * np.abs(v) / b) * Pi
    ok = np.isfinite(dW)
    W = float(np.trapezoid(dW[ok], s.t[ok])) if ok.sum() > 2 else np.nan
    Wad = float(np.trapezoid(((2 * np.abs(v) / b) * rho * (1 + P.w(rho)))[ok], s.t[ok])) if ok.sum() > 2 else np.nan
    frac_pos = float(np.nanmean(Theta[ok] > 0)) if ok.sum() else np.nan
    return dict(fate=fate, tau=float(s.t[-1]), b_end=float(b[-1]), v_end=float(v[-1]), bdd_end=float(bdd[-1]), m_end=float(b[-1] * (1 + v[-1] ** 2) / 2),
                rho_end=float(rho[-1]), W_visc=W, W_adiab=Wad, frac_Theta_pos=frac_pos, Pi_max=float(np.nanmax(np.abs(Pi[ok]))) if ok.sum() else np.nan,
                nec_min=float(np.nanmin((rho + P.w(rho) * rho + Pi)[ok] / rho[ok])) if ok.sum() else np.nan)


FMAP = {"asymptotic": "a", "recross": "r", "collapse": "c", "rho<0": "n", "no-horizon": "-"}


def main():
    bg = T.qb.FamilyBG(T.qb.build_family(T.BASE0, 1.0)); med = T.Medium(bg)
    eps_list = [0.0, 1e-4, -1e-4, 1e-2, -1e-2, 1e-1, -1e-1]
    sens, en, isr, t1 = [], [], [], []
    if "a" in PARTS:
        say("(a) Sensitivity of the landing to the starting offset δ from R₊ (Eckart, variant I, n = 0, ζ₀ = −0.2, ε = 0)")
        for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            r = run(med, T.zeta_fn(-0.2, 0.0), 0.0, delta=d)
            sens.append(dict(delta=d, **{k: r[k] for k in ("fate", "b_end", "m_end", "W_visc", "W_adiab", "Pi_max", "frac_Theta_pos", "nec_min")}))
            say(f"    δ = {d:.0e}: {r['fate']}, b* = {r['b_end']:.4f}, m(b*) = {r['m_end']:.4f}, W_visc = {r['W_visc']:+.4f}, W_adiab = {r['W_adiab']:+.4f}, |Π|max = {r['Pi_max']:.2e}, fraction of τ with Θ > 0: {r['frac_Theta_pos']:.2f}, min NEC_⊥ {r['nec_min']:+.2f}")
    if "b" in PARTS:
        say("\n(b) Energy balance at the attractor (n = 0, ζ₀ = −0.2, δ = 1e-5): W_visc — energy injected by the −ζΘ term (∫(2|ḃ|/b)Π dτ), W_adiab — adiabatic compression")
        for e in eps_list:
            r = run(med, T.zeta_fn(-0.2, 0.0), e)
            en.append(dict(eps=e, **{k: r[k] for k in ("fate", "b_end", "m_end", "rho_end", "W_visc", "W_adiab", "frac_Theta_pos", "nec_min")}))
            say(f"    ε = {e:+.0e}: {r['fate']}, b* = {r['b_end']:.3f}, m* = {r['m_end']:.3f}, ρ* = {r['rho_end']:.4f}, W_visc = {r['W_visc']:+.4f}, W_adiab = {r['W_adiab']:+.4f}, Θ > 0 for {r['frac_Theta_pos']:.0%} of the path, min NEC_⊥ {r['nec_min']:+.2f}")
    if "c" in PARTS:
        say("\n(c) Israel-Stewart (τ_IS Π̇ + Π = −ζΘ), variant I: n = 0 (ζ₀ = −0.2, +0.1) and n = 1 (ζ₀ = ±0.3); τ_IS = 0.1, 1")
        for n, zs in ((0.0, (-0.2, 0.1)), (1.0, (-0.3, 0.3))):
            for z0 in zs:
                for tis in (0.1, 1.0):
                    res = [run(med, T.zeta_fn(z0, n), e, IS_tau=tis, tau_max=120.0) for e in eps_list]
                    fates = "".join(FMAP[r["fate"]] for r in res)
                    degen = np.mean([r["fate"] == "asymptotic" and abs(r["bdd_end"]) < 1e-3 and abs(r["v_end"]) < 1e-3 for r in res])
                    isr.append(dict(n=n, zeta0=z0, tau_IS=tis, fates=fates, degen_frac=float(degen), b_end=[r["b_end"] for r in res], v_end=[r["v_end"] for r in res], bdd_end=[r["bdd_end"] for r in res], nec_min=[r["nec_min"] for r in res], Pi_max=[r["Pi_max"] for r in res]))
                    say(f"    n = {n:.0f}, ζ₀ = {z0:+.1f}, τ_IS = {tis}: {fates} | b_end " + " ".join(f"{r['b_end']:.3f}" for r in res) + " | ḃ_end " + " ".join(f"{r['v_end']:+.0e}" for r in res) + " | b̈_end " + " ".join(f"{r['bdd_end']:+.0e}" for r in res) + f" | degenerate fraction {degen:.2f} | min NEC_⊥ {min(r['nec_min'] for r in res):+.2f} | |Π|max {max(r['Pi_max'] for r in res):.1e}")
    if "d" in PARTS:
        say("\n(d) T1 test (Eckart, n = 0, ζ₀ = −0.2): landing b*, m*, ḃ, b̈ vs ε")
        for e in [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, -1e-5, -1e-4, -1e-3, -1e-2, -3e-2, -1e-1]:
            r = run(med, T.zeta_fn(-0.2, 0.0), e, tau_max=400.0)
            t1.append(dict(eps=e, **{k: r[k] for k in ("fate", "b_end", "m_end", "v_end", "bdd_end", "nec_min")}))
            say(f"    ε = {e:+.0e}: {r['fate']}, b* = {r['b_end']:.4f}, m* = {r['m_end']:.4f}, ḃ = {r['v_end']:+.0e}, b̈ = {r['bdd_end']:+.0e}, min NEC_⊥ {r['nec_min']:+.2f}")
    json.dump(dict(sens=sens, energy=en, IS=isr, t1=t1), open(OUT / f"T29_refine2_{PARTS}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / f"T29_refine2_{PARTS}_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
