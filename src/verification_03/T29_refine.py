"""T29 (refinement): attractor window in ζ₀ for n = 0, 1/2, 1; check of the asymptotic (nodal) approach without a crossing event;
T1 test at n = 0 in the window; Israel-Stewart at n = 0; diagnostics of the sign of entropy production and NEC. Writes logs/verification_03/T29_refine_log.txt, T29_refine.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import T29_viscosity as T
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)

def run_free(med, zeta, eps, tau_max=300.0, IS_tau=None):
    """like run_I, but without stopping at ḃ = 0: track whether v crosses zero (simple exit) or approaches asymptotically (node)."""
    P = med
    roots = np.array(P.r)[np.nonzero(np.diff(np.sign(1 - 2 * (1 + eps) * P.m_b / P.r)) != 0)[0]]
    Rp = roots[-1]; b0 = Rp * (1 - 1e-5)
    m0 = (1 + eps) * P.m_of_b(b0); rho0 = (1 + eps) * P.rho_of_b(b0); v0 = -np.sqrt(max(2 * m0 / b0 - 1, 1e-14))
    def rhs(t, y):
        b, v, rho = y[0], y[1], y[2]
        bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
        w = P.w(rho)
        if IS_tau is None:
            rhod = -(2 * v / b) * rho * (1 + w) + (2 * zeta(rho) / b) * (bdd + 2 * v**2 / b)
            return [v, bdd, rhod]
        Pi = y[3]
        rhod = -(2 * v / b) * (rho * (1 + w) + Pi)
        Theta = (bdd / v if abs(v) > 1e-12 else 0.0) + 2 * v / b
        return [v, bdd, rhod, -(Pi + zeta(rho) * Theta) / IS_tau]
    ev_coll = lambda t, y: y[0] - 0.05; ev_coll.terminal = True
    ev_neg = lambda t, y: y[2]; ev_neg.terminal = True
    ev_pos = lambda t, y: y[1] - 1e-3; ev_pos.terminal = True   # ḃ > 0: re-expansion (simple exit with crossing)
    y0 = [b0, v0, rho0] + ([0.0] if IS_tau is not None else [])
    s = solve_ivp(rhs, (0, tau_max), y0, rtol=1e-10, atol=1e-13, events=[ev_coll, ev_neg, ev_pos], max_step=0.02)
    b, v, rho = s.y[0], s.y[1], s.y[2]
    bdd = 4 * np.pi * b * rho - (v**2 + 1) / (2 * b)
    q = 8 * np.pi * b**2 * rho - 1
    fate = "collapse" if s.t_events[0].size else ("rho<0" if s.t_events[1].size else ("recross" if s.t_events[2].size else "asymptotic"))
    # approach rate: ln|v| vs τ over the last third
    k = slice(len(s.t) * 2 // 3, None)
    rate = np.polyfit(s.t[k], np.log(np.abs(v[k]) + 1e-300), 1)[0] if fate == "asymptotic" and len(s.t) > 30 else np.nan
    w = P.w(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        Theta = np.where(np.abs(v) > 1e-14, bdd / v, np.nan) + 2 * v / b
    z = zeta(rho)
    pp = w * rho - z * Theta if IS_tau is None else w * rho + s.y[3]
    nec = (rho + pp) / rho
    # Eckart entropy production: σ_s = ζΘ²/T ≥ 0 ⇔ ζ ≥ 0; here ζ<0 => negative
    return dict(fate=fate, tau=float(s.t[-1]), b_end=float(b[-1]), v_end=float(v[-1]), bdd_end=float(bdd[-1]), q_end=float(q[-1]), rate=float(rate),
                nec_min=float(np.nanmin(nec)), w_end=float(w[-1]), Theta_end=float(Theta[-1]) if np.isfinite(Theta[-1]) else np.nan,
                v_min=float(np.min(np.abs(v[-50:]))), rho_end=float(rho[-1]))

def main():
    bg = T.qb.FamilyBG(T.qb.build_family(T.BASE0, 1.0)); med = T.Medium(bg)
    eps_list = [0.0, 1e-4, -1e-4, 1e-2, -1e-2, 1e-1, -1e-1]
    say("T29 refinement: attractor window (variant I), approach without a crossing event")
    say("  n | ζ₀ | fates (a = asymptotic approach to ḃ = 0, r = re-expansion ḃ > 0 (simple exit), c = collapse, n = ρ < 0) | b̈ at end | b_end | approach rate d ln|v|/dτ | min NEC_⊥")
    scan = []
    grid = {0.0: [-0.03, -0.05, -0.07, -0.1, -0.15, -0.2, -0.3, -0.5, -0.7, -1.0, 0.01, 0.1],
            0.5: [-0.03, -0.05, -0.07, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, 0.1, 1.0],
            1.0: [-0.03, -0.05, -0.07, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, 0.1, 1.0]}
    for n, zs in grid.items():
        for z0 in zs:
            res = [run_free(med, T.zeta_fn(z0, n), e) for e in eps_list]
            fates = "".join({"asymptotic": "a", "recross": "r", "collapse": "c", "rho<0": "n"}[r["fate"]] for r in res)
            degen = np.mean([r["fate"] == "asymptotic" and abs(r["bdd_end"]) < 1e-3 and abs(r["v_end"]) < 1e-3 for r in res])
            scan.append(dict(n=n, zeta0=z0, fates=fates, degen_frac=float(degen), bdd_end=[r["bdd_end"] for r in res], b_end=[r["b_end"] for r in res], rate=[r["rate"] for r in res], nec_min=[r["nec_min"] for r in res], q_end=[r["q_end"] for r in res], w_end=[r["w_end"] for r in res]))
            say(f"  {n:3.1f} | {z0:+.2f} | {fates} | " + " ".join(f"{r['bdd_end']:+.0e}" for r in res) + " | " + " ".join(f"{r['b_end']:.3f}" for r in res) + " | " + " ".join(f"{r['rate']:+.2f}" if np.isfinite(r['rate']) else "  —  " for r in res) + f" | {min(r['nec_min'] for r in res):+.2e} | degenerate fraction {degen:.2f}")
    # T1 test in the n = 0 window
    say("\n  T1 test (n = 0, ζ₀ = −0.2): κ_eff = −b̈ and ḃ at end vs ε; landing point b*; w(ρ*) and Θ at end")
    t1 = []
    for e in [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 2e-1, -1e-5, -1e-4, -1e-3, -1e-2, -3e-2, -1e-1, -2e-1]:
        r = run_free(med, T.zeta_fn(-0.2, 0.0), e, tau_max=400.0)
        t1.append(dict(eps=e, **{k: r[k] for k in ("fate", "b_end", "v_end", "bdd_end", "q_end", "rate", "nec_min", "w_end", "Theta_end", "rho_end")}))
        say(f"    ε = {e:+.0e}: {r['fate']:10s} b* = {r['b_end']:.4f}, ḃ = {r['v_end']:+.1e}, b̈ = {r['bdd_end']:+.1e}, q = {r['q_end']:+.1e}, rate {r['rate']:+.3f}, w(ρ*) = {r['w_end']:+.3f}, Θ_end = {r['Theta_end']:+.3f}, min NEC_⊥ = {r['nec_min']:+.2f}")
    # linear theory at the landing point: s² − 8πζs + 8πρw = 0
    r = t1[0]
    rho_s = 1 / (8 * np.pi * r["b_end"] ** 2); w_s = r["w_end"]; z = -0.2
    disc = (8 * np.pi * z) ** 2 - 4 * 8 * np.pi * rho_s * w_s
    s1 = (8 * np.pi * z + np.sqrt(disc + 0j)) / 2; s2 = (8 * np.pi * z - np.sqrt(disc + 0j)) / 2
    say(f"    linear theory at b* = {r['b_end']:.3f}: ρ* = {rho_s:.4f}, w* = {w_s:+.3f}, ζ = {z}: s = {s1:.3f}, {s2:.3f}; node when |ζ| ≥ √(ρw/2π) = {np.sqrt(max(rho_s*w_s,0)/(2*np.pi)):.3f}; observed rate {r['rate']:+.3f}")
    # Israel-Stewart at n = 0
    say("\n  Israel-Stewart, n = 0, ζ₀ = −0.2 and −0.1: τ_IS = 0.03, 0.1, 0.3, 1")
    isr = []
    for z0 in (-0.2, -0.1):
        for tis in (0.03, 0.1, 0.3, 1.0):
            res = [run_free(med, T.zeta_fn(z0, 0.0), e, IS_tau=tis) for e in eps_list]
            fates = "".join({"asymptotic": "a", "recross": "r", "collapse": "c", "rho<0": "n"}[r["fate"]] for r in res)
            degen = np.mean([r["fate"] == "asymptotic" and abs(r["bdd_end"]) < 1e-3 and abs(r["v_end"]) < 1e-3 for r in res])
            isr.append(dict(zeta0=z0, tau_IS=tis, fates=fates, degen_frac=float(degen), bdd_end=[r["bdd_end"] for r in res], b_end=[r["b_end"] for r in res]))
            say(f"    ζ₀ = {z0:+.1f}, τ_IS = {tis}: {fates} | b̈ " + " ".join(f"{r['bdd_end']:+.0e}" for r in res) + " | b_end " + " ".join(f"{r['b_end']:.3f}" for r in res) + f" | degenerate fraction {degen:.2f}")
    json.dump(dict(scan=scan, t1=t1, IS=isr), open(OUT / "T29_refine.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T29_refine_log.txt").write_text("\n".join(LOG), encoding="utf-8")

if __name__ == "__main__":
    main()
