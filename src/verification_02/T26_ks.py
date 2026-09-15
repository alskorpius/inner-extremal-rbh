"""T26. The region between the horizons (f < 0) as Kantowski-Sachs (KS) cosmology.
Metric of the T-region: ds^2 = -dr^2/|f| + |f| dt^2 + r^2 dΩ^2 = -dτ^2 + a(τ)^2 dx^2 + b(τ)^2 dΩ^2,
    dτ = dr/√|f|, a = √|f|, b = r, x = t; choose τ increasing as r decreases (inward):  ḃ = -√(-f),  ȧ = f'/2.
KS equations (source T^τ_τ = -ρ_KS, T^x_x = p_x, T^θ_θ = p_⊥; in the T-region ρ_KS = -T^r_r = ρ, p_x = T^t_t = -ρ for the class T^t_t = T^r_r):
    (C)  2(ȧ/a)(ḃ/b) + ḃ²/b² + 1/b² = 8πρ
    (X)  2 b̈/b + ḃ²/b² + 1/b² = -8π p_x
    (Θ)  ä/a + b̈/b + (ȧ/a)(ḃ/b) = -8π p_⊥
    (S)  ρ̇ + (ȧ/a)(ρ + p_x) + 2(ḃ/b)(ρ + p_⊥) = 0
For p_x = -ρ:  ρ = ρ(b),  p_⊥ = -ρ - bρ'(b)/2,  ḃ² = -f(b) = 2m(b)/b - 1,  b̈ = -f'(b)/2  (one-dimensional conservative system).
Parts: (1) identity checks on the base profile, Schwarzschild, de Sitter; (2) phase portrait (b, ḃ) for m -> (1+ε)m and proper time
to R₋ (diverges as (b-R₋)^{-1/2} at a triple root, finite at a simple one; total τ ∝ |ε|^{-1/6}); (3) Lagrangians in KS: EOS fluid,
scalar with a potential (Nariai fixed points -- Jacobian), k-essence (ghost condensate) -- no-go for an open set of regular exits.
Run: python src/verification_02/T26_ks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
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

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- background
BASE0 = dict(u1=1.0, u2=4.0, u3=8.0, w1=0.25, w2=0.35, w3=0.1, s_end=24.0)


class Profile:
    """f, f', f'', ρ, ρ' of the base profile (M = 1), with scaling m -> (1+eps) m."""

    def __init__(self, bg, eps=0.0):
        self.bg, self.eps = bg, eps

    def m(self, r):
        return (1 + self.eps) * self.bg.fields(np.atleast_1d(np.asarray(r, float)))["m"]

    def parts(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        d = self.bg.fields(r)
        k = 1 + self.eps
        m, mp, mpp = k * d["m"], k * d["mp"], k * d["mpp"]
        f = 1 - 2 * m / r
        fp = 2 * m / r**2 - 2 * mp / r
        fpp = -4 * m / r**3 + 4 * mp / r**2 - 2 * mpp / r
        rho = mp / (4 * np.pi * r**2)
        drho = (mpp - 2 * mp / r) / (4 * np.pi * r**2)
        return f, fp, fpp, rho, drho

    def f(self, r):
        return self.parts(r)[0]

    def roots(self, lo=1e-3, hi=4.0, n=400001):
        r = np.geomspace(lo, hi, n)
        f = self.f(r)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        return [brentq(lambda x: self.f(np.array([x]))[0], r[i], r[i + 1], xtol=1e-14) for i in idx]


def part1(P):
    say("(1) KS equations on the base profile: identity checks")
    r = np.linspace(0.75, 1.9, 2000)          # T-region: R- = 0.677 < r < R+ = 1.985
    f, fp, fpp, rho, drho = P.parts(r)
    assert np.all(f < 0)
    # ḃ = -√(-f), ȧ/a = f'/(2√(-f)) (by construction): (C) => -f'/b + (1-f)/b² = 8πρ
    lhs_C = -fp / r + (1 - f) / r**2
    lhs_X = -fp / r + (1 - f) / r**2                     # 2b̈/b + ḃ²/b² + 1/b² with b̈ = -f'/2, ḃ² = -f
    lhs_T = -fpp / 2 - fp / r                            # ä/a + b̈/b + (ȧ/a)(ḃ/b)
    p_perp = -rho - r * drho / 2
    eC = np.max(np.abs(lhs_C / (8 * np.pi * rho) - 1))
    eX = np.max(np.abs(lhs_X / (8 * np.pi * rho) - 1))   # -8π p_x = 8πρ
    eT = np.max(np.abs(lhs_T / (-8 * np.pi * p_perp) - 1))
    say(f"  (C): max|LHS/(8πρ) - 1| = {eC:.1e};  (X) with p_x = -ρ: {eX:.1e};  (Θ) with p_⊥ = -ρ - bρ'/2: {eT:.1e}")
    # Schwarzschild: f = 1 - 2/b => LHS_C = 2/b³·... = -(2/b²)/b + (2/b)/b² = 0 ✓; de Sitter f = 1 - b²/ℓ²: -f'/b + (1-f)/b² = 2/ℓ² + 1/ℓ² = 3/ℓ² = 8πρ_dS ✓
    b = np.array([1.2]); ell = 0.271
    schw = -(2 / b**2) / b + (2 / b) / b**2
    ds = (2 * b / ell**2) / b + (b**2 / ell**2) / b**2
    say(f"  Schwarzschild: LHS_C = {schw[0]:.1e} (=0);  de Sitter: LHS_C·ℓ²/3 = {ds[0]*ell**2/3:.6f} (=1);  conservation (S) with p_x = -ρ ⇔ dρ/db = -2(ρ+p_⊥)/b -- an identity by the definition of p_⊥")
    # trajectory b(τ) from R+ downward: ḃ = -√(-f)
    roots = P.roots()
    Rm, Rp = roots[0], roots[-1]
    say(f"  base profile roots: {[round(x, 6) for x in roots]} (R- = {Rm:.6f}, R+ = {Rp:.6f})")
    sol = solve_ivp(lambda t, y: [-np.sqrt(max(-P.f(np.array([y[0]]))[0], 0.0))], (0, 60.0), [Rp * (1 - 1e-6)], rtol=1e-10, atol=1e-13, dense_output=True, max_step=0.05)
    tau = np.linspace(0, sol.t[-1], 2000); bt = sol.sol(tau)[0]
    bdot2 = np.gradient(bt, tau) ** 2
    chk = np.max(np.abs(bdot2[100:-100] + P.f(bt[100:-100])))
    say(f"  trajectory b(τ) from R+: over τ = {sol.t[-1]:.1f} M it reached b = {bt[-1]:.5f} (R- = {Rm:.5f}); max|ḃ² + f(b)| = {chk:.1e} -- the profile f(b) = -ḃ² is recovered as a single trajectory")
    return Rm, Rp, dict(eC=eC, eX=eX, eT=eT, tau_end=sol.t[-1], b_end=float(bt[-1]))


def tau_to(P, b_from, b_to):
    """proper time ∫_{b_to}^{b_from} dr/√(-f)."""
    g = lambda r: 1.0 / np.sqrt(-P.f(np.array([r]))[0])
    val, err = quad(g, b_to, b_from, limit=400)
    return val


def part2(bg, Rm0, Rp0):
    say("\n(2) Phase portrait and proper time to R-")
    P0 = Profile(bg, 0.0)
    # divergence of τ at the triple root: τ(x) = ∫_{R-+x}^{R+} dr/√(-f)
    xs = np.array([3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    taus = np.array([tau_to(P0, Rp0 * (1 - 1e-9), Rm0 + x) for x in xs])
    slope = np.polyfit(np.log(xs[2:]), np.log(taus[2:]), 1)[0]
    say(f"  τ(R- + x): x = {xs.tolist()}\n           τ = {[round(float(t), 3) for t in taus]};  slope d ln τ/d ln x (x ≤ 3e-2) = {slope:.3f} (expected -1/2 for f ∝ x³)")
    rows = []
    eps_list = [0.0, 1e-4, -1e-4, 1e-2, -1e-2, 1e-1, -1e-1]
    portrait = {}
    for eps in eps_list:
        P = Profile(bg, eps)
        roots = P.roots()
        Rp = roots[-1]
        Rm = roots[0] if eps == 0.0 else (roots[-2] if len(roots) >= 2 else np.nan)
        f, fp, fpp, _, _ = P.parts(np.array([Rm]))
        kappa = fp[0] / 2
        bdd = -fp[0] / 2
        if eps == 0.0:
            tau_tot = np.inf
        else:
            tau_tot = tau_to(P, Rp * (1 - 1e-9), Rm * (1 + 1e-9))
        r = np.linspace(Rm * (1 + 1e-6), Rp * (1 - 1e-6), 1500)
        portrait[eps] = (r, -np.sqrt(-P.f(r)))
        rows.append(dict(eps=eps, n_roots=len(roots), R_minus=Rm, R_plus=Rp, kappa_minus=kappa, bdd_at_Rminus=bdd, tau_total=tau_tot))
        say(f"  ε = {eps:+.0e}: roots {len(roots)}, R- = {Rm:.5f}, R+ = {Rp:.5f}, κ- = {kappa:+.2e}, b̈(R-) = -f'/2 = {bdd:+.2e}, τ(R+→R-) = {tau_tot if np.isfinite(tau_tot) else '∞'}")
    # τ_total ∝ |ε|^{-1/6}
    e_abs = [1e-4, 1e-2, 1e-1]
    for sgn in (+1, -1):
        tt = [r["tau_total"] for r in rows if r["eps"] in [sgn * e for e in e_abs]]
        p = np.polyfit(np.log(e_abs), np.log(tt), 1)[0]
        say(f"  slope d ln τ_tot / d ln|ε| for sign(ε) = {sgn:+d}: {p:.3f} (analytics: -1/6)")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for eps, (r, bd) in portrait.items():
            ax[0].plot(r, bd, lw=2.2 if eps == 0 else 1.0, label=f"ε = {eps:+.0e}" if eps else "ε = 0 (triple root)")
        ax[0].set_xlabel("b = r / M"); ax[0].set_ylabel("ḃ = db/dτ"); ax[0].set_title("Phase portrait of the T-region, ḃ = −√(−f_ε(b))"); ax[0].legend(fontsize=7); ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlim(0.5, 2.0)
        ax[1].loglog(xs, taus, "o-", label="τ(R₋+x), ε = 0")
        ax[1].loglog(xs, taus[2] * (xs / xs[2]) ** -0.5, "--", label="∝ x^{-1/2}")
        ax[1].set_xlabel("x = b − R₋"); ax[1].set_ylabel("τ, M"); ax[1].set_title("Proper time to R₋ (triple root)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(OUT / "T26_phase.png", dpi=130); plt.close(fig)
        say(f"  plot: {OUT / 'T26_phase.png'}")
    except Exception as e:  # noqa: BLE001
        say(f"  plot not built: {e}")
    return rows, dict(x=xs.tolist(), tau=taus.tolist(), slope=slope)


# ---------------------------------------------------------------- (3b) scalar in KS
def scalar_system(V, dV, d2V):
    """state y = (h, b, bd, phi, psi), h = ȧ/a, psi = φ̇. Source: ρ = ψ²/2 + V, p_x = p_⊥ = ψ²/2 - V."""
    def rhs(t, y):
        h, b, bd, phi, psi = y
        p = psi**2 / 2 - V(phi)
        bdd = (b / 2) * (-8 * np.pi * p - bd**2 / b**2 - 1 / b**2)
        hd = -8 * np.pi * p - bdd / b - h * bd / b - h**2
        psid = -(h + 2 * bd / b) * psi - dV(phi)
        return [hd, bd, bdd, psi, psid]

    def constraint(y):
        h, b, bd, phi, psi = y
        return 2 * h * bd / b + bd**2 / b**2 + 1 / b**2 - 8 * np.pi * (psi**2 / 2 + V(phi))
    return rhs, constraint


def jacobian(rhs, y0, h=1e-6):
    n = len(y0); J = np.zeros((n, n)); f0 = np.array(rhs(0, y0))
    for j in range(n):
        yp = np.array(y0, float); yp[j] += h
        ym = np.array(y0, float); ym[j] -= h
        J[:, j] = (np.array(rhs(0, yp)) - np.array(rhs(0, ym))) / (2 * h)
    return J


def part3b():
    say("\n(3b) Scalar field φ(τ) with a potential in KS: Nariai fixed points (b = b*, φ = φ*, V'(φ*) = 0, 1/b*² = 8πV*, h = ±1/b*)")
    V0, mu = 0.05, 2.0
    V = lambda p: V0 * (1 + mu * (p**2 - 1) ** 2)
    dV = lambda p: V0 * mu * 4 * p * (p**2 - 1)
    d2V = lambda p: V0 * mu * (12 * p**2 - 4)
    rhs, constr = scalar_system(V, dV, d2V)
    out = []
    for phi_s, kind in ((1.0, "minimum of V"), (0.0, "maximum of V")):
        Vs = V(phi_s); bs = 1 / np.sqrt(8 * np.pi * Vs)
        for hs in (+1 / bs, -1 / bs):
            y0 = [hs, bs, 0.0, phi_s, 0.0]
            res = np.max(np.abs(rhs(0, y0)))
            J = jacobian(rhs, y0)
            ev = np.linalg.eigvals(J)
            # analytics: {±1/b*} ∪ roots of s² + h* s + V'' = 0 ∪ {-2h*}
            roots_phi = np.roots([1, hs, d2V(phi_s)])
            ana = sorted([1 / bs, -1 / bs, -2 * hs] + list(roots_phi), key=lambda z: (np.real(z), np.imag(z)))
            num = sorted(ev, key=lambda z: (np.real(z), np.imag(z)))
            say(f"  φ* = {phi_s} ({kind}), V* = {Vs:.4f}, b* = {bs:.4f}, h* = {hs:+.4f}: |rhs| = {res:.1e}, |C| = {abs(constr(y0)):.1e}")
            say(f"      Jacobian eigenvalues: {np.array2string(np.array(num), precision=4)}")
            say(f"      analytics {{±1/b*, -2h*, roots of s²+h*s+V''}}: {np.array2string(np.array(ana), precision=4)}  -> max Re = {max(np.real(ev)):+.4f} > 0: saddle")
            out.append(dict(phi=phi_s, kind=kind, b=bs, h=hs, eig=[[float(np.real(z)), float(np.imag(z))] for z in ev]))
        # trajectories: a small displacement δb of either sign on the constraint surface (h from the constraint for small ḃ)
        # perturbations on the constraint surface at h = h* = +1/b*: δb = h* b*² ḃ (linear) -- along the growing (ḃ > 0) and the contracting (ḃ < 0) direction
        hs = 1 / bs
        for bd0 in (+1e-3, -1e-3):
            b0, phi0, psi0 = bs + hs * bs**2 * bd0, phi_s, 0.0
            h0 = (8 * np.pi * V(phi0) - bd0**2 / b0**2 - 1 / b0**2) * b0 / (2 * bd0)
            y0 = [h0, b0, bd0, phi0, psi0]
            db = (b0 - bs) / bs
            ev_b = lambda t, y: y[1] - 0.2 * bs; ev_b.terminal = True
            ev_B = lambda t, y: y[1] - 5 * bs; ev_B.terminal = True
            s = solve_ivp(rhs, (0, 200 * bs), y0, rtol=1e-10, atol=1e-13, events=[ev_b, ev_B], max_step=0.05 * bs)
            fate = "b → 0 (sphere collapses)" if s.t_events[0].size else ("b → ∞ (sphere inflates)" if s.t_events[1].size else "b stayed in the vicinity")
            say(f"      start δb/b* = {db:+.0e}, ḃ = {bd0}: over τ = {s.t[-1]:.2f} ({s.t[-1]/bs:.1f} b*) -- {fate}; |C| at the end {abs(constr(s.y[:, -1])):.1e}; φ at the end {s.y[3, -1]:.4f}")
    return out


def part3c():
    say("\n(3c) k-essence L = K(X), X = φ̇²/2, K = -X + X²/(2X*): ghost condensate K'(X*) = 0, ρ* = -K(X*) = X*/2, p* = -ρ*")
    # field equation: d/dτ (a b² K'(X) φ̇) = 0 => K'(X) φ̇ = C/(a b²). Test field on an SdS-like background of the T-region of the base profile.
    bg = qb.FamilyBG(qb.build_family(BASE0, 1.0))
    P = Profile(bg, 0.0)
    roots = P.roots(); Rm, Rp = roots[0], roots[-1]
    Xs = 0.5
    say("  branch C = 0: X ≡ X* exactly for any a, b -- the source is identically Λ = 8πρ* (Schwarzschild-de Sitter in the T-region: f = 1 - 2M/b - Λb²/3, no more than two positive roots, no degenerate inner one)")
    r = np.array([1.9, 1.5, 1.2, 1.0, 0.8, 0.7, 0.68, Rm + 1e-4])
    f = P.f(r); ab2 = np.sqrt(-f) * r**2
    for C in (1e-3, 1e-2):
        # K' φ̇ = (X/X* - 1)·√(2X) = C/(ab²) -- solve for X at each b
        Xv = []
        for v in C / ab2:
            g = lambda X: (X / Xs - 1) * np.sqrt(2 * X) - v
            Xv.append(brentq(g, Xs * (1 + 1e-12), 1e9) if v > 0 else Xs)
        Xv = np.array(Xv)
        say(f"  C = {C:.0e}: b = {np.array2string(r, precision=4)}\n            a b² = {np.array2string(ab2, precision=3)}\n            X/X* − 1 = {np.array2string(Xv / Xs - 1, precision=3)}  -- diverges as a → 0 (any horizon), regular only at C = 0")
    return dict(C_zero="Λ", note="C≠0: X-X* ∝ 1/(a b²) → ∞ at the horizon")


def main():
    bg = qb.FamilyBG(qb.build_family(BASE0, 1.0))
    P0 = Profile(bg, 0.0)
    Rm, Rp, p1 = part1(P0)
    rows, tau_info = part2(bg, Rm, Rp)
    say("\n(3a) Fluid with EOS p_x = -ρ, p_⊥ = w(ρ)ρ: (S) gives d ln ρ/d ln b = -2(1 + w(ρ)) ⇒ σ = 2(1 + w); the equation is autonomous in ln b ⇒ the solutions are shifts"
        " ρ(b) = P(b/b₀) (one free scale b₀); triple root: w(ρ*) = 0 at the point b* = ξ* b₀ and 8πb*²ρ* = 1 ⇒ b₀ is fixed ⇒ M* = m(∞) is unique: codimension 1."
        " A scale-free EOS (w = const ⇔ power-law profile) has no slope dip ⇒ no triple root.")
    # illustration: base w(ρ) and scale shift
    r = np.linspace(0.05, 1.9, 400); f, fp, fpp, rho, drho = P0.parts(r)
    w = (-rho - r * drho / 2) / rho
    seg = (r > 0.5) & (r < 0.9)
    say(f"  base: w(ρ) ranges from {w.min():+.3f} (core, σ = 0) to {w.max():+.3f} (cutoff); descending zero of w (σ = 2, p_⊥ = 0) at b = {r[seg][np.argmin(np.abs(w[seg]))]:.3f} (R- = {Rm:.3f})")
    sc = 1.01
    say(f"  scale shift b₀ → {sc} b₀ at the same EOS: 8πR-²ρ(R-) → {sc**2:.4f}, M → {sc:.2f} M: κ- ≠ 0 by T1 (ε ≈ {sc-1:.2f} → κ- ≈ -{2.0*(sc-1)**(2/3):.3f})")
    p3b = part3b()
    p3c = part3c()
    say("\nSummary (3): a regular exit from the T-region = a trajectory asymptotically approaching a fixed point (b*, ḃ = 0) (a degenerate horizon in the static picture)."
        " For a smooth autonomous system such trajectories form the stable manifold of a saddle (measure zero) or a degenerate point (triple root: coefficient"
        " 1/b*² + 4πb*ρ'(b*) = 0) -- there is no open set, by uniqueness of solutions. A sink would require a term with ḃ in p_x (a dissipative, non-Lagrangian medium) or a non-smooth right-hand side.")
    json.dump(dict(part1=p1, part2=rows, tau=tau_info, scalar=p3b, kessence=p3c), open(OUT / "T26_ks.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T26_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
