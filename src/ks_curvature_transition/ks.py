"""Stage 1c, part 3e(1): anisotropic variant -- curvature transition in the Kantowski-Sachs region.

The region f < 0 of the metric ds^2 = -f dt^2 + dR^2/f + R^2 dOmega^2 is a homogeneous anisotropic
Kantowski-Sachs cosmology, where R is time. The Schwarzschild interior is its vacuum case. We introduce an
anisotropic vacuum-like source T^t_t = T^R_R = -rho(R) (invariant under boosts in the (t,R) plane),
p_perp = -rho - R rho'/2 (from conservation). Then f = 1 - 2 m(R)/R, m' = 4 pi R^2 rho, and the whole dynamics
of the Kantowski-Sachs region is given by F(R) = -f = 2 m(R)/R - 1: the proper time of a comoving observer is
d tau = dR / sqrt(F).

Curvature transition: the vacuum Kretschmann invariant K = 48 M^2/R^6 reaches the threshold K_c at R = B_c;
the profile rho(R) = rho_c * (1 - tanh((R - B_c)/delta))/2 with normalization m(inf) = M (mass continuity:
outside the layer the metric is exactly Schwarzschild). Then rho_c = 3M/(4 pi B_c^3) (1 + O(delta/B_c)) and a
de Sitter core with ell^2 = B_c^3/(2M) = 3/(8 pi rho_c): the core density is set by the curvature threshold,
not by matter. Open questions: are negative energies needed in the layer (WEC/NEC), what is the transverse
pressure, what happens to K, the horizons, the surface gravity of the inner horizon, the infall time. Limit
delta -> 0: the layer -> a thin spacelike shell with S^theta_theta = 3M/(8 pi B_c^2 sqrt(F_c)) (Israel
junction conditions).

Run: python src/ks_curvature_transition/ks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach_map"))
import approach_map as am  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name


class KSFamily(am.Family):
    """F(R) = 1 - 2 m(R)/R with m from the profile rho(R); derivatives obtained analytically via rho, rho'."""

    def __init__(self, M, B_c, delta, label=None):
        self.M, self.B_c, self.delta = M, B_c, delta
        # normalization of rho_c: m(inf) = M
        raw = lambda R: 0.5 * (1 - np.tanh((R - B_c) / delta))
        I, _ = quad(lambda R: 4 * np.pi * R**2 * raw(R), 0, B_c + 40 * delta, limit=200)
        self.rho_c = M / I
        self._raw = raw
        super().__init__("ks_transition", label or f"KS transition B_c={B_c:.3g} δ={delta:.3g}", None, dict(m=M),
                         x_min=1e-3, x_max=12.0)
        # tabulated mass function for fast lookup
        Rg = np.r_[np.geomspace(1e-6, B_c + 40 * delta, 20000), np.linspace(B_c + 40 * delta, 50.0, 2000)]
        rho = self.rho_c * raw(Rg)
        mg = cumulative_trapezoid(4 * np.pi * Rg**2 * rho, Rg, initial=0.0)
        self._Rg, self._mg = Rg, mg

    def rho(self, R):
        return self.rho_c * self._raw(R)

    def drho(self, R):
        z = (R - self.B_c) / self.delta
        return -self.rho_c * 0.5 / (self.delta * np.cosh(z) ** 2)

    def mass(self, R):
        R = np.asarray(R, dtype=float)
        return np.interp(R, self._Rg, self._mg)

    def geometry(self, x):
        R = np.asarray(x, dtype=float)
        m = self.mass(R)
        m1 = 4 * np.pi * R**2 * self.rho(R)
        m2 = 8 * np.pi * R * self.rho(R) + 4 * np.pi * R**2 * self.drho(R)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1 - 2 * m / R
            fp = 2 * m / R**2 - 2 * m1 / R
            fpp = -4 * m / R**3 + 4 * m1 / R**2 - 2 * m2 / R
            D = (2 * m / R) / R**2
        return dict(R=R, Rp=np.ones_like(R), Rpp=np.zeros_like(R), f=f, fp=fp, fpp=fpp, D=D)


def analyze(M, B_c_over_2M, delta_over_Bc):
    B_c = B_c_over_2M * 2 * M
    delta = delta_over_Bc * B_c
    fam = KSFamily(M, B_c, delta)
    K_c = 48 * M**2 / B_c**6
    ell = np.sqrt(3 / (8 * np.pi * fam.rho_c))
    s, x, T = am.characterize(fam)
    # layer: maximum transverse pressure and minima of the energy conditions
    Rl = np.linspace(max(B_c - 8 * delta, 1e-3), B_c + 8 * delta, 4001)
    Tl = am.tensors(fam, Rl)
    # direct checks of the source formulas: rho and p_perp = -rho - R rho'/2 against the framework eigenvalues
    rho_direct = fam.rho(Rl)
    pt_direct = -rho_direct - Rl * fam.drho(Rl) / 2
    err_rho = float(np.max(np.abs(np.asarray(Tl["rho8pi"]) / (8 * np.pi) - rho_direct) / (rho_direct.max() + 1e-30)))
    err_pt = float(np.max(np.abs(np.asarray(Tl["pt8pi"]) / (8 * np.pi) - pt_direct) / (np.abs(pt_direct).max() + 1e-30)))
    # proper time of a KS comoving observer from the outer horizon to the inner one
    hs = s["horizons"]
    tau_in = None
    if len(hs) >= 2:
        R_in, R_out = hs[0]["R"], hs[-1]["R"]
        Fneg = lambda R: -float(fam.geometry(R)["f"])
        tau_in, _ = quad(lambda R: 1 / np.sqrt(max(Fneg(R), 1e-300)), R_in * (1 + 1e-9), R_out * (1 - 1e-9), limit=400)
    tau_schw = np.pi * M
    # tidal accelerations along the path (a_r = -f''/2, a_perp = -f'/(2R)) in the region f<0
    mask = np.asarray(T["f"]) < 0
    a_r = -0.5 * np.asarray(T["fpp"])[mask]
    a_t = -np.asarray(T["fp"])[mask] / (2 * np.asarray(T["R"])[mask])
    # thin shell (limit): S^th_th = 3M/(8 pi B_c^2 sqrt(F_c)), compare with the p_perp integral over proper time
    F_c = -float(fam.geometry(B_c)["f"])
    S_shell = 3 * M / (8 * np.pi * B_c**2 * np.sqrt(F_c)) if F_c > 0 else np.nan
    pt_int, _ = quad(lambda R: (-fam.rho(R) - R * fam.drho(R) / 2) / np.sqrt(max(-float(fam.geometry(R)["f"]), 1e-300)),
                     B_c - 8 * delta, B_c + 8 * delta, limit=400)
    out = dict(B_c_over_2M=B_c_over_2M, delta_over_Bc=delta_over_Bc, B_c=B_c, delta=delta, K_c=K_c, rho_c=fam.rho_c,
               rho_c_expected=3 * M / (4 * np.pi * B_c**3), ell=ell, ell_over_Bc=ell / B_c,
               n_horizons=s["n_horizons"], R_plus=s.get("R_plus"), R_minus=s.get("R_minus"), kappa_minus=s.get("kappa_minus"),
               kappa_minus_expected_dS=-1 / ell, K_center=s["K_center"], K_center_over_Kc=s["K_center"] / K_c,
               K_max=s["K_max"], x_K_max=s["x_K_max"], K_max_over_Kc=s["K_max"] / K_c,
               T_H_over_schw=s.get("T_H_over_schw"), b_c_over_schw=s.get("b_c_over_schw"), df_at_R3m=s.get("df_at_R3m"),
               min_rho8pi=s["min_rho8pi"], min_nec_r=s["min_nec_r"], min_nec_t=s["min_nec_t"], min_sec=s["min_sec"],
               NEC_violated=s["NEC_violated"], WEC_violated=s["WEC_violated"], SEC_violated=s["SEC_violated"],
               pt_max_over_rho_c=float(np.max(pt_direct) / fam.rho_c), pt_max_expected_over_rho_c=float(B_c / (4 * delta)),
               err_rho=err_rho, err_pt=err_pt, tau_horizon_to_inner=tau_in, tau_schw_horizon_to_center=tau_schw,
               tidal_r_max=float(np.max(np.abs(a_r))), tidal_t_max=float(np.max(np.abs(a_t))),
               tidal_schw_at_Bc=2 * M / B_c**3, shell_limit_S=S_shell, pt_proper_integral=pt_int)
    return out, fam, x, T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = 1.0
    rows = []
    print("Anisotropic curvature transition in the Kantowski-Sachs region (M = 1)")
    for Bc in (0.5, 0.2, 0.05):
        for d in (0.3, 0.1, 0.03, 0.01):
            r, fam, x, T = analyze(M, Bc, d)
            rows.append(r)
            print(f"  B_c/2M={Bc:4g} δ/B_c={d:5g}: K_c={r['K_c']:.3g}, rho_c={r['rho_c']:.4g} (expected {r['rho_c_expected']:.4g}), ell/B_c={r['ell_over_Bc']:.3f}; "
                  f"horizons={r['n_horizons']} R-={r['R_minus']}, κ-={r['kappa_minus']} (dS: {r['kappa_minus_expected_dS']:.3g}); "
                  f"K(0)/K_c={r['K_center_over_Kc']:.3f}, K_max/K_c={r['K_max_over_Kc']:.3f} at R={r['x_K_max']:.3g}; "
                  f"WEC violated={r['WEC_violated']} NEC violated={r['NEC_violated']} min ρ+p⊥={r['min_nec_t']:.3g}; "
                  f"p⊥max/ρ_c={r['pt_max_over_rho_c']:.3g} (expected {r['pt_max_expected_over_rho_c']:.3g}); "
                  f"τ(2M→R-)={r['tau_horizon_to_inner']:.4f} vs Schwarzschild πM={r['tau_schw_horizon_to_center']:.4f}; "
                  f"|a_r|max={r['tidal_r_max']:.3g}, |a⊥|max={r['tidal_t_max']:.3g}; S_shell={r['shell_limit_S']:.3g} vs ∫p⊥dτ={r['pt_proper_integral']:.3g}; "
                  f"outside: ΔT_H={r['T_H_over_schw']-1:.1e}, Δb_c={r['b_c_over_schw']-1:.1e}; err(ρ,p⊥)={r['err_rho']:.1e},{r['err_pt']:.1e}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # plot for B_c/2M = 0.2, δ/B_c = 0.1 and 0.01
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for d, c in ((0.1, "C0"), (0.01, "C3")):
        r, fam, x, T = analyze(M, 0.2, d)
        R = np.asarray(T["R"], float)
        axes[0, 0].plot(R, T["f"], c, label=f"δ/B_c={d}")
        axes[0, 1].plot(R, np.asarray(T["rho8pi"]) / (8 * np.pi), c, label=f"ρ, δ/B_c={d}")
        axes[0, 1].plot(R, np.asarray(T["pt8pi"]) / (8 * np.pi), c, ls="--", label=f"p_⊥, δ/B_c={d}")
        axes[1, 0].plot(R, T["K"], c, label=f"δ/B_c={d}")
        axes[1, 1].plot(R, np.asarray(T["nec_t"]) / (8 * np.pi), c, label=f"ρ+p_⊥, δ/B_c={d}")
        axes[1, 1].plot(R, np.asarray(T["sec"]) / (8 * np.pi), c, ls="--", label=f"ρ+p_R+2p_⊥, δ/B_c={d}")
    schw = am.Family("s", "Schwarzschild", am.F_schwarzschild, dict(m=M))
    Ts = am.tensors(schw, x)
    axes[0, 0].plot(np.asarray(Ts["R"]), Ts["f"], "k:", label="Schwarzschild")
    axes[1, 0].plot(np.asarray(Ts["R"]), Ts["K"], "k:", label="Schwarzschild")
    axes[0, 0].axhline(0, color="gray", lw=0.5)
    axes[0, 0].set(xlim=(0, 2.5), ylim=(-1.5, 1.1), xlabel="R/M", ylabel="f", title="f(R): region f<0 is Kantowski-Sachs; de Sitter core at R<ℓ")
    axes[0, 1].set(xlim=(0, 0.8), xlabel="R/M", ylabel="density / pressure", yscale="symlog", title="Transition layer: ρ ≥ 0, large positive p_⊥")
    axes[1, 0].set(xlim=(0, 2.5), yscale="log", xlabel="R/M", ylabel="K", title="Curvature: threshold K_c at R = B_c, saturation in the core")
    axes[1, 0].axhline(48 * M**2 / (0.4) ** 6, color="gray", ls="--", label="K_c")
    axes[1, 1].set(xlim=(0, 0.8), xlabel="R/M", yscale="symlog", title="Energy conditions: NEC/WEC satisfied, SEC violated in the core")
    axes[1, 1].axhline(0, color="gray", lw=0.5)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Anisotropic curvature transition (B_c = 0.4 M): outside the layer the metric is exactly Schwarzschild")
    fig.tight_layout()
    fig.savefig(OUT / "ks_transition.png", dpi=130)
    plt.close(fig)
    print("\n->", OUT)


if __name__ == "__main__":
    main()
