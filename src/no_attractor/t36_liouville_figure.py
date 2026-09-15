"""T36 figure: the theorem made visible as area preservation.

The theorem says a degenerate horizon can never be an attractor because the
T-region flow preserves phase-space volume.  An attractor is a blob of initial
conditions collapsing onto a point; Liouville forbids exactly that.  So the
honest illustration is a blob carried by the flow, next to a published case
where a blob really does collapse.

Panel (a)  the Kantowski-Sachs T-region flow b'' = -f'(b)/2 of this project's
           profile.  A ring of initial conditions is carried by the flow; the
           enclosed area is measured at each snapshot.  The area is invariant:
           this IS the theorem.
Panel (b)  the comparison is NOT an invented friction term.  It is the
           cosmological case of Remmen and Carroll, Phys. Rev. D 88, 083518
           (2013), arXiv:1309.2611: phi'' + 3 H phi' + V'(phi) = 0 for
           V = m^2 phi^2 / 2, whose phase-space divergence is -3H.  There the
           blob does collapse onto the slow-roll attractor, which is why that
           paper needs a different conserved measure to recover Liouville.
           Here there is no friction term to begin with.
Panels (c) fixed-point type as f''(b*) changes sign: saddle, centre, and the
           triple root between them.  Auxiliary, and drawn small on purpose.

SCOPE, and it belongs in the caption as well as here: the area-preserving
statement holds for the velocity-INDEPENDENT case.  Once p_perp depends on
b' the flow is no longer Hamiltonian - its divergence is
16 pi b v dp_perp/dq + v/b, which vanishes only on v = 0 - and a blob may
legitimately change area.  What survives there is the weaker, local statement:
the Jacobian trace at the fixed point is zero, so the horizon is never
asymptotically stable.  Drawing a blob for that case would illustrate a claim
that was not proved.

Run: python src/no_attractor/t36_liouville_figure.py
Needs data/no_attractor/t36_profile_cache.npz (build it with t36_profile.py) and
data/no_attractor/t36_core.json (for the measured area error).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
sys.path.insert(0, str(HERE))

import t36_profile as t36p

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def polygon_area(x, y):
    """Shoelace area of the closed curve traced by the blob boundary."""
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ----------------------------------------------------------------- panel (a)
def ks_blob(P, Rm, Rp, n_ring=4000, tau_end=2.1, n_snap=4):
    """Integration time is chosen for RESOLUTION, not for want of a longer run.

    The flow is exponentially shearing: the monodromy determinant stays 1 to
    12 digits out to tau = 18, but the singular-value ratio reaches 3e6 there,
    and a polygon of finitely many boundary points stops resolving the filament
    long before that - the measured shoelace area then degrades into
    discretisation noise even though the true area is exactly conserved.  At
    tau = 2.1 the stretch is about 8: deformation is plainly visible and the
    boundary is still resolved, so the area measured from the drawn curve is
    meaningful.  The determinant at large tau is reported separately from
    t36_core.json.
    """
    b0 = 0.5 * (Rm + Rp)
    v0 = float(np.sqrt(max(-P.f(np.array([b0]))[0], 0.0)))
    rb, rv = 0.055 * (Rp - Rm), 0.055 * max(v0, 0.1)
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring0 = np.stack([b0 + rb * np.cos(th), v0 + rv * np.sin(th)])

    def rhs(tau, y):
        m = y.size // 2
        bb = np.clip(y[:m], 1e-6, None)
        fp = P.parts(bb)[1]
        return np.concatenate([-y[m:], fp / 2])

    sol = solve_ivp(rhs, (0, tau_end), ring0.ravel(), rtol=1e-11, atol=1e-13,
                    dense_output=True, max_step=0.05)
    taus = np.linspace(0, tau_end, n_snap)
    snaps, areas = [], []
    for t in taus:
        y = sol.sol(t)
        m = y.size // 2
        bb, vv = y[:m], y[m:]
        snaps.append((bb, vv))
        areas.append(polygon_area(bb, vv))
    return b0, v0, taus, snaps, np.array(areas)


# ----------------------------------------------------------------- panel (b)
def rc_blob(m=1.0, n_ring=4000, t_end=0.5, n_snap=4):
    """Remmen-Carroll: phi'' + 3 H phi' + m^2 phi = 0, 3 H^2 = phi'^2/2 + m^2 phi^2/2.

    Reduced Planck units.  Divergence of the (phi, phi') flow is -3H < 0, so the
    blob contracts - the behaviour their paper has to reconcile with Liouville.
    """
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring0 = np.stack([5.0 + 0.55 * np.cos(th), 0.6 * np.sin(th)])

    def rhs(t, y):
        k = y.size // 2
        phi, dphi = y[:k], y[k:]
        H = np.sqrt(np.maximum((0.5 * dphi**2 + 0.5 * m**2 * phi**2) / 3.0, 0.0))
        return np.concatenate([dphi, -3 * H * dphi - m**2 * phi])

    sol = solve_ivp(rhs, (0, t_end), ring0.ravel(), rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.05)
    ts = np.linspace(0, t_end, n_snap)
    snaps, areas = [], []
    for t in ts:
        y = sol.sol(t)
        k = y.size // 2
        snaps.append((y[:k], y[k:]))
        areas.append(polygon_area(y[:k], y[k:]))
    return ts, snaps, np.array(areas)


# ----------------------------------------------------------------- panels (c)
def model_f(b, kind):
    """Local models of f near a root, differing only in the sign of f''."""
    if kind == "saddle":            # double root, f'' < 0
        return -(b - 1.0) ** 2
    if kind == "centre":            # double root, f'' > 0
        return (b - 1.0) ** 2
    return -(b - 1.0) ** 3          # triple root, f'' = 0


def mini_portrait(ax, kind, title):
    """Orbits as level sets of H = v^2/2 + f(b)/2.

    These flows are velocity-independent, so they are exactly Hamiltonian and
    the level sets of H *are* the orbits - drawing them is exact. An earlier
    version used streamplot, whose coarse integration made the triple-root
    panel look as though it had a closed orbit around the fixed point. It has
    none: there v' = -3(b-1)^2/2 <= 0, so v decreases monotonically and no
    trajectory can return (checked numerically).
    """
    b = np.linspace(0.55, 1.45, 400)
    v = np.linspace(-0.45, 0.45, 400)
    B, V = np.meshgrid(b, v)
    H = V**2 / 2 + model_f(B, kind) / 2
    ax.contour(B, V, H, levels=14, colors="0.55", linewidths=0.5)
    # direction of travel: b' = -v, so motion is leftward above the axis
    for vv in (0.30, -0.30):
        ax.annotate("", xy=(1.0 - 0.16 * np.sign(vv), vv), xytext=(1.0, vv),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35"))
    ax.plot([1.0], [0.0], "ko", ms=3.5)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.set_xlim(b[0], b[-1]); ax.set_ylim(v[0], v[-1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def main():
    say("T36 figure: phase-space area under the T-region flow")
    P = t36p.Profile(0.0, grid=t36p.load_grid())
    roots = P.roots()
    Rm, Rp = roots[0], roots[-1]
    say(f"  profile horizons: R- = {Rm:.6f}, R+ = {Rp:.6f}")

    core = json.loads((OUT / "t36_core.json").read_text(encoding="utf-8"))
    det_err = core["B"]["B2"]["max_det_error"]
    say(f"  measured area error from t36_core.json: max|det J - 1| = {det_err:.2e}")

    b0, v0, taus, snaps, areas = ks_blob(P, Rm, Rp)
    rel = np.abs(areas / areas[0] - 1.0)
    say(f"  KS blob: area {areas[0]:.6e} -> {areas[-1]:.6e}, "
        f"max relative change {rel.max():.2e}")

    ts, rc_snaps, rc_areas = rc_blob()
    say(f"  Remmen-Carroll blob: area {rc_areas[0]:.4e} -> {rc_areas[-1]:.4e}, "
        f"ratio {rc_areas[-1] / rc_areas[0]:.4f}")

    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 1.0], hspace=0.50, wspace=0.45)
    axa = fig.add_subplot(gs[0, :2])
    axb = fig.add_subplot(gs[0, 2])

    cmap = plt.get_cmap("viridis")
    for i, ((bb, vv), t) in enumerate(zip(snaps, taus)):
        c = cmap(i / max(len(snaps) - 1, 1))
        axa.fill(bb, vv, color=c, alpha=0.30, lw=0)
        axa.plot(np.append(bb, bb[0]), np.append(vv, vv[0]), color=c, lw=1.0)
        # Label each snapshot where it sits. A legend would cover the first
        # ring, and rounding the four times to whole M printed "1 M" twice.
        axa.annotate(rf"$\tau = {t:.1f}\,M$",
                     xy=(bb.mean(), vv.max()), xytext=(0, 5),
                     textcoords="offset points", ha="center",
                     fontsize=6.5, color=c)
    axa.set(xlabel=r"$b$  [$M$]", ylabel=r"$v = -\dot b$")
    axa.margins(x=0.10, y=0.18)
    axa.text(0.02, 0.94, "(a)  Kantowski-Sachs T-region: area conserved",
             transform=axa.transAxes, fontsize=7.5)
    axa.text(0.02, 0.05,
             "velocity-independent law:\n"
             r"flow is Hamiltonian, $\nabla\!\cdot\!F \equiv 0$",
             transform=axa.transAxes, fontsize=6.2, color="0.3")
    axa.text(0.98, 0.05, rf"$\max|\det J - 1| = {det_err:.1e}$",
             transform=axa.transAxes, fontsize=6.8, ha="right",
             bbox=dict(fc="w", ec="0.7", lw=0.5, pad=2.0))

    for i, ((pp, dpp), t) in enumerate(zip(rc_snaps, ts)):
        c = cmap(i / max(len(rc_snaps) - 1, 1))
        axb.fill(pp, dpp, color=c, alpha=0.30, lw=0)
        axb.plot(np.append(pp, pp[0]), np.append(dpp, dpp[0]), color=c, lw=1.0)
    axb.set(xlabel=r"$\phi$  [$M_{\rm Pl}$]", ylabel=r"$\dot\phi$")
    axb.yaxis.set_label_coords(-0.26, 0.5)
    axb.text(0.03, 0.94, "(b)  Remmen-Carroll", transform=axb.transAxes, fontsize=7.5)
    axb.text(0.03, 0.05,
             rf"$\nabla\!\cdot\!F = -3H$" "\n"
             rf"area $\times {rc_areas[-1] / rc_areas[0]:.2f}$",
             transform=axb.transAxes, fontsize=6.2, color="0.3")

    for j, (kind, title) in enumerate(
            [("saddle", r"$f^{\prime\prime}(b_*) < 0$: saddle"),
             ("triple", r"$f^{\prime\prime}(b_*) = 0$: triple root"),
             ("centre", r"$f^{\prime\prime}(b_*) > 0$: centre")]):
        mini_portrait(fig.add_subplot(gs[1, j]), kind, title)
    fig.text(0.008, 0.28, "(c)", fontsize=7.5)

    fig.savefig(FIG / "fig8_liouville.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig8_liouville.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    say(f"  -> {FIG / 't36_liouville.pdf'}")

    caption = (
        "Phase-space area under the interior flow. (a) A ring of initial conditions in the "
        "Kantowski-Sachs T-region, carried by b'' = -f'(b)/2 for the profile of this work, shown at "
        f"four times; the enclosed area changes by at most {rel.max():.0e} over the integration, and the "
        f"monodromy determinant of the variational equation satisfies max|det J - 1| = {det_err:.1e}. An "
        "attractor would be this blob collapsing onto a point, which the vanishing divergence forbids. "
        "(b) The same experiment for the cosmological case of Remmen and Carroll (Phys. Rev. D 88, "
        "083518 (2013)), phi'' + 3H phi' + m^2 phi = 0, whose divergence is -3H: there the blob does "
        f"contract, by a factor {rc_areas[-1] / rc_areas[0]:.3f} over the interval shown, which is why "
        "recovering Liouville in that setting requires a different, diverging measure. The interior flow "
        "carries no such friction term. (c) Type of the fixed point as f''(b_*) changes sign: the "
        "degenerate horizon is the boundary between a saddle and a centre, never asymptotically stable. "
        "IMPORTANT: panel (a) and the area statement apply to velocity-independent constitutive laws, "
        "for which the flow is Hamiltonian. If the transverse pressure depends on b' the flow is not "
        "Hamiltonian and the area is not conserved; what holds there is the weaker local statement that "
        "the Jacobian trace vanishes at the fixed point, so the degenerate horizon is never "
        "asymptotically stable."
    )
    (FIG / "fig8_liouville_caption.txt").write_text(caption + "\n", encoding="utf-8")
    say("  -> caption written")
    (LOGS / "t36_liouville_figure_log.txt").write_text("\n".join(LOG) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
