"""Variant A: do post-bounce crossings depend on the initial data. (1) Start from rest with different R0 (E<0, weakly bound
shells at large R0); (2) marginally bound shells E = 0 (for the Hayward form there is no bounce -- asymptotic contraction into
the de Sitter core, cf. Fazzini-Giesel-Rullit 2026). Measure of "how much matter passes through the caustic": m_M(M(r), R_x) for
the crossing shells (weight of the dust component at 1/Y) -- median and maximum. Hayward, ell = 0.271 M_tot.
Run: python src/inhomogeneous_collapse/variant_A.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent; OUT = HERE.parents[1] / "data" / HERE.name; LOGS = HERE.parents[1] / "logs" / HERE.name; LOGS.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(HERE))
import ltb_bounce as lb

ellA = 0.27101577964329177
mm = lb.hayward_mass(ellA); K0 = 24 / ellA**4
LOG = []; rows = []
def say(s):
    print(s, flush=True); LOG.append(s)

def analyse(label, prof, rs, tau, E_override=None):
    sh = [lb.integrate_shell(mm, prof, r, tau, E_override=E_override) for r in rs]
    tb = np.array([s["t_bounce"][0] if len(s["t_bounce"]) else np.nan for s in sh])
    tx = np.array([s["t_cross"][0] if len(s["t_cross"]) else np.nan for s in sh])
    crushed = int(sum(s["crushed"] for s in sh))
    cross = np.nonzero(np.isfinite(tx))[0]
    after = cross[np.isfinite(tb[cross]) & (tx[cross] > tb[cross])]
    before = cross[~(np.isfinite(tb[cross]) & (tx[cross] > tb[cross]))]
    mM = []; Rx = []; rhod_ratio = []
    for i in after:
        s = sh[i]
        sx = lb.integrate_shell(mm, prof, s["r"], np.array([0.0, tx[i] - 1e-9]), E_override=E_override)
        d = mm.all(s["M"], sx["R"][-1]); mM.append(float(d["m_M"])); Rx.append(float(sx["R"][-1]))
    dt = (tx[after] - tb[after]) / ellA
    Kb = []
    for s, t in zip(sh, tb):
        if np.isfinite(t):
            sb = lb.integrate_shell(mm, prof, s["r"], np.array([0.0, t]), E_override=E_override)
            Kb.append(float(lb.fields(mm, prof, s["r"], sb["R"][-1:], sb["V"][-1:], sb["Y"][-1:], sb["W"][-1:])["K"][0]) / K0)
    Rmin = float(np.nanmin([np.nanmin(s["R"]) for s in sh]))
    row = dict(label=label, n=len(rs), n_bounced=int(np.isfinite(tb).sum()), crushed=crushed, n_cross=int(len(cross)), n_after=int(len(after)), n_before=int(len(before)),
               dtau_after_over_ell=dict(min=float(dt.min()), median=float(np.median(dt)), max=float(dt.max())) if len(dt) else None,
               K_b_over_K0=dict(min=min(Kb), max=max(Kb)) if Kb else None, mM_at_cross=dict(median=float(np.median(mM)), max=float(max(mM))) if mM else None,
               Rx_over_ell=dict(min=float(min(Rx)) / ellA, max=float(max(Rx)) / ellA) if Rx else None, R_min_over_ell=Rmin / ellA,
               tau_b=dict(min=float(np.nanmin(tb)), max=float(np.nanmax(tb))) if np.isfinite(tb).any() else None)
    rows.append(row)
    say(f"{label}: shells {len(rs)}, bounced {row['n_bounced']}, crushed {crushed}; crossings {len(cross)} (after bounce {len(after)}, other {len(before)}); "
        f"dtau/ell {row['dtau_after_over_ell']}; K_b/K0 {row['K_b_over_K0']}; m_M at caustic {row['mM_at_cross']}; R_x/ell {row['Rx_over_ell']}; min R/ell = {Rmin/ellA:.3g}; tau_b {row['tau_b']}")

say("Variant A. Hayward ell = 0.271 M_tot, profile eps = +0.1 (dense center), 80 interior shells.")
for r0 in (20.0, 100.0, 500.0):
    prof = lb.MassProfile(0.1, r0)
    tau_c = np.pi * r0**1.5 / 2**1.5
    tau = np.linspace(0, 1.3 * tau_c, 2001)
    rs = r0 * np.linspace(0, 1, 81)[1:-1]
    analyse(f"from rest, R0 = {r0:g} M (E = -m/r; tau_c = {tau_c:.0f})", prof, rs, tau)
say("\nMarginally bound shells E = 0 (start with V = -sqrt(2m/r) at R = r), r0 = 20:")
for eps in (0.1, -0.1, 0.0):
    prof = lb.MassProfile(eps, 20.0)
    tau = np.linspace(0, 200, 2001)
    rs = 20.0 * np.linspace(0, 1, 81)[1:-1]
    analyse(f"E = 0, eps = {eps:+.1f}", prof, rs, tau, E_override=0.0)
say("\nWeakly unbound E = +0.01 (r0 = 20), eps = +0.1:")
prof = lb.MassProfile(0.1, 20.0)
analyse("E = +0.01, eps = +0.1", prof, 20.0 * np.linspace(0, 1, 81)[1:-1], np.linspace(0, 200, 2001), E_override=0.01)
(OUT / "variant_A.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
(LOGS / "variant_A_log.txt").write_text("\n".join(LOG), encoding="utf-8")
