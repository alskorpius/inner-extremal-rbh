"""Postprocessing: exact bounce/crossing times from events (re-integrating the interior shells), crossing statistics
relative to the bounce, pre-bounce NEC and K (region tau < min(tau_b, tau_x)), caustic strength (|Y| at which
K doubles relative to 24/ell^4, on a fine grid between the bounce and the crossing), absolute-value check of the scenario's m_M.
Run: python src/inhomogeneous_collapse/postprocess.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


ellA = 0.27101577964329177
scen = lb.ScenarioMass()
models = {"hayward_ellA": lb.hayward_mass(ellA), "hayward_ell0.05": lb.hayward_mass(0.05), "scenario": scen, "local_lamA": lb.local_mass(ellA)}
cases = {"hayward_ellA": [0.0, 0.01, 0.1, 0.3, -0.01, -0.1, -0.3], "hayward_ell0.05": [0.0, 0.1, -0.1], "scenario": [0.0, 0.1, -0.1], "local_lamA": [0.0, 0.1]}
rows = []
for name, mm in models.items():
    ell = 0.05 if name == "hayward_ell0.05" else ellA
    K0 = 24 / ell**4
    for eps in cases[name]:
        prof = lb.MassProfile(eps, 20.0)
        step = 4 if name == "scenario" else 1
        rs = 20.0 * np.linspace(0, 1, 161)[1:-1:step]          # interior shells (excluding the surface)
        tau = np.linspace(0, 230, 461)
        sh = [lb.integrate_shell(mm, prof, r, tau) for r in rs]
        tb = np.array([s["t_bounce"][0] if len(s["t_bounce"]) else np.nan for s in sh])
        tx = np.array([s["t_cross"][0] if len(s["t_cross"]) else np.nan for s in sh])
        cross = np.nonzero(np.isfinite(tx))[0]
        after = cross[tx[cross] > tb[cross]]
        before = cross[tx[cross] <= tb[cross]]
        dt = (tx[after] - tb[after]) / ell
        # pre-bounce / pre-crossing region on a coarse grid
        pre_nec, pre_K, pre_rho = np.inf, 0.0, np.inf
        for s in sh:
            lim = min(s["t_bounce"][0] if len(s["t_bounce"]) else np.inf, s["t_cross"][0] if len(s["t_cross"]) else np.inf)
            k = s["t"] < lim
            if not k.any():
                continue
            f = lb.fields(mm, prof, s["r"], s["R"][k], s["V"][k], s["Y"][k], s["W"][k])
            pre_nec = min(pre_nec, float(np.nanmin(f["nec_perp"]))); pre_K = max(pre_K, float(np.nanmax(f["K"]))); pre_rho = min(pre_rho, float(np.nanmin(f["rho"])))
        # K at the bounce
        Kb = []
        for s, t in zip(sh, tb):
            if np.isfinite(t):
                sb = lb.integrate_shell(mm, prof, s["r"], np.array([0.0, t]))
                Kb.append(float(lb.fields(mm, prof, s["r"], sb["R"][-1:], sb["V"][-1:], sb["Y"][-1:], sb["W"][-1:])["K"][0]))
        # caustic strength at the first shell that crosses after the bounce
        fa = None
        good = after[(tx[after] - tb[after]) > 1e-6]
        if len(good):
            i = good[np.argmin(tx[good])]
            r = rs[i]
            tf = np.linspace(tb[i] + 1e-9, tx[i] - 1e-9, 4001)
            sf = lb.integrate_shell(mm, prof, r, np.concatenate([[0.0], tf]))
            f = lb.fields(mm, prof, r, sf["R"][1:], sf["V"][1:], sf["Y"][1:], sf["W"][1:])
            Kf, Yf = f["K"], np.abs(sf["Y"][1:])
            big = np.nonzero(Kf > 2 * K0)[0]
            fa = dict(r=float(r), M_r=float(prof.M(r)), tau_b=float(tb[i]), tau_x=float(tx[i]), R_x=float(sf["R"][-1]), R_x_over_ell=float(sf["R"][-1] / ell),
                      Y_at_K_doubling=float(Yf[big[0]]) if len(big) else None, Y_min_on_grid=float(Yf.min()), K_max_on_grid_over_K0=float(np.nanmax(Kf) / K0),
                      m_M_at_x=float(mm.all(prof.M(r), sf["R"][-1])["m_M"]), nec_min_between=float(np.nanmin(f["nec_perp"])),
                      rho_d_over_rho_v_at_Y0p1=float((f["rho_d"] / f["rho_v"])[np.argmin(np.abs(Yf - 0.1))]))
        row = dict(label=f"{name}_eps{eps}", ell=ell, n_inner=len(rs), n_cross=int(len(cross)), n_after=int(len(after)), n_before=int(len(before)),
                   dtau_after_over_ell=dict(min=float(dt.min()), median=float(np.median(dt)), max=float(dt.max())) if len(dt) else None,
                   tau_b=dict(min=float(np.nanmin(tb)), max=float(np.nanmax(tb))), K_bounce_over_K0=dict(min=min(Kb) / K0, max=max(Kb) / K0) if Kb else None,
                   pre=dict(min_nec_perp=pre_nec, max_K_over_K0=pre_K / K0, min_rho=pre_rho), first_after=fa)
        rows.append(row)
        say(f"{row['label']:26s} shells {len(rs)}: crossings {len(cross)} (after bounce {len(after)}, before {len(before)}); dtau/ell after bounce: {row['dtau_after_over_ell']}; "
            f"tau_b {row['tau_b']['min']:.3f}...{row['tau_b']['max']:.3f}; K_b/K0 {row['K_bounce_over_K0']}; before bounce/crossing: min(rho+p_perp)={pre_nec:.3g}, K_max/K0={pre_K/K0:.3g}, min rho={pre_rho:.3g}")
        if fa:
            say(f"    first after bounce: r={fa['r']:.3f} (M(r)={fa['M_r']:.3f}), R_x/ell={fa['R_x_over_ell']:.3g}, |Y| at K>2K0: {fa['Y_at_K_doubling']} (min |Y| on grid {fa['Y_min_on_grid']:.2e}, K_max/K0 on grid {fa['K_max_on_grid_over_K0']:.3g}); "
                f"m_M at the crossing point {fa['m_M_at_x']:.3e}; rho_d/rho_v at |Y|=0.1: {fa['rho_d_over_rho_v_at_Y0p1']:.3e}; min(rho+p_perp) between bounce and crossing {fa['nec_min_between']:.3g}")

Rt = np.array([0.01, 0.1, 0.5, 1.0, 2.0]); h = 1e-5
d = scen.all(1.0, Rt)
fd_M = (scen.all(1.0 + h, Rt)["m"] - scen.all(1.0 - h, Rt)["m"]) / (2 * h)
say(f"scenario m_M at R={Rt.tolist()}: analytic {np.array2string(d['m_M'], precision=4)}, finite differences {np.array2string(fd_M, precision=4)}, absolute difference {np.array2string(np.abs(fd_M - d['m_M']), precision=2)} (in the core m_M -> 0: de Sitter does not depend on M)")
rows.append(dict(label="scenario_mM_check", R=Rt.tolist(), m_M=d["m_M"].tolist(), fd=fd_M.tolist()))
(OUT / "postprocess.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
(LOGS / "postprocess_log.txt").write_text("\n".join(LOG), encoding="utf-8")
