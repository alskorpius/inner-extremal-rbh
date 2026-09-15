"""T18, item 2 (refinement): the criterion "the layer does not create curvature above its own threshold" (experiments/ks_curvature_transition):
curvature in the truncation layer K_cut = max K on [R_c e^{-3 w3}, R_c e^{+3 w3}] (R_c = u3 R1 — the truncation center) against the background curvature
that "triggers" the transition, K_trig = 48 M^2/R_c^6 (the Schwarzschild value at the truncation radius). Same scan as in T18_halo.py (B);
minimum halo at K_cut/K_trig <= 2 and <= 10. Run: python src/verification_01/T18_trigger.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
sys.path.insert(0, str(HERE))
import T18_halo as th  # noqa: E402
import qnm_band as qb  # noqa: E402

rows = []
print("s_end, w3, σ_min | λ | halo % | R_c = u3 R1 | R_c/R+ | K_cut | K_trig = 48/R_c^6 | K_cut/K_trig | K_cut/K(0)", flush=True)
for s_end in (24.0, 100.0, 500.0):
    for w3 in (0.1, 0.05, 0.02, 0.01):
        for smin in (1.0, 0.5, 0.2, 0.0):
            base = dict(th.BASE0, s_end=s_end, w3=w3)
            fam = qb.build_family(base, smin)
            if fam is None:
                continue
            bg = qb.FamilyBG(fam)
            lam = float(np.sqrt(3 / (8 * np.pi * fam.rho_c)))
            Rc = base["u3"] * fam.R1
            R = np.geomspace(Rc * np.exp(-3 * w3), Rc * np.exp(3 * w3), 4000)
            m, f, fp, fpp, K = th.geometry(bg, R)
            Kcut = float(K.max()); Ktrig = 48 / Rc**6; K0 = 24 / lam**4
            halo = 1 - float(bg.fields(np.array([bg.rh]))["m"][0])
            rows.append(dict(s_end=s_end, w3=w3, sigma_min=smin, lam=lam, halo=halo, Rc=Rc, Rc_over_Rp=Rc / bg.rh, K_cut=Kcut, K_trig=Ktrig, ratio_trig=Kcut / Ktrig, ratio_K0=Kcut / K0))
            print(f"{s_end:>5}, {w3:<5}, {smin} | {lam:.3f} | {100*halo:.3f} | {Rc:.3f} | {Rc/bg.rh:.2f} | {Kcut:.3g} | {Ktrig:.3g} | {Kcut/Ktrig:.1f} | {Kcut/K0:.4f}", flush=True)
for lim in (2.0, 5.0, 10.0, 30.0):
    ok = [r for r in rows if r["ratio_trig"] <= lim]
    if ok:
        b = min(ok, key=lambda r: r["halo"])
        print(f"K_cut/K_trig <= {lim}: minimum halo {100*b['halo']:.3f} % (s_end={b['s_end']}, w3={b['w3']}, σ_min={b['sigma_min']}, λ={b['lam']:.3f}); total configurations {len(ok)}", flush=True)
    else:
        print(f"K_cut/K_trig <= {lim}: no configurations", flush=True)
base_row = [r for r in rows if r["s_end"] == 24.0 and r["w3"] == 0.1 and r["sigma_min"] == 1.0][0]
print(f"base: K_cut/K_trig = {base_row['ratio_trig']:.1f}, halo {100*base_row['halo']:.2f} %")
json.dump(rows, open(OUT / "T18_trigger.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
