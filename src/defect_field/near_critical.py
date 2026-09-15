"""Supplement: nontrivial supercritical solutions near alpha = 1/2 (x_h -> inf as alpha -> 1/2+) -- more initial guesses."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import defect_monopole as dm
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = []
for alpha in (0.55, 0.6, 0.7):
    best = None
    for xh in (4.0, 5.0, 6.5, 8.0, 10.0, 13.0):
        for fh in (0.9, 0.95, 0.98):
            for a in (0.38, 0.42):
                try:
                    p, r, ier = dm.match_super(alpha, [a, xh, fh])
                except Exception:
                    continue
                if r < 1e-7 and p[0] > 1e-3 and (best is None or r < best[1]):
                    best = (p, r)
    if best is None:
        print(f"alpha = {alpha}: nontrivial matching not found"); rows.append(dict(alpha=alpha, found=False)); continue
    a, xh, fh = best[0]
    sig_h, hd = dm.sigma_at_horizon(xh, fh, alpha)
    print(f"alpha = {alpha}: a = {a:.5f}, x_h = {xh:.4f}, f_h = {fh:.4f}, f'_h = {hd['fp_h']:+.4f}, kappa_h = {hd['kappa_h']:+.5f}, 8 pi r^2 rho|_h = {hd['ind']:.4f}, sigma_h = {sig_h:.4f}, residual {best[1]:.1e}")
    rows.append(dict(alpha=alpha, found=True, a=a, x_h=xh, f_h=fh, kappa_h=hd["kappa_h"], ind_h=hd["ind"], sigma_h=sig_h))
json.dump(rows, open(dm.OUT / "near_critical.json", "w", encoding="utf-8"), indent=1, default=float)
