"""T29: static interior beyond the attractor's landing point (b*, ρ* = 1/(8πb*²), m* = b*/2) from the base-profile EOS: f(b) for b < b*, regularity at the center."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import T29_viscosity as T
bg = T.qb.FamilyBG(T.qb.build_family(T.BASE0, 1.0)); med = T.Medium(bg)
out = []
for bstar in (1.44, 1.52, 1.58, 1.72):
    rho_s = 1 / (8 * np.pi * bstar**2); m_s = bstar / 2
    def rhs(b, y):
        rho, m = y
        return [-2 * rho * (1 + med.w(rho)) / b, 4 * np.pi * b**2 * rho]
    s = solve_ivp(rhs, (bstar, 1e-3), [rho_s, m_s], rtol=1e-10, atol=1e-14, dense_output=True, max_step=1e-3)
    b = np.geomspace(1e-3, bstar * (1 - 1e-6), 20000)
    rho, m = s.sol(b)
    f = 1 - 2 * m / b
    i = np.argmin(f)
    w_s = float(med.w(np.array([rho_s]))[0])
    lam_eff = np.sqrt(3 / (8 * np.pi * rho[0])) / m_s
    print(f"b* = {bstar}: ρ* = {rho_s:.4f}, w(ρ*) = {w_s:+.2f} (σ = {2*(1+w_s):.1f}), f''(b*+) = 16π w ρ* = {16*np.pi*w_s*rho_s:+.3f}; inside: min f = {f[i]:+.4f} at b = {b[i]:.3f}; f(0.5 b*) = {np.interp(0.5*bstar, b, f):+.3f}; "
          f"center: ρ(1e-3)/ρ_c = {rho[0]/med.rho_c:.4f}, m ∝ b³? m(1e-3)/(4π/3 ρ_c 1e-9) = {m[0]/(4*np.pi/3*med.rho_c*1e-9):.3f}; ℓ_eff/m* = {lam_eff:.3f}; number of roots of f inside: {int(np.sum(np.diff(np.sign(f)) != 0))}")
    out.append(dict(bstar=bstar, rho_s=rho_s, w_s=w_s, fmin=float(f[i]), b_fmin=float(b[i]), nroots=int(np.sum(np.diff(np.sign(f)) != 0)), lam_eff=float(lam_eff), rho0_over_rhoc=float(rho[0]/med.rho_c)))
json.dump(out, open(OUT / "T29_interior.json", "w", encoding="utf-8"), indent=1)
