"""Estimate of the semiclassical zone at the scenario's degenerate inner horizon, following McMaken 2023 (SOURCES item 74).
Unruh vacuum: <T_uu> = (1/(192 pi^2 r^2)) (kappa' Delta - kappa^2 + kappa(r+)^2) (their (58a), hbar = 1, G = c = 1); at r = r-, kappa(r-) = 0:
<T_uu>(r-) = kappa_+^2/(192 pi^2 r-^2). Freely falling observer with energy E: u_dot ~ 2E/f, rho_obs = <T_uu> u_dot^2 = 4E^2 kappa_+^2/(192 pi^2 r-^2 f^2).
Backreaction zone: rho_obs = rho_c = 3/(8 pi ell^2) (restoring hbar = l_P^2): f_crit = E kappa_+ l_P ell sqrt(32 pi/(576 pi^2))/r-;
near the triple root f ~ c (r - r-)^3, c = |f'''(r-)|/6  =>  dr = (f_crit/c)^{1/3}.
Run: python src/ori_triple_root/semiclassical_zone.py
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "inhomogeneous_collapse"))
import ltb_bounce as lb  # noqa: E402

scen = lb.ScenarioMass()
hz = lb.static_horizons(scen, 1.0)
r_minus, kap_minus = hz[0]; r_plus, kap_plus = hz[-1]
ell = scen.ell
# f''' at the triple root: approximation f = c (r - r-)^3 from points to the right
f = lambda r: 1 - 2 * float(scen.all(1.0, r)["m"]) / r
dr = np.array([0.02, 0.04, 0.06, 0.08])
c_est = np.array([-f(r_minus + d) / d**3 for d in dr])
c = float(np.median(c_est))
print(f"Scenario: r- = {r_minus:.5f}, kappa- = {kap_minus:.1e}, r+ = {r_plus:.5f}, kappa+ = {kap_plus:.5f}, ell = {ell:.4f}; f ~ -c (r - r-)^3 to the right of r-, c = {c:.3f} (from dr = {dr.tolist()}: {np.round(c_est, 3).tolist()})")
G, cc, Msun, hbar = 6.67430e-11, 2.99792458e8, 1.98847e30, 1.054571817e-34
l_P = np.sqrt(hbar * G / cc**3)
E = 1.0
pref = np.sqrt(32 * np.pi / (576 * np.pi**2))
print(f"<T_uu>(r-) = kappa_+^2/(192 pi^2 r-^2) = {kap_plus**2/(192*np.pi**2*r_minus**2):.4e} (units M^-4, hbar = 1); rho_c = {3/(8*np.pi*ell**2):.4f} M^-2")
for M_s in (3.0, 10.0, 4e6):
    M = M_s * Msun * G / cc**2   # m
    f_crit = E * kap_plus * (l_P / M) * ell * pref / r_minus   # dimensionless (kappa_+, ell, r- in units of M)
    d_over_M = (f_crit / c) ** (1 / 3)
    d = d_over_M * M
    tau = d / cc  # proper crossing time at dr/dtau ~ E = 1
    print(f"M = {M_s:g} M_sun (M = {M:.3e} m): f_crit = {f_crit:.3e}; zone |r - r-| < {d:.3e} m = {d_over_M:.2e} M; crossing time ~ {tau:.1e} s; "
          f"for comparison, the Planck length {l_P:.2e} m, (l_P M)^(1/2) = {np.sqrt(l_P*M):.2e} m")
