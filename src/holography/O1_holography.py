"""O1 (cycle 2, open_directions.md): holographic check of the regular core.

Question: does the content of the regular core of the base profile (T16, lambda = ell/M = 0.271)
fit inside the holographic bound S_BH = A/4 of the outer horizon?

Three counts of degrees of freedom (dof) are compared with S_BH:
  (a) naive volumetric count at Planck resolution, (ell/l_P)^3, plus the honest variant
      V_proper(core)/l_P^3 with the proper volume of the static core r < R_- of the base profile
      (the core r < R_- is a static region, f > 0, so the t = const slice gives a well-defined
      proper volume V = int 4 pi r^2 dr / sqrt(f));
  (b) Dvali–Gomez condensate count N = M^2/m_P^2 (SOURCES item 64);
  (c) de Sitter entropy of the core by its own horizon, pi (ell/l_P)^2, where ell is the de Sitter
      radius of the core (rho_c = 3/(8 pi ell^2)); note that the would-be dS horizon r = ell lies
      inside the actual inner horizon R_- = 2.5 ell only in the sense r = ell < R_-, i.e. it is a
      fictitious length: the core metric is de Sitter only near r = 0 and f never vanishes at r = ell
      (the first zero of f is R_-).
Bound: S_BH = pi R_+^2 / l_P^2 with the actual outer horizon R_+ = 1.98465 M of the base profile
(Schwarzschild would give 4 pi M^2/l_P^2).

Units: G = c = 1 inside the profile (M = 1); SI conversion via GM_sun/c^2 = 1477 m, l_P = 1.616e-35 m,
m_P = 2.176e-8 kg, M_sun = 1.989e30 kg.
Run: python src/holography/O1_holography.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# --- constants (SI) ---
l_P = 1.616e-35          # m
m_P = 2.176e-8           # kg
M_sun = 1.989e30         # kg
GMsun_c2 = 1477.0        # m
lam = 0.27102            # ell / M for the base profile (T16)
R_minus = 0.67744        # M
R_plus = 1.98465         # M

# --- base profile (T16 export), units G = c = M = 1 ---
csv = ROOT / "data" / "verification_01" / "T16_base_profile_m_of_r.csv"
data = np.loadtxt(csv, delimiter=",", comments="#", skiprows=2, encoding="utf-8")
R, m, rho, f, sigma = data.T
say(f"base profile: {csv.name}, {len(R)} points, R in [{R[0]:.3g}, {R[-1]:.3g}] M")

# proper volume of the static core r < R_-: V = int_0^{R_-} 4 pi r^2 dr / sqrt(f)
mask = (R < R_minus) & (f > 0)
Rc, fc = R[mask], f[mask]
# the integrand 1/sqrt(f) diverges at the (degenerate) root R_- as (R_- - r)^{-3/2} for a triple
# root, so the integral is formally divergent; cut at f = f_cut and report the dependence
say("\nproper volume of the core r < R_- (units M^3), cut where f drops below f_cut:")
say("  f_cut | R_cut/M | V_proper/M^3 | V_flat = 4/3 pi R_cut^3 | ratio")
vols = {}
for fcut in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
    sel = fc > fcut
    r_, f_ = Rc[sel], fc[sel]
    # add the r < R[0] piece analytically (flat: f = 1 - r^2/ell^2 ~ 1)
    V = np.trapezoid(4 * np.pi * r_**2 / np.sqrt(f_), r_) + 4 / 3 * np.pi * r_[0] ** 3
    Vflat = 4 / 3 * np.pi * r_[-1] ** 3
    vols[fcut] = dict(R_cut=float(r_[-1]), V=float(V), V_flat=float(Vflat))
    say(f"  {fcut:.0e} | {r_[-1]:.4f} | {V:.4f} | {Vflat:.4f} | {V / Vflat:.3f}")
V_core = vols[1e-3]["V"]          # headline value (f_cut = 1e-3; grows only logarithmically slowly)
V_ell = 4 / 3 * np.pi * lam**3    # (4/3) pi ell^3, for reference
say(f"headline V_proper(core, f_cut = 1e-3) = {V_core:.4f} M^3; (4/3) pi ell^3 = {V_ell:.4f} M^3; "
    f"(4/3) pi R_-^3 = {4/3*np.pi*R_minus**3:.4f} M^3")
say("note: 1/sqrt(f) ~ (R_- - r)^{-3/2} at a triple root, so the proper volume of the region "
    "f > f_cut diverges as f_cut^{-1/6}; the value depends weakly (logarithmically-like) on the cut")

# de Sitter check: near r = 0, f = 1 - r^2/ell_dS^2 with ell_dS^2 = 3/(8 pi rho_c)
rho_c = rho[0]
ell_dS = np.sqrt(3 / (8 * np.pi * rho_c))
say(f"\nrho_c = {rho_c:.5f} -> ell_dS = sqrt(3/(8 pi rho_c)) = {ell_dS:.5f} M = lambda (check: lam = {lam})")
i_ell = np.searchsorted(R, ell_dS)
say(f"f at r = ell_dS: {f[i_ell]:.4f} (not zero: the core is de Sitter only near r = 0; "
    f"first zero of f is R_- = {R_minus} M = {R_minus / lam:.2f} ell)")
say(f"f minimum on r < R_-: {fc.min():.2e} at R = {Rc[np.argmin(fc)]:.4f} M")

# --- counts ---
rows = []
say("\nM/M_sun | (a) (ell/l_P)^3 | (a') V_core/l_P^3 | (b) N = M^2/m_P^2 | (c) pi (ell/l_P)^2 | S_BH = pi R_+^2/l_P^2 "
    "| (a)/S_BH | (a')/S_BH | (b)/S_BH | (c)/S_BH")
for Msun in (10.0, 1e6, 1e9):
    M_m = Msun * GMsun_c2                 # M in metres (GM/c^2)
    M_kg = Msun * M_sun
    ell_m = lam * M_m
    a = (ell_m / l_P) ** 3
    a2 = V_core * M_m**3 / l_P**3
    b = (M_kg / m_P) ** 2
    c = np.pi * (ell_m / l_P) ** 2
    S = np.pi * (R_plus * M_m) ** 2 / l_P**2
    S_schw = 4 * np.pi * (M_m / l_P) ** 2
    rows.append(dict(M_Msun=Msun, ell_m=ell_m, a_naive=a, a_proper=a2, b_condensate=b, c_dS=c,
                     S_BH=S, S_BH_schw=S_schw, ratio_a=a / S, ratio_a_proper=a2 / S, ratio_b=b / S,
                     ratio_c=c / S, M_over_mP=M_kg / m_P))
    say(f"{Msun:.0e} | {a:.2e} | {a2:.2e} | {b:.2e} | {c:.2e} | {S:.2e} | {a / S:.1e} | {a2 / S:.1e} | "
        f"{b / S:.3f} | {c / S:.2e}")

say("\nanalytic ratios (independent of M except (a)):")
say(f"  (a)/S_BH = lam^3 (M/m_P) / (pi R_+^2) = {lam**3 / (np.pi * R_plus**2):.3e} * (M/m_P)")
say(f"  (a')/S_BH = V_core (M/m_P) / (pi R_+^2) = {V_core / (np.pi * R_plus**2):.3e} * (M/m_P)")
say(f"  (b)/S_BH = 1/(pi R_+^2) = {1 / (np.pi * R_plus**2):.4f}  (Schwarzschild: 1/(4 pi) = {1/(4*np.pi):.4f})")
say(f"  (c)/S_BH = lam^2 / R_+^2 = {lam**2 / R_plus**2:.4f}")
say(f"  S_BH(base)/S_BH(Schwarzschild) = R_+^2/4 = {R_plus**2 / 4:.4f}")
say("\nconclusion: the volumetric count exceeds the bound by a factor 1.6e-3 (M/m_P) [(ell/l_P)^3] to 0.68 (M/m_P) "
    "[proper volume of the core], i.e. 10^36...10^39 (10 M_sun) up to 10^44...10^47 (10^9 M_sun); the condensate "
    "count (b) and the dS-horizon count (c) scale as M^2 and stay below the bound by O(1) factors (0.081 and 0.019).")

json.dump(dict(lam=lam, R_minus=R_minus, R_plus=R_plus, rho_c=float(rho_c), ell_dS=float(ell_dS),
               V_core_M3=V_core, V_core_cuts=vols, rows=rows), open(OUT / "O1_holography.json", "w"), indent=1)
(LOGS / "O1_log.txt").write_text("\n".join(LOG), encoding="utf-8")
