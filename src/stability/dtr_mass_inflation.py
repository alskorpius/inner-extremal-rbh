"""Mass inflation via a model of two intersecting light shells (DTR relation) in metrics of class f(r; m).

The Dray-'t Hooft-Redmount relation for four regions around the intersection point r0 (A -- outside both shells,
B -- behind the ingoing shell, C -- behind the outgoing shell, D -- between them after the intersection):
    f_A(r0) f_D(r0) = f_B(r0) f_C(r0).
For Schwarzschild f = 1 - 2m/r: as r0 -> r_- (f_A -> 0) the mass m_D -> infinity -- classical mass inflation.
For Hayward f(m) = 1 - 2 m r^2/(r^3 + 2 m ell^2), bounded below: f(m -> inf) = 1 - r^2/ell^2; for r0 > ell this
gives f_inf < 0, and the equation f_A f_D = f_B f_C may have NO solution at finite m_D (when the required f_D < f_inf).
Question: how does m_D(r0) behave as r0 -> R_- for small shells (dm_in, dm_out ≪ M) and what does this mean physically:
a finite "inflated" mass (saturation) or no solution (breakdown of the thin-shell model).
Also: tidal forces and K in region D for the resulting m_D.
Run: python src/stability/dtr_mass_inflation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach_map"))
import approach_map as am  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name


def f_hay(r, m, ell):
    return 1 - 2 * m * r**2 / (r**3 + 2 * m * ell**2)


def f_schw(r, m, ell=None):
    return 1 - 2 * m / r


def inner_horizon(m, ell):
    fam = am.Family("h", "h", am.F_hayward, dict(m=m, ell=ell))
    hs = am.horizons(fam)
    return hs[0]["R"] if len(hs) >= 2 else None


def dtr_solve(f, r0, mA, dm_in, dm_out, ell, m_max=1e12):
    """Region A: mass mA; B (behind the ingoing shell): mA + dm_in; C (behind the outgoing shell, from inside): mA - dm_out;
    D: unknown m_D. Solve f(mA) f(mD) = f(mB) f(mC) for m_D."""
    fA, fB, fC = f(r0, mA, ell), f(r0, mA + dm_in, ell), f(r0, mA - dm_out, ell)
    target = fB * fC / fA
    g = lambda mD: f(r0, mD, ell) - target
    # look for the root on (0, m_max]; f is monotonic in m at fixed r
    lo, hi = 1e-12, m_max
    if g(lo) * g(hi) > 0:
        return None, target, f(r0, hi, ell)
    return brentq(g, lo, hi, xtol=1e-12, rtol=1e-12), target, f(r0, hi, ell)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M, ell = 1.0, 0.2
    Rm = inner_horizon(M, ell)
    print(f"Hayward M={M}, ell={ell}: inner horizon R_- = {Rm:.6f}; f(m->inf, r) = 1 - r^2/ell^2 -> negative for r > ell")
    rows = []
    for dm in (1e-3, 1e-6):
        print(f"\nShells dm_in = dm_out = {dm}: inflated mass m_D at intersection at r0 = R_- (1 + eps)")
        for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            r0 = Rm * (1 + eps)
            mD_h, tgt_h, finf = dtr_solve(f_hay, r0, M, dm, dm, ell)
            mD_s, tgt_s, _ = dtr_solve(f_schw, 2 * M * (1 + 0) * 0 + r0 * (2 * M / Rm), M, dm, dm, ell)   # Schwarzschild: rescale r0 to its 'inner' -- no; use r0 -> 2M(1+eps) below
            # Schwarzschild has no inner horizon; for comparison we would use Reissner-Nordstrom? Limit ourselves to Hayward and an estimate via f_A.
            fA = f_hay(r0, M, ell)
            row = dict(dm=dm, eps=eps, r0=r0, f_A=fA, target_fD=tgt_h, f_inf=finf, m_D=mD_h)
            rows.append(row)
            print(f"  eps={eps:6.0e}: f_A={fA:+.3e}, required f_D={tgt_h:+.4e}, f(m->inf)={finf:+.4f}, m_D = {('no solution (f_D < f_inf)' if mD_h is None else f'{mD_h:.4g}')}")
    # estimate: at which eps the solution disappears -- target < f_inf; f_A ~ -2 kappa_- (r0 - R_-) => target ~ fB fC / f_A
    (OUT / "dtr_mass_inflation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    # curvature in region D at the found m_D
    print("\nCurvature of region D (K_max, tidal) at the found m_D -- saturation at fixed ell:")
    for mD in (1, 10, 100, 1e4, 1e6):
        fam = am.Family("h", "h", am.F_hayward, dict(m=mD, ell=ell), x_min=1e-4, x_max=max(12.0, 3 * mD))
        x = np.geomspace(1e-4, 3 * ell, 4000)
        T = am.tensors(fam, x)
        print(f"  m_D={mD:8g}: K_max(core)={np.nanmax(T['K']):.6g} (24/ell^4={24/ell**4:.6g}), max|f''/2|={np.nanmax(np.abs(np.asarray(T['fpp'])))/2:.4g}, R_-={inner_horizon(mD, ell)}")
    print("->", OUT)


if __name__ == "__main__":
    main()
