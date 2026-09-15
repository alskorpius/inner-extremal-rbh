"""4D Lagrangian for the scenario medium: nonlinear electrodynamics (NED) with a magnetic charge.
Moreno-Sarbach conventions (gr-qc/0208090): S = (1/4pi) int sqrt(-g) (R/4 - L(F)), F = (1/4) F_mn F^mn;
magnetic monopole F_th_ph = Q sin th => F = Q^2/(2 r^4); the Einstein equations give m' = r^2 L (dually, their (11): m' = -r^2 H),
i.e. L(F) = m'(r)/r^2 = 4 pi rho(r) (in the Fan-Wang 1610.02636 convention, L_FW = 4 m'/r^2, F_FW = 2Q^2/r^4 -- the same physics).
The magnetic solution with L(F) is equivalent (F-P duality, their (8)) to the electric one in the P-formalism with H(P) = -L(-P):
H_P = L_F, H_PP = -L_FF, kappa = 1 + 2 H_P^{-1} H_PP P = 1 + 2 F L_FF/L_F.
Moreno-Sarbach linear-stability conditions outside the event horizon (their (51)-(52), (48) and section IV): H < 0, H_P > 0,
kappa > 0, the matrix S (their (47)) positive definite; sufficient 0 < 2 N kappa <= l(l+1); necessary kappa >= 0.
Here: we reconstruct L(F) for the branch-A profile, check L > 0, L_F > 0, kappa, det S for l = 2, 3, 4 at r > r_h;
weak field: L ~ F^(s_end/4); the theory's constants (L_max, F_1) and the consequence ell = const (a fundamental scale).
Run: python src/ned_lagrangian/ned_reconstruct.py
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
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[0] / "inhomogeneous_collapse"))
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


scen = lb.ScenarioMass()
ell, rho_c, Itot = scen.ell, scen.rho_c, scen.Itot
M = 1.0
R1 = (2 * M * ell**2 / (3 * Itot)) ** (1 / 3)
LU, ln_s, sig = scen.LU, scen.ln_s, scen.sig
sigp = np.gradient(sig, LU)                 # d sigma / d ln u
u = np.exp(LU)
R = u * R1
rho = rho_c * np.exp(ln_s)
d = scen.all(M, R)
m = d["m"]
N = 1 - 2 * m / R
Q = 1.0
F = Q**2 / (2 * R**4)
L = 4 * np.pi * rho                          # = m'/r^2 (checked below)
say(f"Branch-A profile: ell = {ell:.5f}, rho_c = {rho_c:.4f}, R1 = {R1:.5f}; check L = m'/r^2: max relative difference {np.max(np.abs(d['m_R']/R**2/L - 1)):.1e}")
# derivatives with respect to F via R: dF/dR = -4F/R
dL_dR = 4 * np.pi * (-sig * rho / R)
L_F = dL_dR / (-4 * F / R)                   # = pi sigma rho R / F > 0
L_FF = np.gradient(L_F, R) / (-4 * F / R)
kappa_num = 1 + 2 * F * L_FF / L_F
kappa_an = sig / 2 - 1 - sigp / (2 * sig)    # analytically: L ~ F^{sigma/4} locally
ok = sig > 1e-3
say(f"kappa numerical vs. analytic sigma/2 - 1 - sigma'/(2 sigma): max difference for sigma > 1e-3: {np.max(np.abs(kappa_num[ok] - kappa_an[ok])):.2e}")
kappa = np.where(ok, kappa_an, np.nan)
# horizons
hz = [(Rr, k) for Rr, k in lb.static_horizons(scen, M)]
Rh = hz[-1][0]
say(f"Horizons (R, kappa_surf): {[(round(a, 5), round(b, 6)) for a, b in hz]}; domain of the Moreno-Sarbach theorem: r > r_h = {Rh:.5f}")
out = np.nonzero(R > Rh)[0]
ins = np.nonzero((R < Rh) & (R > 1e-3))[0]
say(f"Outside the horizon: min L = {L[out].min():.3e} > 0: {bool((L[out] > 0).all())}; min L_F = {L_F[out].min():.3e} > 0: {bool((L_F[out] > 0).all())}; "
    f"kappa from {np.nanmin(kappa[out]):.4f} to {np.nanmax(kappa[out]):.4f} (> 0: {bool(np.nanmin(kappa[out]) > 0)}); sigma from {sig[out].min():.3f} to {sig[out].max():.3f}")
say(f"Inside the horizon (informational, the theorem does not apply): kappa < 0 for R in [{R[ins][kappa[ins] < 0].min():.4f}, {R[ins][kappa[ins] < 0].max():.4f}] (core sigma->0 and the dip sigma=1: kappa -> -1 and -0.5); "
    f"min kappa = {np.nanmin(kappa[ins]):.3f}")
# Moreno-Sarbach matrix S (their (47)) for l = 2,3,4 at r > r_h
H = -L; HP = L_F
a = 6 * m / R + 2 * R**2 * H
res = {}
for l in (2, 3, 4):
    lam = (l - 1) * (l + 2)
    b = lam + 4 * HP * Q**2 / R**2
    c1 = lam + 1 - N - 2 * R**2 * H
    c2 = c1 + 4 * N * kappa
    w = c1 + 2 * N * kappa
    S11 = lam / (R**2 * (a + lam)) * (c1 + 2 * N * b / (a + lam))
    S12 = -np.sqrt(4 * lam) * HP * Q / (R**3 * (a + lam)) * (w + 2 * N * b / (a + lam))
    S22 = kappa * l * (l + 1) / R**2 + 4 * HP * Q**2 / (R**4 * (a + lam)) * (c2 + 2 * N * b / (a + lam))
    det = S11 * S22 - S12**2
    o = out
    suff = 2 * N[o] * kappa[o] <= l * (l + 1)
    res[l] = dict(min_S11=float(np.nanmin(S11[o])), min_S22=float(np.nanmin(S22[o])), min_det=float(np.nanmin(det[o])), min_a=float(a[o].min()),
                  sufficient_48_holds_everywhere=bool(suff.all()), sufficient_48_fails_for_R_above=float(R[o][~suff].min()) if (~suff).any() else None,
                  min_det_where_48_fails=float(np.nanmin(det[o][~suff])) if (~suff).any() else None)
    say(f"l={l}: outside the horizon min S11 = {res[l]['min_S11']:.3e}, min S22 = {res[l]['min_S22']:.3e}, min det S = {res[l]['min_det']:.3e}, min a = {res[l]['min_a']:.3f}; "
        f"sufficient condition (48) 2N kappa <= l(l+1) holds everywhere: {res[l]['sufficient_48_holds_everywhere']}"
        + (f" (violated for R > {res[l]['sufficient_48_fails_for_R_above']:.3f}, but there det S >= {res[l]['min_det_where_48_fails']:.3e} > 0 by direct check)" if (~suff).any() else ""))
# weak field and the theory's constants
tail = R > 3.0
n_tail = np.polyfit(np.log(F[tail]), np.log(L[tail]), 1)[0]
say(f"Weak field (R > 3M): L ~ F^{n_tail:.2f} (expected s_end/4 = 6): there is no Maxwell term -- the 'magnetic charge' Q does not source a Coulomb field; field energy density ~ R^-{4*n_tail:.0f}")
L_max = 4 * np.pi * rho_c
F1 = Q**2 / (2 * R1**4)
say(f"Theory constants: L_max = 4 pi rho_c = {L_max:.4f} (= 3/(2 ell^2)), F_1 = Q^2/(2 R1^4) = {F1:.4g} (at Q = 1). Family of solutions of the theory at varying Q: R1 ∝ Q^(1/2), M ∝ Q^(3/2), ell = const.")
# consequence: in a fixed theory the triple root occurs only at M* (Q*); as Q -> Q(1±dq) the mass becomes M(1±1.5dq) and kappa_- detunes as for fixed ell
for dq in (-0.02, 0.0, 0.02):
    Mq = (1 + dq) ** 1.5
    hzq = lb.static_horizons(scen, Mq)
    say(f"  Q/Q* = {1+dq:.2f}: M/M* = {Mq:.4f}, horizons (R, kappa): {[(round(a_, 4), round(b_, 5)) for a_, b_ in hzq]}")
# save L(F) as a table
np.savetxt(OUT / "L_of_F_branchA.csv", np.column_stack([R, F, L, L_F, kappa]), delimiter=",", header="R, F=Q^2/(2R^4) (Q=1), L=4 pi rho, L_F, kappa=1+2F L_FF/L_F", comments="")
json.dump(dict(ell=ell, rho_c=rho_c, R1=R1, horizons=hz, L_max=L_max, F1=F1, weak_field_power=n_tail, MS=res,
               kappa_out=dict(min=float(np.nanmin(kappa[out])), max=float(np.nanmax(kappa[out]))), kappa_in_min=float(np.nanmin(kappa[ins]))),
          open(OUT / "ned_reconstruct.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
(LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
say(f"-> {OUT}")
