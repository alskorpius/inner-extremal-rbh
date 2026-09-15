"""Stability of the layer inside the horizon (R- < R < R+) for the NED realization: the Moreno-Sarbach system (46) continued inward
(local equations; N < 0, r is the time coordinate). In variables (t, r*) with dr*/dr = 1/N: (d_t^2 - d_r*^2) u + N V u = 0;
for a mode e^{ikt}: u'' = (N V - k^2) u along r* (r* grows inward: -inf at R+, +inf at R- for the triple root).
Growth along r* occurs where N V - k^2 has positive eigenvalues; the coefficient of l(l+1) in N V22 equals N kappa/r^2:
for N < 0 and kappa < 0 it is positive and grows with l - Hadamard ill-posedness for the NED polarization.
We compute: kappa(R) inside, intervals of kappa < 0, values of V on R-, the transfer matrix (4x4) from R+(1-δ) to R-(1+δ) for l = 2,5,10,20,
k = 0 and k = 0.5; the amplification exponent ln||T||. For comparison - the Hayward NED realization (ell = 0.271, kappa = sigma - 4).
Run: python src/polar_qnm/interior_layer.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "0.05"]
import polar_qnm as pq  # noqa: E402
import ltb_bounce as lb  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


class HaywardBG:
    def __init__(self, ell, M=1.0):
        self.ell, self.M, self.scen = ell, M, "hayward"
        hz = lb.static_horizons(lb.hayward_mass(ell), M)
        self.rminus, self.rh = hz[0][0], hz[-1][0]

    def fields(self, r):
        M, ell = self.M, self.ell
        m = M * r**3 / (r**3 + 2 * M * ell**2)
        mp = 6 * M**2 * ell**2 * r**2 / (r**3 + 2 * M * ell**2) ** 2
        mpp = 12 * M**2 * ell**2 * r * (2 * M * ell**2 - 2 * r**3) / (r**3 + 2 * M * ell**2) ** 3
        sig = 6 * r**3 / (r**3 + 2 * M * ell**2)
        sigp = 18 * r**3 * 2 * M * ell**2 / (r**3 + 2 * M * ell**2) ** 2
        return dict(m=m, mp=mp, mpp=mpp, sigma=sig, sigp=sigp)


def analyse(name, bg, rminus, rplus, ls=(2, 5, 10, 20), ks=(0.0, 0.5), delta=1e-3):
    say(f"\n=== {name}: R- = {rminus:.5f}, R+ = {rplus:.5f}")
    r = np.linspace(rminus * (1 + 1e-4), rplus * (1 - 1e-4), 20001)
    N, V11, V12, V22, info = pq.potential_matrix(bg, r, 2)
    kap = info["kappa"]
    neg = r[kap < 0]
    # intervals of kappa<0
    ints = []
    if len(neg):
        br = np.nonzero(np.diff(neg) > 2 * (r[1] - r[0]))[0]
        starts = np.concatenate([[neg[0]], neg[br + 1]]); ends = np.concatenate([neg[br], [neg[-1]]])
        ints = [(float(a), float(b)) for a, b in zip(starts, ends)]
    say(f"  kappa_MS inside: min {kap.min():.3f} at R = {r[np.argmin(kap)]:.3f}; intervals of kappa < 0: {[(round(a,3), round(b,3)) for a, b in ints]}; kappa(R-+) = {kap[0]:+.3f}, kappa(R+-) = {kap[-1]:+.3f}")
    say(f"  at R-: V11 = {V11[0]:+.3f}, V22 = {V22[0]:+.3f}, V12 = {V12[0]:+.3f} (l=2); N V22 -> {N[0]*V22[0]:+.2e}; sign of N V (growth in r* when N V > 0): V11: {'growth' if N[0]*V11[0] > 0 else 'oscillation'}, V22: {'growth' if N[0]*V22[0] > 0 else 'oscillation'}")
    # growth regions for k=0: where N V22 > 0 (leading term l(l+1) N kappa / r^2)
    for l in ls:
        N, V11, V12, V22, info = pq.potential_matrix(bg, r, l)
        g22 = N * V22 > 0; g11 = N * V11 > 0
        say(f"  l={l}: N V22 > 0 (Phi growth) for R in [{r[g22].min() if g22.any() else float('nan'):.3f}, {r[g22].max() if g22.any() else float('nan'):.3f}] ({g22.mean()*100:.0f}% of interval); N V11 > 0 (Psi growth): {g11.mean()*100:.0f}% of interval")
    # transfer matrix along r*: u'' = (N V - k^2) u; integrate over r (dr* = dr/N): d/dr = (1/N) d/dr*
    results = {}
    r_lo, r_hi = rminus * (1 + delta), rplus * (1 - delta)
    rg = np.linspace(rminus * (1 + 1e-5), rplus * (1 - 1e-5), 40001)
    for l in ls:
        Ng, v11g, v12g, v22g, _ = pq.potential_matrix(bg, rg, l)
        P = lambda rr, arr: float(np.interp(rr, rg, arr))
        for k in ks:
            def rhs(rr, y):
                Nv = P(rr, Ng); v11 = P(rr, v11g); v12 = P(rr, v12g); v22 = P(rr, v22g)
                Y = y.reshape(4, 4)
                A = np.zeros((4, 4)); A[0, 2] = 1; A[1, 3] = 1
                A[2, 0] = Nv * v11 - k**2; A[2, 1] = Nv * v12; A[3, 0] = Nv * v12; A[3, 1] = Nv * v22 - k**2
                return ((A @ Y) / Nv).ravel()
            sol = solve_ivp(rhs, (r_hi, r_lo), np.eye(4).ravel(), method="DOP853", rtol=1e-8, atol=1e-10)
            T = sol.y[:, -1].reshape(4, 4)
            svals = np.linalg.svd(T, compute_uv=False)
            results[(l, k)] = float(np.log(svals[0]))
            say(f"  transfer R+ -> R- (δ = {delta}): l={l}, k={k}: ln||T|| = {np.log(svals[0]):+.2f}; status {sol.status}")
    Ng, v11g, v12g, v22g, _ = pq.potential_matrix(bg, rg, 2)
    for d in (1e-2, 1e-3, 1e-4):
        def rhs2(rr, y):
            Nv = float(np.interp(rr, rg, Ng)); v11 = float(np.interp(rr, rg, v11g)); v12 = float(np.interp(rr, rg, v12g)); v22 = float(np.interp(rr, rg, v22g))
            Y = y.reshape(4, 4); A = np.zeros((4, 4)); A[0, 2] = 1; A[1, 3] = 1
            A[2, 0] = Nv * v11; A[2, 1] = Nv * v12; A[3, 0] = Nv * v12; A[3, 1] = Nv * v22
            return ((A @ Y) / Nv).ravel()
        sol = solve_ivp(rhs2, (r_hi, rminus * (1 + d)), np.eye(4).ravel(), method="DOP853", rtol=1e-8, atol=1e-10)
        T = sol.y[:, -1].reshape(4, 4); sv = np.linalg.svd(T, compute_uv=False)
        rr_ = np.linspace(rminus * (1 + d), r_hi, 200001)
        rstar_span = float(np.trapezoid(1 / np.abs(np.interp(rr_, rg, Ng)), rr_))
        say(f"  l=2, k=0, up to R-(1+{d}): ln||T|| = {np.log(sv[0]):+.2f}; extent in r*: {rstar_span:.3g}")
    return results


def main():
    scen = lb.ScenarioMass()
    bgA = pq.Background(scen)
    hz = lb.static_horizons(scen, 1.0)
    rA = analyse("Scenario (branch A, NED realization)", bgA, hz[0][0], hz[-1][0])
    bgH = HaywardBG(0.27101577964329177)
    rH = analyse("Hayward ell = 0.271 (NED realization; kappa = sigma - 4)", bgH, bgH.rminus, bgH.rh)
    json.dump(dict(scenario={f"l{l}_k{k}": v for (l, k), v in rA.items()}, hayward={f"l{l}_k{k}": v for (l, k), v in rH.items()}),
              open(OUT / "interior_layer.json", "w", encoding="utf-8"), indent=1)
    (LOGS / "interior_layer_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
