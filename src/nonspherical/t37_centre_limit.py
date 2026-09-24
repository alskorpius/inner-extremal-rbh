"""Centre limit of the polar (Moreno-Sarbach) operator in the de Sitter core, with the potential of
t37_core_stability.py.

Near r = 0, f -> 1 and the Sturm-Liouville system -(f u')' + W u = omega^2 u / f reduces to u'' = (A / r^2) u,
A = lim r^2 W (2x2: metric and matter channels); u ~ r^s with s(s - 1) = eigenvalue of A.

Besides the l(l+1) coefficient c_t^2(0) = -1 - p/2, the matter channel contains the l-independent static term
H_P^{-1/2} (N (H_P^{1/2})')' with H_P Q^2 = r^2 (2m' - r m'')/2 = 2 pi r^4 rho sigma ~ r^(p+4) for a plateau of
index p. It adds k(k - 1)/r^2 with k = (p+4)/2, so

    r^2 W_22 -> -(p+2)/2 l(l+1) + (p+2)(p+4)/4 = -(p+2)/4 [2 l(l+1) - (p+4)].

Fall to the centre (coefficient < -1/4) therefore needs 2 l(l+1) > p + 4 + 1/(p+2). This script checks the limit
on a fine grid for the four profiles of t37_core_stability.py and prints the l = 2 spectrum of branch A against
the inner cut-off. Large p, where float64 loses H_P Q^2 to cancellation, is handled exactly in t37_large_p.py.

Run: python src/nonspherical/t37_centre_limit.py        (~1 min)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import t37_core_stability as t  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def limit_W22(core, l, r0):
    r = np.geomspace(r0 / 3, r0 * 3, 4001)
    N, V11, V12, V22, info = t.pq.potential_matrix(t._BgAdapter(core), r, l)
    kap = t.core_quantities(core, r)["kappa"]
    V22 = V22 + (kap - info["kappa"]) * l * (l + 1) / r**2
    i = len(r) // 2
    return float(V22[i] * r[i] ** 2)


def main():
    res = {}
    cores = [(t.ScenarioCore(), 4), (t.hayward_core(), 3), (t.bardeen_core(), 2), (t.dymnikova_core(), 3)]
    say("Centre limit r^2 W_22 of the matter channel (fine grid around r0)")
    for core, p in cores:
        rows = []
        for l in [2, 3, 5, 10]:
            v = [limit_W22(core, l, r0) for r0 in (3e-3, 1e-2)]
            old = -(1 + p / 2) * l * (l + 1)
            full = -(p + 2) / 4 * (2 * l * (l + 1) - (p + 4))
            rows.append(dict(l=l, r2W22_at_r3em3=v[0], r2W22_at_r1em2=v[1], l_l1_part_only=old, full_formula=full,
                             falls_to_centre=bool(full < -0.25)))
            say(f"  {core.name[:34]:34s} p={p} l={l:2d}: {v[0]:9.3f} (r=3e-3) {v[1]:9.3f} (r=1e-2);"
                f"  l(l+1) part only {old:8.2f};  full formula {full:8.2f}")
        res[core.name] = dict(p=p, rows=rows)

    core = cores[0][0]
    say("\nBranch A, l = 2: lowest omega^2 of the two-channel solver on [r_cut, 0.6]")
    spec = []
    for rc in [0.1, 0.05, 0.02, 0.01, 0.005]:
        w0, _, wPhi, rpk = t.ms_two_channel_min(core, rc, 0.6, 2, n=3001)
        nu = float(np.sqrt(-w0)) if w0 < 0 else 0.0
        spec.append(dict(r_cut=rc, omega2=float(w0), nu=nu, nu_times_rcut=nu * rc, matter_weight=float(wPhi)))
        say(f"  r_cut={rc:6.3f}: omega^2 = {w0:11.4f}  nu = {nu:8.3f}  nu*r_cut = {nu*rc:6.3f}")
    res["branchA_l2_spectrum"] = spec
    (OUT / "t37_centre_limit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (LOGS / "t37_centre_limit_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
