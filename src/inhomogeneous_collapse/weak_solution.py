"""Weak solution beyond the caustic: "sticky dust" in the LTB model with effective mass (Hayward form, global ell).
Shells i (label r_i, cumulative mass M_i = M(r_i), dust mass mu_i = M_i - M_{i-1}) move according to
    Rddot = F(R, M_i) = -m/R^2 + m_R/R,  m = M R^3/(R^3 + 2 M ell^2)   (same as ltb_bounce; agrees up to the first merger).
When the ordering R_i > R_{i+1} is violated (crossing), two shells merge into one (inelastic collision) with local
relativistic 4-momentum conservation in the static frame (crossings after the bounce -- in the untrapped core, f > 0):
    gamma_k = sqrt(f + Rdot_k^2)/sqrt(f), beta_k = Rdot_k/sqrt(f + Rdot_k^2);  beta' = sum(mu gamma beta)/sum(mu gamma),
    mu'_rest = sum(mu gamma)/gamma' (>= mu_i + mu_j: kinetic energy -> "heat").
Prescription: the heat (mu'_rest - mu_i - mu_j) is taken to be radiated away and does NOT gravitate (cumulative labels M
are unchanged); the merged shell gets the label M_{i+1} and mass mu_i + mu_j. All caustics before the bounce (eps < 0)
are outside the scope of this treatment.
Output: number of mergers, merged-mass fraction, total heat, fate of the merged shells (re-bounce / exit through R-),
largest single-shell mass. Run: python src/inhomogeneous_collapse/weak_solution.py
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


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


ELL = 0.27101577964329177


def m_eff(M, R):
    return M * R**3 / (R**3 + 2 * M * ELL**2)


def F(R, M):
    m = m_eff(M, R)
    mR = 3 * M * R**2 * 2 * M * ELL**2 / (R**3 + 2 * M * ELL**2) ** 2
    return -m / R**2 + mR / R


def run(eps=0.1, r0=20.0, N=200, dt=0.004, tau_end=330.0, merge=True, label="sticky"):
    prof = lb.MassProfile(eps, r0)
    r = r0 * np.linspace(0, 1, N + 1)[1:]
    Mlab = prof.M(r)
    mu = np.diff(np.concatenate([[0.0], Mlab]))
    R = r.copy(); V = np.zeros_like(R)
    E = -m_eff(Mlab, R) / R                       # LTB energy (for diagnostics f = 1 + 2E - Rdot^2)
    heat = 0.0; merges = []; t = 0.0
    tb = np.full(N, np.nan)                        # first bounce of the label (indexed by the original shell: kept for the living ones)
    alive_ids = [[i] for i in range(N)]
    hist_t, hist_R, hist_nlive = [], [], []
    rec_every = int(0.5 / dt)
    step = 0
    while t < tau_end:
        # RK4 for all living shells
        def acc(Rv):
            return F(Rv, Mlab)
        k1v = acc(R); k1r = V
        k2v = acc(R + 0.5 * dt * k1r); k2r = V + 0.5 * dt * k1v
        k3v = acc(R + 0.5 * dt * k2r); k3r = V + 0.5 * dt * k2v
        k4v = acc(R + dt * k3r); k4r = V + dt * k3v
        Rn = R + dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        Vn = V + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        # bounces
        newly = (V < 0) & (Vn >= 0)
        for i in np.nonzero(newly)[0]:
            for j in alive_ids[i]:
                if np.isnan(tb[j]):
                    tb[j] = t
        R, V = Rn, Vn
        t += dt
        # crossings
        if merge:
            while True:
                bad = np.nonzero(R[:-1] > R[1:])[0]
                if len(bad) == 0:
                    break
                i = int(bad[0])
                Rx = 0.5 * (R[i] + R[i + 1])
                # local f for the static frame: use the label of the outer shell
                f_loc = 1 - 2 * m_eff(Mlab[i + 1], Rx) / Rx
                if f_loc <= 0:
                    # no static frame in the trapped region -- nonrelativistic momentum conservation (flagged)
                    mu_t = mu[i] + mu[i + 1]
                    Vm = (mu[i] * V[i] + mu[i + 1] * V[i + 1]) / mu_t
                    mrest = mu_t; kind = "trapped-NR"
                else:
                    g = np.sqrt(f_loc + V[i:i + 2] ** 2) / np.sqrt(f_loc)
                    b = V[i:i + 2] / np.sqrt(f_loc + V[i:i + 2] ** 2)
                    P = np.sum(mu[i:i + 2] * g * b); Eg = np.sum(mu[i:i + 2] * g)
                    bm = P / Eg; gm = 1 / np.sqrt(1 - bm**2)
                    mrest = Eg / gm; Vm = np.sqrt(f_loc) * gm * bm
                    mu_t = mu[i] + mu[i + 1]; kind = "core"
                heat += mrest - mu_t
                merges.append(dict(tau=t, R=float(Rx), R_over_ell=float(Rx / ELL), mu1=float(mu[i]), mu2=float(mu[i + 1]), V1=float(V[i]), V2=float(V[i + 1]),
                                   Vm=float(Vm), heat=float(mrest - mu_t), kind=kind, ids=(alive_ids[i][0], alive_ids[i + 1][-1])))
                # merger: keep position i+1 with label M_{i+1}
                R = np.delete(R, i); V = np.delete(V, i); Mlab = np.delete(Mlab, i); E = np.delete(E, i)
                mu[i + 1] = mu_t; mu = np.delete(mu, i)
                alive_ids[i + 1] = alive_ids[i] + alive_ids[i + 1]; del alive_ids[i]
                R[i] = Rx; V[i] = Vm
        step += 1
        if step % rec_every == 0:
            hist_t.append(t); hist_R.append(R.copy()); hist_nlive.append(len(R))
    return dict(t=hist_t, R=hist_R, nlive=hist_nlive, merges=merges, heat=heat, mu=mu, Mlab=Mlab, R_end=R, V_end=V, alive_ids=alive_ids, tb=tb, r=r, N=N)


def summarize(res, label):
    merges = res["merges"]
    N = res["N"]
    say(f"\n=== {label}: shells {N} -> {len(res['R_end'])} alive at the end; mergers {len(merges)}; total 'heat' {res['heat']:.4e} (fraction of M_tot)")
    if merges:
        taus = np.array([m["tau"] for m in merges]); Rs = np.array([m["R_over_ell"] for m in merges]); kinds = [m["kind"] for m in merges]
        say(f"  mergers: tau from {taus.min():.2f} to {taus.max():.2f}; R/ell from {Rs.min():.3g} to {Rs.max():.3g}; in the core (f>0): {kinds.count('core')}, in the trapped region: {kinds.count('trapped-NR')}")
        big = max(res["mu"]); ibig = int(np.argmax(res["mu"]))
        say(f"  largest merged shell: mu = {big:.4f} (from {len(res['alive_ids'][ibig])} originals), R = {res['R_end'][ibig]:.4g}, V = {res['V_end'][ibig]:+.4g} at the end")
    # fate: where the shells end up and how many times they bounced
    Rend = res["R_end"]
    say(f"  at the end: R from {Rend.min():.3g} to {Rend.max():.3g}; inside R- (0.293): {int((Rend < 0.293).sum())}, between horizons: {int(((Rend > 0.293) & (Rend < 1.962)).sum())}, outside R+: {int((Rend > 1.962).sum())}")
    # history of the number alive
    t = np.array(res["t"]); nl = np.array(res["nlive"])
    for tt in (90, 100, 110, 130, 160, 200, 250, 300):
        i = np.argmin(np.abs(t - tt))
        if i < len(nl):
            Rr = res["R"][i]
            say(f"  tau = {t[i]:6.1f}: alive {nl[i]:4d}, median R = {np.median(Rr):.3g}, min R = {Rr.min():.3g}, max R = {Rr.max():.3g}")


def main():
    say("Weak solution 'sticky dust' beyond the caustic: Hayward ell = 0.271, eps = +0.1, R0 = 20 (10 R_s).")
    ref = run(merge=False, label="no mergers (control)")
    # control: time of first bounce against ltb_bounce (95.36 for the innermost at eps=0.1, 99.93 for the outer)
    say(f"Control without mergers: tau_b of interior shells from {np.nanmin(ref['tb']):.3f} to {np.nanmax(ref['tb']):.3f} (ltb_bounce: 95.36 ... 99.93); "
        f"first violation of the ordering R_i > R_i+1 (crossing) -- from the history: " + (str(next((round(t_, 2) for t_, Rr in zip(ref['t'], ref['R']) if np.any(Rr[:-1] > Rr[1:])), 'none'))))
    res = run(merge=True, label="sticky")
    summarize(res, "sticky dust, eps = +0.1")
    res2 = run(eps=0.3, merge=True, label="sticky eps0.3")
    summarize(res2, "sticky dust, eps = +0.3")
    res3 = run(eps=0.01, merge=True, label="sticky eps0.01")
    summarize(res3, "sticky dust, eps = +0.01")
    json.dump(dict(eps01=dict(n_merges=len(res["merges"]), heat=res["heat"], merges=res["merges"][:50], mu_max=float(max(res["mu"])), nlive_end=len(res["R_end"])),
                   eps03=dict(n_merges=len(res2["merges"]), heat=res2["heat"], mu_max=float(max(res2["mu"])), nlive_end=len(res2["R_end"])),
                   eps001=dict(n_merges=len(res3["merges"]), heat=res3["heat"], mu_max=float(max(res3["mu"])), nlive_end=len(res3["R_end"]))),
              open(OUT / "weak_solution.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "weak_solution_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
