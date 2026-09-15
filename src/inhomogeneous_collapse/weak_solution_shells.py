"""Weak solution beyond the caustic, version 2: N thin Israel dust shells in the effective geometry (Hayward, global ell).
Shell k: rest mass mu_k, labels M_in (cumulative interior mass) and M_out = M_in + dM_k (dM_k -- contribution to the Misner-Sharp
mass, fixed by the labels and unchanging), radius R. Israel equation for a dust shell between the regions f_in = 1 - 2 m(M_in,R)/R
and f_out:
    sqrt(f_in + Rdot^2) - sqrt(f_out + Rdot^2) = mu/R  =>  Rdot^2 = G(R) = (Dm(R)/mu + mu/(2R))^2 - f_in(R),  Dm = m(M_out,R) - m(M_in,R),
    Rddot = G'(R)/2 (independent of the sign of Rdot).
Initial data: rest at R = r_k => mu_k = r_k (sqrt f_in - sqrt f_out). Continuum limit -- LTB (control: bounce times).
Merger on contact: (a) if f_mid > 0 between the shells -- relativistic 4-momentum conservation in the static frame of the
middle region: beta' = sum(mu gamma beta)/sum(mu gamma), mu' = sum(mu gamma)/gamma' (heat -> rest mass), then Rdot' from beta';
    (b) otherwise (trapped region) -- Rdot' = the mu-weighted average of Rdot, mu' from the Israel equation (the root closest
    to mu_i + mu_j).
After the merger: M_in = M_in(inner), M_out = M_out(outer). Consistency check: G(R) >= 0 for the merged shell;
on inconsistency (no real root mu') the merger is flagged and mu' = mu_i + mu_j is used.
Run: python src/inhomogeneous_collapse/weak_solution_shells.py
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


def G_of(R, mu, Min, Mout):
    fin = 1 - 2 * m_eff(Min, R) / R
    Dm = m_eff(Mout, R) - m_eff(Min, R)
    return (Dm / mu + mu / (2 * R)) ** 2 - fin


def acc(R, mu, Min, Mout, h=1e-6):
    return (G_of(R * (1 + h), mu, Min, Mout) - G_of(R * (1 - h), mu, Min, Mout)) / (2 * h * R) / 2


def run(eps=0.1, r0=20.0, N=200, dt=0.004, tau_end=330.0, merge=True):
    prof = lb.MassProfile(eps, r0)
    r = r0 * np.linspace(0, 1, N + 1)[1:]
    Mout = prof.M(r); Min = np.concatenate([[0.0], Mout[:-1]])
    fin0 = 1 - 2 * m_eff(Min, r) / r; fout0 = 1 - 2 * m_eff(Mout, r) / r
    mu = r * (np.sqrt(fin0) - np.sqrt(fout0))
    R = r.copy(); V = np.zeros_like(R)
    t = 0.0; merges = []; heat = 0.0; inconsistent = 0
    tb = np.full(N, np.nan); alive = [[i] for i in range(N)]
    hist = []
    rec = int(0.5 / dt); step = 0
    while t < tau_end:
        a1 = acc(R, mu, Min, Mout); k1r = V
        a2 = acc(R + 0.5 * dt * k1r, mu, Min, Mout); k2r = V + 0.5 * dt * a1
        a3 = acc(R + 0.5 * dt * k2r, mu, Min, Mout); k3r = V + 0.5 * dt * a2
        a4 = acc(R + dt * k3r, mu, Min, Mout); k4r = V + dt * a3
        Rn = R + dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        Vn = V + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        for i in np.nonzero((V < 0) & (Vn >= 0))[0]:
            for j in alive[i]:
                if np.isnan(tb[j]):
                    tb[j] = t
        R, V = Rn, Vn; t += dt
        if merge:
            while True:
                bad = np.nonzero(R[:-1] > R[1:])[0]
                if len(bad) == 0:
                    break
                i = int(bad[0]); Rx = 0.5 * (R[i] + R[i + 1])
                Mi, Mo = Min[i], Mout[i + 1]
                f_mid = 1 - 2 * m_eff(Mout[i], Rx) / Rx
                mu_sum = mu[i] + mu[i + 1]
                if f_mid > 0:
                    g = np.sqrt(f_mid + V[i:i + 2] ** 2) / np.sqrt(f_mid); b = V[i:i + 2] / np.sqrt(f_mid + V[i:i + 2] ** 2)
                    P = np.sum(mu[i:i + 2] * g * b); Eg = np.sum(mu[i:i + 2] * g)
                    bm = P / Eg; gm = 1 / np.sqrt(1 - bm**2); mu_new = Eg / gm; Vm = np.sqrt(f_mid) * gm * bm; kind = "core-rel"
                    # consistency with the Israel equation for the given labels: G(Rx; mu_new) should equal Vm^2 -- otherwise adjust mu_new via the root
                    fin = 1 - 2 * m_eff(Mi, Rx) / Rx; Dm = m_eff(Mo, Rx) - m_eff(Mi, Rx); a_ = np.sqrt(fin + Vm**2)
                    disc = a_**2 - 2 * Dm / Rx
                    if disc >= 0:
                        roots = Rx * (a_ + np.array([1, -1]) * np.sqrt(disc))
                        mu_new = float(roots[np.argmin(np.abs(roots - mu_new))])
                    else:
                        inconsistent += 1; kind += "-inc"; mu_new = mu_sum
                else:
                    Vm = (mu[i] * V[i] + mu[i + 1] * V[i + 1]) / mu_sum
                    fin = 1 - 2 * m_eff(Mi, Rx) / Rx; Dm = m_eff(Mo, Rx) - m_eff(Mi, Rx); a_ = np.sqrt(fin + Vm**2)
                    disc = a_**2 - 2 * Dm / Rx
                    if disc >= 0:
                        roots = Rx * (a_ + np.array([1, -1]) * np.sqrt(disc)); mu_new = float(roots[np.argmin(np.abs(roots - mu_sum))]); kind = "trapped-NR"
                    else:
                        inconsistent += 1; kind = "trapped-inc"; mu_new = mu_sum
                heat += mu_new - mu_sum
                merges.append(dict(tau=float(t), R=float(Rx), R_over_ell=float(Rx / ELL), mu_sum=float(mu_sum), mu_new=float(mu_new), Vm=float(Vm), kind=kind, f_mid=float(f_mid)))
                R = np.delete(R, i); V = np.delete(V, i); Min = np.delete(Min, i + 1); Mout = np.delete(Mout, i); mu = np.delete(mu, i)
                alive[i + 1] = alive[i] + alive[i + 1]; del alive[i]
                R[i] = Rx; V[i] = Vm; mu[i] = mu_new
                # if G(R) < 0 after the merger -- state outside the domain of definition: flag it
                if G_of(Rx, mu_new, Min[i], Mout[i]) < -1e-9:
                    inconsistent += 1; merges[-1]["kind"] += "-Gneg"
        step += 1
        if step % rec == 0:
            hist.append((t, R.copy(), V.copy(), mu.copy(), Mout.copy()))
    return dict(hist=hist, merges=merges, heat=heat, inconsistent=inconsistent, R=R, V=V, mu=mu, Min=Min, Mout=Mout, alive=alive, tb=tb, N=N, mu0_sum=float(np.sum(r * (np.sqrt(fin0) - np.sqrt(fout0)))))


def summarize(res, label):
    m_ = res["merges"]
    say(f"\n=== {label}: {res['N']} -> {len(res['R'])} alive; mergers {len(m_)}; heat (rest-mass increase) {res['heat']:.4e} with total rest mass {res['mu0_sum']:.4f}; inconsistent mergers {res['inconsistent']}")
    if m_:
        kinds = {}
        for mm in m_:
            kinds[mm["kind"]] = kinds.get(mm["kind"], 0) + 1
        say(f"  merger types: {kinds}; R/ell from {min(mm['R_over_ell'] for mm in m_):.3g} to {max(mm['R_over_ell'] for mm in m_):.3g}; tau from {m_[0]['tau']:.2f} to {m_[-1]['tau']:.2f}")
    ib = int(np.argmax(res["mu"]))
    say(f"  largest shell: mu = {res['mu'][ib]:.4f} (dM = {res['Mout'][ib]-res['Min'][ib]:.4f}, from {len(res['alive'][ib])} originals), R = {res['R'][ib]:.4g}, V = {res['V'][ib]:+.4g}")
    for (tt, Rr, Vv, muu, Mo) in res["hist"]:
        if abs(tt - round(tt)) < 1e-6 and int(round(tt)) in (90, 100, 110, 130, 160, 200, 250, 300, 330):
            fo = 1 - 2 * m_eff(Mo, Rr) / Rr
            say(f"  tau = {tt:6.1f}: alive {len(Rr):4d}, R: min {Rr.min():.3g}, median {np.median(Rr):.3g}, max {Rr.max():.3g}; largest: R = {Rr[np.argmax(muu)]:.3g}, V = {Vv[np.argmax(muu)]:+.3g}, f_out = {fo[np.argmax(muu)]:+.3g}")


def main():
    say("Thin Israel shells, sticky merging; Hayward ell = 0.271, R0 = 20, N = 200.")
    ref = run(merge=False)
    say(f"Control without mergers: tau_b from {np.nanmin(ref['tb']):.3f} to {np.nanmax(ref['tb']):.3f} (LTB: 95.36 ... 99.93 at eps = 0.1)")
    out = {}
    for eps in (0.1, 0.01, 0.3):
        res = run(eps=eps)
        summarize(res, f"eps = {eps:+.2f}")
        out[str(eps)] = dict(n_merges=len(res["merges"]), heat=res["heat"], inconsistent=res["inconsistent"], n_alive=len(res["R"]), mu_max=float(res["mu"].max()),
                             R_end=[float(x) for x in res["R"]], V_end=[float(x) for x in res["V"]], merges_head=res["merges"][:30])
    json.dump(out, open(OUT / "weak_solution_shells.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "weak_solution_shells_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
