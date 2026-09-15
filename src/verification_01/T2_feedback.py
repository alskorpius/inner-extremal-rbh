"""T2 (the reviewer's verification tasks): mass inflation as feedback — is there an attractor to kappa_- = 0.
Iteration "static -> inflation (Ori model, CR 2021 (10), (32)) -> static":
  background: M_bg(r) = (1 + eps) M_scen(1, r) [+ accumulated addition]; the law in the parameter m is linearized: M(m, r) = M_bg(r) + (m - 1) G(r), G = dM_scen/dm;
  Ori: dR/dv = f_-/2, m_+' = m_-' f_+/f_- (A_-/A_+ = 1 for a linear law), m_-(v) = 1 - beta/v^p (Price tail);
  N_e = ln(M_+(v)/M_+(v0)), M_+ = M(m_+, R) — the Misner–Sharp mass on the shell;
  substitution: (P1) delta m(r) = (m_+(v(r)) - 1) G(r) on the trajectory r = R(v) with a smooth cutoff for r < R_end;
               (P1s) same, smoothed with a Gaussian in ln r of width ell/r (mimicking dissipation);
               (P2) step: (m_+(v_end) - 1) G(r) for r > R_end.
The sign of kappa_- at the inner boundary of the trapped region is always <= 0 (f decreases through zero moving outward), so the "two signs"
are realized by the two signs of eps (heavier/lighter profile); a positive kappa_- at this boundary is topologically impossible.
Run: python src/verification_01/T2_feedback.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

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
ELL = scen.ell
RG = np.geomspace(1e-3, 4.0, 40001)
LRG = np.log(RG)
_d = scen.all(1.0, RG)
M_SCEN = np.asarray(_d["m"], float)
G_SCEN = np.asarray(_d["m_M"], float)
G_spl = CubicSpline(LRG, G_SCEN)


class Background:
    def __init__(self, M_arr):
        self.M_arr = np.asarray(M_arr, float)
        self.spl = CubicSpline(LRG, self.M_arr)

    def M(self, r):
        return self.spl(np.log(r))

    def f(self, r):
        return 1 - 2 * self.M(r) / r

    def fprime(self, r):
        lr = np.log(r)
        return 2 * self.M(r) / r**2 - 2 * self.spl(lr, 1) / r**2   # dM/dr = (dM/dln r)/r

    def roots(self):
        f = self.f(RG)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        return [brentq(self.f, RG[i], RG[i + 1], xtol=1e-14) for i in idx]

    def trapped(self, r_ref):
        """(r_-, r_+, kappa_-, kappa_+) of the trapped region containing r_ref; None if it does not exist."""
        rs = self.roots()
        below = [r for r in rs if r < r_ref]; above = [r for r in rs if r > r_ref]
        if not below or not above or self.f(r_ref) >= 0:
            return None
        rm, rp = max(below), min(above)
        return rm, rp, 0.5 * self.fprime(rm), 0.5 * self.fprime(rp)


def Mfun(bg, m, r):
    return bg.M(r) + (m - 1.0) * G_spl(np.log(r))


def ori_run(bg, N_target, eps0=1e-3, p=11, v0=50.0, v_end=1e4, R0=None, dm_plus=1e-4, continue_asym=True):
    tr = bg.trapped(R0 or 1.3)
    if tr is None:
        return None
    rm, rp, km, kp = tr
    R0 = R0 or 0.5 * (rm + rp)
    beta = eps0 * v0**p
    m_minus = lambda v: 1.0 - beta / v**p
    dm_minus = lambda v: p * beta / v ** (p + 1)

    def rhs(lv, y):
        v = np.exp(lv)
        R, mp = y
        fm = 1 - 2 * Mfun(bg, m_minus(v), R) / R
        fp = 1 - 2 * Mfun(bg, mp, R) / R
        if abs(fm) < 1e-14:            # guard against division by zero near the root (below machine precision of f_-)
            fm = -1e-14
        return [v * fm / 2, v * dm_minus(v) * fp / fm]

    y0 = [R0, m_minus(v0) * (1 + dm_plus)]
    mp0 = y0[1]

    def ev_N(lv, y):            # N_e = ln(m_+/m_+(v0)) — growth of the mass parameter of the R_+ region (as in the task)
        return np.log(abs(y[1] / mp0)) - N_target
    ev_N.terminal = True; ev_N.direction = 1

    def ev_stick(lv, y):        # the shell has "stuck" to zero of f_- (|f_-| < 1e-12): further integration is not possible (stiffness, machine precision)
        v = np.exp(lv)
        return (1 - 2 * Mfun(bg, m_minus(v), y[0]) / y[0]) + 1e-12
    ev_stick.terminal = True; ev_stick.direction = 1

    lv = np.linspace(np.log(v0), np.log(v_end), 3000)
    sol = solve_ivp(rhs, (lv[0], lv[-1]), y0, method="LSODA", rtol=1e-9, atol=1e-13, t_eval=lv, events=[ev_N, ev_stick])
    v = np.exp(sol.t); R, mp = sol.y
    for k in (0, 1):
        if sol.t_events[k].size:
            v = np.append(v, np.exp(sol.t_events[k][0])); R = np.append(R, sol.y_events[k][0][0]); mp = np.append(mp, sol.y_events[k][0][1])
    Ne_num = float(np.log(abs(mp[-1] / mp0)))
    dm_tail = m_minus(v[-1]) - m_minus(v0)                      # tail influx over the same time
    excess = float((mp[-1] - mp0 - dm_tail) / dm_tail) if dm_tail > 0 else np.nan
    v_num = float(v[-1])
    continued = False
    if not sol.t_events[0].size and km < 0 and continue_asym:
        # asymptotic continuation (Ori's law, CR 2021 (12)–(13)): dN/dv = max(|kappa_-| - (p+1)/v, 0), R - r_- ∝ e^{-|kappa_-| v}
        k = abs(km); vs = v[-1]; Rs = R[-1]; Ns = np.log(abs(mp[-1] / mp0))
        # v_f: N(v_f) = N_target
        def Nfun(vv):
            return Ns + k * (vv - vs) - (p + 1) * np.log(vv / vs) if vv > (p + 1) / k else Ns + max(k * (vv - vs) - (p + 1) * np.log(vv / vs), 0.0)
        v1 = max(vs, (p + 1) / k)
        vf = v1
        while Nfun(vf) < N_target and vf < 1e12:
            vf *= 1.05
        if Nfun(vf) >= N_target:
            vf = brentq(lambda x: Nfun(x) - N_target, v1, vf)
            vc = np.geomspace(vs * (1 + 1e-9), vf, 400)
            Nc = np.array([Nfun(x) for x in vc])
            mpc = mp0 * np.exp(Nc)
            Rc = rm + (Rs - rm) * np.exp(-k * (vc - vs))
            v = np.append(v, vc); R = np.append(R, Rc); mp = np.append(mp, mpc)
            continued = True
    Mplus = np.array([Mfun(bg, a, b) for a, b in zip(mp, R)])
    Ne = float(np.log(abs(mp[-1] / mp0)))
    return dict(v=v, R=R, mp=mp, Mplus=Mplus, Ne=Ne, Ne_num=Ne_num, excess=excess, v_last=float(v[-1]), v_num=v_num, rm=rm, rp=rp, km=km, kp=kp,
                status=int(sol.status), hit=bool(sol.t_events[0].size) or continued, stuck=bool(sol.t_events[1].size), continued=continued)


def add_dm(bg, run, prescription="P1", smooth_ell=False, w=0.005):
    ok = np.isfinite(run["R"]) & np.isfinite(run["mp"])
    R, mp = run["R"][ok], run["mp"][ok]
    order = np.argsort(R)
    Rs, mps = R[order], mp[order]
    R_end = float(Rs[0])
    if prescription == "P1":
        dmp = np.interp(RG, Rs, mps - 1.0, left=mps[0] - 1.0, right=mps[-1] - 1.0)
        S = 1 / (1 + np.exp(-(RG - R_end) / w))          # smooth cutoff inward of R_end
        dm = dmp * G_SCEN * S
    else:                                                  # P2: step outside of R_end
        S = 1 / (1 + np.exp(-(RG - R_end) / w))
        dm = (mps[0] - 1.0) * G_SCEN * S
    if smooth_ell:
        sig = ELL / R_end                                   # width in ln r
        dl = LRG[1] - LRG[0]
        n = int(4 * sig / dl)
        ker = np.exp(-0.5 * (np.arange(-n, n + 1) * dl / sig) ** 2); ker /= ker.sum()
        dm = np.convolve(dm, ker, mode="same")
    return Background(bg.M_arr + dm), float(np.max(dm)), R_end


def eps_for_kappa(kappa_target, sign):
    """|eps| via bisection: kappa_-(eps) at the inner boundary of the trapped region containing r = 1.3."""
    def kap(e):
        tr = Background((1 + e) * M_SCEN).trapped(1.3)
        return tr[2] if tr else np.nan
    lo, hi = 1e-9, 0.2
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        k = kap(sign * mid)
        if np.isnan(k) or abs(k) > abs(kappa_target):
            hi = mid
        else:
            lo = mid
    return sign * np.sqrt(lo * hi)


def iterate(eps, N_target, n_iter, prescription="P1", smooth_ell=False, tag=""):
    bg = Background((1 + eps) * M_SCEN)
    tr = bg.trapped(1.3)
    hist = [dict(it=0, kappa=tr[2], r_minus=tr[0], r_plus=tr[1], Ne_cum=0.0, dm_max=0.0)]
    Ne_cum = 0.0
    for it in range(1, n_iter + 1):
        run = ori_run(bg, N_target)
        if run is None:
            say(f"    [{tag}] iteration {it}: no trapped region — stop"); break
        bg, dm_max, R_end = add_dm(bg, run, prescription, smooth_ell)
        tr = bg.trapped(1.3)
        Ne_cum += run["Ne"]
        if tr is None:
            say(f"    [{tag}] iteration {it}: after the substitution there is no trapped region (roots {bg.roots()}) — stop")
            hist.append(dict(it=it, kappa=np.nan, r_minus=np.nan, r_plus=np.nan, Ne_cum=Ne_cum, dm_max=dm_max, Ne_run=run["Ne"], v_last=run["v_last"], R_end=R_end))
            break
        hist.append(dict(it=it, kappa=tr[2], r_minus=tr[0], r_plus=tr[1], Ne_cum=Ne_cum, dm_max=dm_max, Ne_run=run["Ne"], v_last=run["v_last"], R_end=R_end, hit=run["hit"]))
        say(f"    [{tag}] it. {it}: numerically N_e = {run['Ne_num']:.2e} up to v = {run['v_num']:.3g} (sticking: {run['stuck']}, excess over the tail {run['excess']:+.2e}); "
            f"{'Ori asymptotics up to v = %.3g, N_e = %.2f' % (run['v_last'], run['Ne']) if run['continued'] else 'no continuation'}; R_end - r_- = {R_end - run['rm']:.1e}, max dm = {dm_max:.2e}; "
            f"new r_- = {tr[0]:.5f}, R_+ = {tr[1]:.4f}, kappa_- = {tr[2]:+.4e}")
        if not run["hit"]:
            say(f"    [{tag}] target N_e not reached and continuation is not possible — stop"); break
    return hist


def main():
    t0 = time.time()
    say(f"T2: mass-inflation feedback. Scenario profile, ell = {ELL:.4f}; the law in m is linearized (G = dM/dm).")
    tr0 = Background(M_SCEN).trapped(1.3)
    say(f"  unperturbed background: r_- = {tr0[0]:.5f}, R_+ = {tr0[1]:.5f}, kappa_- = {tr0[2]:+.2e}, kappa_+ = {tr0[3]:.4f}")
    # sign of kappa_-: both signs of eps
    say("\n1) Amplitude perturbation m -> (1 + eps) m: kappa_- at the inner boundary of the trapped region")
    starts = []
    for kt in (1e-3, 1e-2, 3e-2, 8e-2):
        for sgn in (+1, -1):
            e = eps_for_kappa(kt, sgn)
            tr = Background((1 + e) * M_SCEN).trapped(1.3)
            roots = Background((1 + e) * M_SCEN).roots()
            say(f"  |kappa_-| ~ {kt:.0e}, eps = {e:+.3e}: roots {[round(r, 4) for r in roots]}, r_- = {tr[0]:.4f}, kappa_- = {tr[2]:+.4e} (law: kappa ~ eps^(2/3) => |eps| ~ {kt**1.5:.1e}·C)")
            starts.append((kt, sgn, e))
    say("  Conclusion: for both signs of eps, kappa_- < 0 (a single root, shifted inward for eps > 0 and outward for eps < 0); kappa_- > 0 at the trapped-region boundary is unreachable.")

    results = {}
    say("\n2) Iterations \"static -> inflation -> static\", prescription P1 (addition along the trajectory), N_e = 1, 3, 10")
    for kt, sgn, e in starts:
        for Nt in ((1, 3, 10) if kt in (1e-3, 8e-2) else (3,)):
            tag = f"kappa~{sgn * kt:+.0e}, N={Nt}"
            say(f"  start {tag} (eps = {e:+.2e}):")
            hist = iterate(e, Nt, 4, "P1", False, tag)
            results[tag] = dict(eps=e, kappa0=sgn * kt, N_target=Nt, hist=hist)
            say(f"    [{time.time() - t0:.0f} s]")
    say("\n3) Substitution variants for |kappa_-| ~ 1e-3 and 8e-2, N_e = 3: smoothing over ell (P1s) and step (P2)")
    for kt, sgn, e in starts:
        if kt not in (1e-3, 8e-2):
            continue
        for presc, sm, name in (("P1", True, "P1s"), ("P2", False, "P2")):
            tag = f"kappa~{sgn * kt:+.0e}, N=3, {name}"
            say(f"  start {tag}:")
            hist = iterate(e, 3, 4, presc, sm, tag)
            results[tag] = dict(eps=e, kappa0=sgn * kt, N_target=3, prescription=name, hist=hist)
    # summary of d kappa / d N_e
    say("\n4) Summary: d kappa_- / d N_e (first step) and the behavior of |kappa_-|")
    summary = []
    for tag, r in results.items():
        h = [x for x in r["hist"] if np.isfinite(x["kappa"])]
        if len(h) < 2:
            summary.append(dict(tag=tag, note="no iterations")); continue
        dk = (h[1]["kappa"] - h[0]["kappa"]) / max(h[1]["Ne_cum"], 1e-12)
        ks = np.array([x["kappa"] for x in h])
        trend = "decreasing" if abs(ks[-1]) < abs(ks[0]) * 0.9 else ("growing" if abs(ks[-1]) > abs(ks[0]) * 1.1 else "constant")
        summary.append(dict(tag=tag, kappa_start=float(ks[0]), kappa_end=float(ks[-1]), dkappa_dNe_first=float(dk), Ne_total=float(h[-1]["Ne_cum"]), trend=trend, n_iter=len(h) - 1))
        say(f"  {tag:28s}: kappa {ks[0]:+.3e} -> {ks[-1]:+.3e} over N_e = {h[-1]['Ne_cum']:.2f} ({len(h) - 1} it.); d kappa/d N_e (1st step) = {dk:+.3e}; |kappa| {trend}")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for tag, r in results.items():
            if r.get("prescription") or r["N_target"] != 3:
                continue
            h = [x for x in r["hist"] if np.isfinite(x["kappa"])]
            ax.plot([x["Ne_cum"] for x in h], [abs(x["kappa"]) for x in h], "o-", label=tag)
        ax.set_yscale("log"); ax.set_xlabel("accumulated N_e"); ax.set_ylabel("|kappa_-|"); ax.legend(fontsize=7); ax.set_title("T2: |kappa_-| across iterations (P1, N_e = 3 per run)")
        fig.tight_layout(); fig.savefig(OUT / "T2_kappa_vs_Ne.png", dpi=110)
    except Exception as ex:  # noqa: BLE001
        say(f"plot not built: {ex}")
    json.dump(dict(results=results, summary=summary), open(OUT / "T2_feedback.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "T2_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
