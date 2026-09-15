"""Field model of a defect: Einstein gravity + a scalar triplet with the "hedgehog" ansatz phi^a = eta f(r) x^a/r, V = (lambda/4)(phi^2 - eta^2)^2.
Metric ds^2 = -e^{2 delta} N dt^2 + dr^2/N + r^2 dOmega^2, N = 1 - 2m/r. Units: G = 1, r = delta_c x with delta_c = 1/(sqrt(lambda) eta),
rho = lambda eta^4 rho~, m = delta_c m~; the single parameter is alpha = 4 pi eta^2 (critical monopole 8 pi eta^2 = 1 <=> alpha = 1/2).
    rho~ = N f'^2/2 + f^2/x^2 + (f^2 - 1)^2/4,   m~' = alpha x^2 rho~,   delta' = alpha x f'^2,
    N x^2 f'' + (N' x^2 + 2 N x + delta' N x^2) f' = x^2 [2 f/x^2 + f (f^2 - 1)].
Tasks: (1) subcritical family (alpha < 1/2): shoot on a = f'(0) for f -> 1; the slope shape sigma = -d ln rho~/d ln x;
(2) supercritical (alpha > 1/2): a regular (non-degenerate) horizon N = 0 at finite x -- two-sided shooting (center + horizon)
on (a, x_h, f_h); triple-root indicators at the horizon: 8 pi r^2 rho = 2 alpha x_h^2 rho~_h (need 1), sigma_h (need 2), kappa_h;
(3) check of the analytic double-root condition: x_h^2 = 1/(1 - 1/(2 alpha)), f_h^2 = 1 - 2/x_h^2.
Run: python src/defect_field/defect_monopole.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
LOGS = Path(__file__).resolve().parents[2] / "logs" / Path(__file__).resolve().parent.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def rhs(x, y, alpha):
    f, fp, m, d = y
    N = 1 - 2 * m / x
    rho = N * fp**2 / 2 + f**2 / x**2 + (f**2 - 1) ** 2 / 4
    mp = alpha * x**2 * rho
    Np = (2 * m / x - 2 * mp) / x
    dp = alpha * x * fp**2
    fpp = (x**2 * (2 * f / x**2 + f * (f**2 - 1)) - (Np * x**2 + 2 * N * x + dp * N * x**2) * fp) / (N * x**2)
    return [fp, fpp, mp, dp]


def rho_of(x, f, fp, m):
    N = 1 - 2 * m / x
    return N * fp**2 / 2 + f**2 / x**2 + (f**2 - 1) ** 2 / 4


def from_center(a, alpha, x_end, x0=1e-4, dense=False, events=None):
    y0 = [a * x0, a, alpha * (a**2 / 2 + 1 / 12) * x0**3, 0.0]
    return solve_ivp(rhs, (x0, x_end), y0, args=(alpha,), rtol=1e-10, atol=1e-13, dense_output=dense, events=events, max_step=0.05)


# ---------- (1) subcritical family ----------
def shoot_sub(alpha, x_end=400.0):
    """a too small -- f drifts down (f < 0 or f' < 0 while f < 1 far out); too large -- f > 1 + margin."""
    def cls(a):
        ev_hi = lambda x, y, al: y[0] - 1.3; ev_hi.terminal = True
        ev_lo = lambda x, y, al: y[0] + 0.05; ev_lo.terminal = True
        ev_N = lambda x, y, al: 1 - 2 * y[2] / x - 1e-6; ev_N.terminal = True
        s = from_center(a, alpha, x_end, events=[ev_hi, ev_lo, ev_N])
        if s.t_events[0].size: return +1
        if s.t_events[1].size: return -1
        if s.t_events[2].size:      # de Sitter horizon (V ~ 1/4 like Lambda): the field either did not rise (a too small) or overshot
            return -1 if s.y_events[2][0][0] < 1 else +1
        # reached the end: check where f is heading
        return +1 if s.y[0, -1] > 1 else -1
    lo, hi = 0.01, 5.0
    assert cls(lo) == -1 and cls(hi) == +1, (cls(lo), cls(hi))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = cls(mid)
        if c == 0: return mid, "horizon"
        if c > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi), "ok"


def analyse_sub(alpha, a, x_end=200.0):
    ev_N = lambda x, y, al: 1 - 2 * y[2] / x - 1e-6; ev_N.terminal = True
    s = from_center(a, alpha, x_end, dense=True, events=[ev_N])
    x = np.geomspace(1e-3, s.t[-1] * 0.999, 40000)
    f, fp, m, d = s.sol(x)
    # cut off where the shooting diverges (|f - 1| > 0.02 after reaching the plateau)
    on = np.nonzero(np.abs(f - 1) < 0.02)[0]
    if on.size:
        bad = np.nonzero((np.abs(f - 1) > 0.02) & (x > x[on[0]]))[0]
        if bad.size:
            keep = x < 0.9 * x[bad[0]]
            x, f, fp, m, d = x[keep], f[keep], fp[keep], m[keep], d[keep]
    rho = rho_of(x, f, fp, m); N = 1 - 2 * m / x
    sig = -np.gradient(np.log(rho), np.log(x))
    Mtail = m - alpha * x           # m~ ~ alpha x + M: the remainder (core mass)
    deficit = 2 * alpha
    return dict(x=x, f=f, N=N, rho=rho, sig=sig, m=m, M_core=Mtail, N_inf=1 - deficit, x_end=s.t[-1], horizon=bool(s.t_events[0].size))


# ---------- (2) supercritical: regular horizon ----------
def from_horizon(xh, fh, alpha, x_m):
    rho_h = fh**2 / xh**2 + (fh**2 - 1) ** 2 / 4
    mp_h = alpha * xh**2 * rho_h
    Np_h = (1 - 2 * mp_h) / xh
    fp_h = (2 * fh / xh**2 + fh * (fh**2 - 1)) / Np_h
    eps = 1e-5 * xh
    y0 = [fh - eps * fp_h, fp_h, xh / 2 - eps * mp_h, 0.0]
    s = solve_ivp(rhs, (xh - eps, x_m), y0, args=(alpha,), rtol=1e-10, atol=1e-13, max_step=0.02)
    return s, dict(rho_h=rho_h, Np_h=Np_h, fp_h=fp_h, kappa_h=Np_h / 2, ind=2 * alpha * xh**2 * rho_h)


def match_super(alpha, guess):
    def res(p):
        a, xh, fh = p
        if xh <= 0.2 or a <= 0: return [1e3, 1e3, 1e3]
        x_m = 0.55 * xh
        sc = from_center(a, alpha, x_m)
        sh, _ = from_horizon(xh, fh, alpha, x_m)
        if not (sc.success and sh.success): return [1e3, 1e3, 1e3]
        return [sc.y[0, -1] - sh.y[0, -1], sc.y[1, -1] - sh.y[1, -1], sc.y[2, -1] - sh.y[2, -1]]
    p, info, ier, msg = fsolve(res, guess, full_output=True, xtol=1e-12)
    return p, np.max(np.abs(res(p))), ier


def sigma_at_horizon(xh, fh, alpha):
    """sigma = -d ln rho / d ln x at the horizon from nearby values (on the N > 0 side, interior)."""
    s, hd = from_horizon(xh, fh, alpha, xh * 0.98)
    x = s.t; f, fp, m = s.y[0], s.y[1], s.y[2]
    rho = rho_of(x, f, fp, m)
    k = np.argsort(x)
    sig = -np.gradient(np.log(rho[k]), np.log(x[k]))
    return float(sig[-1]), hd


def main():
    say("Field model of a defect: gravitating global monopole (hedgehog + Mexican hat), units delta_c = 1/(sqrt(lambda) eta), alpha = 4 pi eta^2")
    say("\n(1) Subcritical family: shoot on a = f'(0), f -> 1; slope shape sigma(x)")
    rows1 = []
    for alpha in (0.0, 0.05, 0.2, 0.4, 0.49):
        a, st = shoot_sub(alpha)
        R = analyse_sub(alpha, a)
        x, sig, rho, N = R["x"], R["sig"], R["rho"], R["N"]
        tail = x > 3.0
        i_max = int(np.argmax(sig)); s_max = float(sig[i_max]); x_smax = float(x[i_max])
        sig_min_after = float(np.min(sig[(x > x_smax)])) if np.any(x > x_smax) else np.nan
        # where does 2 alpha x^2 rho~ = 1 hold (8 pi r^2 rho = 1)?
        ind = 2 * alpha * x**2 * rho
        cross = x[np.nonzero(np.sign(ind[:-1] - 1) * np.sign(ind[1:] - 1) < 0)[0]]
        Mc = float(R["M_core"][-1])
        row = dict(alpha=alpha, a=a, status=st, sigma_max=s_max, x_sigma_max=x_smax, sigma_min_after_max=sig_min_after, sigma_tail=float(sig[-1]),
                   ind_max=float(ind.max()), ind_cross=[float(c) for c in cross], N_inf=R["N_inf"], N_min=float(N.min()), M_core=Mc, x_end=R["x_end"], horizon=R["horizon"])
        rows1.append(row)
        say(f"  alpha = {alpha:.2f}: a = {a:.6f} [{st}]; sigma: 0 -> max {s_max:.2f} at x = {x_smax:.2f} -> min after the max {sig_min_after:.3f} -> tail {row['sigma_tail']:.3f}; "
            f"8 pi r^2 rho: max {row['ind_max']:.3f}, crossings of level 1: {row['ind_cross']}; N_min = {row['N_min']:.3f}, N(inf) = {row['N_inf']:.2f}; core mass M~ = {Mc:+.3f} delta_c; horizon: {R['horizon']}")
    say("\n(2) Supercritical family (alpha > 1/2): regular non-degenerate horizon, two-sided shooting on (a, x_h, f_h)")
    rows2 = []
    guess = [0.35, 3.2, 0.75]
    for alpha in (0.55, 0.6, 0.7, 0.8, 1.0, 1.2):
        best = None
        for g in ([guess[0], np.sqrt(6 / alpha) * 0.95, guess[2]], [0.5, np.sqrt(6 / alpha) * 0.8, 0.6], [0.25, np.sqrt(6 / alpha), 0.9], [0.4, np.sqrt(6 / alpha) * 0.7, 0.4]):
            try:
                p, r, ier = match_super(alpha, g)
            except Exception:  # noqa: BLE001
                continue
            if r < 1e-7 and (best is None or r < best[1]):
                best = (p, r)
        if best is None:
            say(f"  alpha = {alpha:.2f}: matching not found (residual > 1e-7 for 4 initial guesses)")
            rows2.append(dict(alpha=alpha, found=False)); continue
        a, xh, fh = best[0]
        sig_h, hd = sigma_at_horizon(xh, fh, alpha)
        x_dbl = 1 / np.sqrt(1 - 1 / (2 * alpha)); f_dbl = np.sqrt(max(1 - 2 / x_dbl**2, 0))
        row = dict(alpha=alpha, found=True, a=a, x_h=xh, f_h=fh, resid=best[1], kappa_h=hd["kappa_h"], ind_h=hd["ind"], sigma_h=sig_h, fp_h=hd["fp_h"], x_double=x_dbl, f_double=f_dbl)
        rows2.append(row)
        guess = [a, xh, fh]
        say(f"  alpha = {alpha:.2f}: a = {a:.5f}, x_h = {xh:.4f} delta_c, f_h = {fh:.4f}, f'_h = {hd['fp_h']:+.3f}; residual {best[1]:.1e}; kappa_h = N'_h/2 = {hd['kappa_h']:+.4f} (1/delta_c); "
            f"8 pi r^2 rho|_h = {hd['ind']:.3f} (triple root: 1), sigma_h = {sig_h:.3f} (triple root: 2); analytic double root: x = {x_dbl:.3f}, f = {f_dbl:.3f}")
    say("\n(3) Analytics at the degenerate horizon (N = N' = 0, smooth field): the field equation gives 2f/x^2 + f(f^2-1) = 0 => f^2 = 1 - 2/x^2;"
        " 8 pi r^2 rho = 1 => 2 alpha (1 - 1/x^2) = 1 => x^2 = 1/(1 - 1/(2 alpha)) -- only for alpha > 1/2 and at the scale delta_c (a constant of the theory);"
        " N'' = 0 (triple root) => (r^2 rho)' = 2 r V = 0 => V(f_h) = 0 => f_h^2 = 1, incompatible with f^2 = 1 - 2/x^2: a triple root is impossible.")
    json.dump(dict(sub=rows1, sup=rows2), open(OUT / "defect_monopole.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "defect_monopole_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
