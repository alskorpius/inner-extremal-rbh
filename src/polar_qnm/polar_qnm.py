"""Polar sector of the quasinormal modes for the NED realization of the scenario profile - following the Moreno-Sarbach system (gr-qc/0208090, (45)-(46)):
    (d_t^2 - d_r*^2) u + N V u = 0,  u = (Psi, Phi),  V = [[V11, V12], [V12, V22]],
    V11 = [l(l+1) lam - 2 N lam + a r Delta r]/(r^2 (a+lam)) + 2 N lam b/(r^2 (a+lam)^2),
    V22 = kappa l(l+1)/r^2 + 4 H_P Q^2/(r^4 (a+lam)) (lam + 1 - N - 2 r^2 H + 4 N kappa) + H_P^{-1/2} Delta H_P^{1/2} + 8 N H_P Q^2 b/(r^4 (a+lam)^2),
    V12 = -sqrt(4 lam H_P) Q W / r^3,  W = (lam + 1 - N - 2 r^2 H + 2 N kappa)/(a+lam) + 2 N b/(a+lam)^2,
    a = 6m/r + 2 r^2 H, b = lam + 4 H_P Q^2/r^2, lam = (l-1)(l+2), r Delta r = 1 - N + 2 r^2 H, N = 1 - 2m/r,
    H = -m'/r^2 (= -L), H_P = L_F, kappa = 1 + 2 F L_FF/L_F; H_P Q^2 and sqrt(H_P) Q do not depend on Q.
Control: Q -> 0 (Schwarzschild) - V11 = Zerilli potential / N (checked), Phi decouples with the Maxwell potential.
Method: time evolution (leapfrog, 2nd order) with a Gaussian pulse, omega extracted by fitting a damped sinusoid;
the scenario shift relative to Schwarzschild is a difference within one code. Run: python src/polar_qnm/polar_qnm.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit, brentq

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


class Background:
    """m(r), m'(r), m''(r), sigma for r > r_h; Schwarzschild when scen=None."""

    def __init__(self, scen=None, M=1.0):
        self.scen, self.M = scen, M
        if scen is None:
            self.rh = 2 * M
        else:
            hz = lb.static_horizons(scen, M)
            self.rh = hz[-1][0]

    def fields(self, r):
        if self.scen is None:
            z = np.zeros_like(r)
            return dict(m=self.M + z, mp=z, mpp=z, sigma=z + 1.0, sigp=z)
        d = self.scen.all(self.M, r)
        R1 = (2 * self.M * self.scen.ell**2 / (3 * self.scen.Itot)) ** (1 / 3)
        lu = np.log(r / R1)
        sig = np.interp(lu, self.scen.LU, self.scen.sig)
        sigp = np.interp(lu, self.scen.LU, np.gradient(self.scen.sig, self.scen.LU))
        return dict(m=d["m"], mp=d["m_R"], mpp=d["m_RR"], sigma=sig, sigp=sigp)


def potential_matrix(bg, r, l):
    f = bg.fields(r)
    m, mp, mpp, sig, sigp = f["m"], f["mp"], f["mpp"], f["sigma"], f["sigp"]
    lam = (l - 1) * (l + 2)
    N = 1 - 2 * m / r
    H = -mp / r**2
    rDr = 1 - N + 2 * r**2 * H
    a = 6 * m / r + 2 * r**2 * H
    HPQ2 = r**2 * (2 * mp - r * mpp) / 2          # H_P Q^2 = -(r^5/2) dL/dr, L = m'/r^2
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(sig > 1e-12, sig / 2 - 1 - sigp / (2 * np.maximum(sig, 1e-300)), 1.0)
    if bg.scen is None:
        kappa = np.ones_like(r)
    b = lam + 4 * HPQ2 / r**2
    V11 = (l * (l + 1) * lam - 2 * N * lam + a * rDr) / (r**2 * (a + lam)) + 2 * N * lam * b / (r**2 * (a + lam) ** 2)
    W = (lam + 1 - N - 2 * r**2 * H + 2 * N * kappa) / (a + lam) + 2 * N * b / (a + lam) ** 2
    V12 = -np.sqrt(4 * lam * np.maximum(HPQ2, 0.0)) * W / r**3
    # H_P^{-1/2} Delta H_P^{1/2} (static part): H_P^{-1/2} d_r (N d_r H_P^{1/2}); H_P proportional to HPQ2 (Q fixed)
    if bg.scen is None:
        term = np.zeros_like(r)
    else:
        sq = np.sqrt(np.maximum(HPQ2, 1e-300))
        dsq = np.gradient(sq, r)
        term = np.gradient(N * dsq, r) / sq
    V22 = kappa * l * (l + 1) / r**2 + 4 * HPQ2 / (r**4 * (a + lam)) * (lam + 1 - N - 2 * r**2 * H + 4 * N * kappa) + term + 8 * N * HPQ2 * b / (r**4 * (a + lam) ** 2)
    return N, V11, V12, V22, dict(kappa=kappa, HPQ2=HPQ2, a=a)


def zerilli(r, M, l):
    lz = (l - 1) * (l + 2) / 2
    N = 1 - 2 * M / r
    return N * (2 * lz**2 * (lz + 1) * r**3 + 6 * lz**2 * M * r**2 + 18 * lz * M**2 * r + 18 * M**3) / (r**3 * (lz * r + 3 * M) ** 2)


def tortoise_grid(bg, rs_min=-80.0, rs_max=400.0, drs=0.05):
    """r(r*) by integrating dr/dr* = N; start from r = r_h (1 + 1e-4) with r* per the Schwarzschild formula."""
    rh = bg.rh
    r = np.geomspace(rh * (1 + 1e-8), 2000.0, 400001)
    N = 1 - 2 * bg.fields(r)["m"] / r
    rs = cumulative_trapezoid(1 / N, r, initial=0.0)
    # normalization: at large r, r* ~ r + 2M ln(r/2M - 1) (M = 1)
    rs += (r[-1] + 2 * np.log(r[-1] / 2 - 1)) - rs[-1]
    grid = np.arange(rs_min, rs_max, drs)
    r_of = np.interp(grid, rs, r)
    return grid, r_of


def evolve(bg, l, drs=0.05, rs_min=-80.0, rs_max=400.0, t_end=320.0, r_obs=40.0, sigma0=1.5, r0=20.0):
    grid, r = tortoise_grid(bg, rs_min, rs_max, drs)
    N, V11, V12, V22, info = potential_matrix(bg, r, l)
    U11, U12, U22 = N * V11, N * V12, N * V22
    dt = 0.5 * drs
    nsteps = int(t_end / dt)
    iobs = int(np.argmin(np.abs(grid - r_obs)))
    # initial data: Gaussian pulse in Psi, Phi = 0
    psi = np.exp(-((grid - r0) ** 2) / (2 * sigma0**2)); phi = np.zeros_like(grid)
    psi_prev = psi.copy(); phi_prev = phi.copy()
    lap = lambda u: (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / drs**2
    ts, ys, ps = [], [], []
    for n in range(nsteps):
        lp, lf = lap(psi), lap(phi)
        psi_new = 2 * psi - psi_prev + dt**2 * (lp - U11 * psi - U12 * phi)
        phi_new = 2 * phi - phi_prev + dt**2 * (lf - U12 * psi - U22 * phi)
        psi_new[0] = psi_new[-1] = 0.0; phi_new[0] = phi_new[-1] = 0.0
        psi_prev, psi = psi, psi_new; phi_prev, phi = phi, phi_new
        ts.append((n + 1) * dt); ys.append(psi[iobs]); ps.append(phi[iobs])
    return np.array(ts), np.array(ys), np.array(ps), info, (grid, r, U11, U12, U22)


def matrix_pencil(t, y, t1, t2, order=8):
    """Matrix pencil method: poles z_k = exp(-i omega_k dt); returns the dominant mode (max. amplitude, Re omega > 0)."""
    k = (t > t1) & (t < t2)
    tt, yy = t[k], y[k]
    dt = tt[1] - tt[0]
    Nn = len(yy); L = Nn // 3
    Y = np.array([yy[i:i + L + 1] for i in range(Nn - L)])
    U, s_, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.conj().T[:, :order]
    V1, V2 = V[:-1, :], V[1:, :]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    om = 1j * np.log(z) / dt          # y ~ sum A exp(-i om t): om = wr - i|wi| ... sign: exp(-i om t) = exp(-i wr t) exp(-wi_abs t)
    # amplitudes
    Z = np.array([z**n for n in range(Nn)])
    A = np.linalg.lstsq(Z, yy, rcond=None)[0]
    idx = [i for i in range(len(om)) if om[i].real > 0.05 and om[i].imag < 0]
    if not idx:
        return np.nan, np.nan
    i = max(idx, key=lambda j: abs(A[j]))
    return float(om[i].real), float(om[i].imag)


def fit_qnm(t, y, t1, t2):
    k = (t > t1) & (t < t2)
    tt, yy = t[k], y[k]
    def model(t, A, wr, wi, ph):
        return A * np.exp(wi * (t - t1)) * np.cos(wr * (t - t1) + ph)
    # initial guess from zero crossings
    zc = tt[np.nonzero(np.sign(yy[:-1]) * np.sign(yy[1:]) < 0)[0]]
    wr0 = np.pi / np.mean(np.diff(zc)) if len(zc) > 3 else 0.37
    p, _ = curve_fit(model, tt, yy, p0=[yy[0] if abs(yy[0]) > 0 else 1e-3, wr0, -0.09, 0.0], maxfev=20000)
    return p[1], p[2]


DRS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05


def main():
    say("Polar QNMs (Moreno-Sarbach system) - time evolution, l = 2, 3.")
    scen = lb.ScenarioMass()
    bgS, bgA = Background(None), Background(scen)
    # potential control: Q -> 0 => V11 N = Zerilli
    r = np.linspace(2.05, 60, 2000)
    for l in (2, 3):
        N, V11, V12, V22, _ = potential_matrix(bgS, r, l)
        say(f"  Zerilli control l={l}: max |N V11 / V_Z - 1| = {np.max(np.abs(N*V11/zerilli(r, 1.0, l) - 1)):.2e}; V12 = {np.max(np.abs(V12)):.1e}")
    # scenario: coupling magnitude and potentials outside the horizon
    rA = np.linspace(bgA.rh * 1.001, 60, 4000)
    NA, V11A, V12A, V22A, info = potential_matrix(bgA, rA, 2)
    NS, V11S, _, _, _ = potential_matrix(bgS, rA, 2)
    say(f"  scenario l=2: r_h = {bgA.rh:.5f}; max |V12|/|V11| outside the horizon = {np.max(np.abs(V12A)/np.abs(V11A)):.2e}; max |N V11 (A) - N V11 (S)| = {np.max(np.abs(NA*V11A - NS*V11S)):.2e} at r = {rA[np.argmax(np.abs(NA*V11A - NS*V11S))]:.3f}; "
        f"kappa outside the horizon from {info['kappa'].min():.2f} to {info['kappa'].max():.2f}; H_P Q^2/r^2 at r = 3: {np.interp(3.0, rA, info['HPQ2']/rA**2):.2e}")
    rows = []
    for l in (2, 3):
        res = {}
        for tag, bg in (("S", bgS), ("A", bgA)):
            t, y, p, info, _ = evolve(bg, l, drs=DRS)
            np.savez_compressed(OUT / f"series_l{l}_{tag}_drs{DRS}.npz", t=t, y=y, p=p)
            wr, wi = matrix_pencil(t, y, 80.0, 200.0)
            wr2, wi2 = fit_qnm(t, y, 80.0, 200.0)
            res[tag] = (wr, wi)
            say(f"  l={l} {tag} (drs={DRS}): omega = {wr:.5f} {wi:+.5f} i  (matrix pencil, t in [80,200]; sinusoid fit: {wr2:.5f} {wi2:+.5f} i); |Phi|_max/|Psi|_max = {np.max(np.abs(p))/np.max(np.abs(y)):.2e}")
        known = {2: (0.37367, -0.08896), 3: (0.59944, -0.09270)}[l]
        say(f"    Schwarzschild l={l}: known value (Leaver) {known[0]} {known[1]:+}i; method error: dRe = {res['S'][0]/known[0]-1:+.2e}, dIm = {res['S'][1]/known[1]-1:+.2e}")
        say(f"    scenario shift (polar sector): dRe/Re = {res['A'][0]/res['S'][0]-1:+.2e}, dIm/Im = {res['A'][1]/res['S'][1]-1:+.2e}")
        rows.append(dict(l=l, schw=res["S"], scen=res["A"], known=known, d_re=res["A"][0] / res["S"][0] - 1, d_im=res["A"][1] / res["S"][1] - 1))
    json.dump(rows, open(OUT / "polar_qnm.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    (LOGS / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT}")


if __name__ == "__main__":
    main()
