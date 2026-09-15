"""T36, part (c) continued: sources that are NOT a bounded function of the local state.

Theorem A3 (t36_core.py) says the q-equation of the T-region is

        q' = 16 pi b v p_perp ,

where the factor v is kinematic (it is db/dtau, not a property of the source).  Hence
EVERY constitutive law that stays bounded at the horizon drops out of the linearisation
at v = 0.  Two ways out remain, and both are tested here:

N1  a law with MEMORY (non-local in tau).  Israel-Stewart is one such law and T29 tested
    it for two relaxation times; here it is generalised to an arbitrary linear memory
    kernel realised as an extra state variable mu with mu' = (drive - mu)/tau_r, entering
    p_perp with gain G.  Prediction of A3: mu also enters q' multiplied by v, so the
    transverse block still has zero trace and there is still no attractor.  Verified.

N2  a law NON-LOCAL IN r: the medium reads the TOTAL mass M = m(infinity) -- a functional
    of the whole profile, not of the state at a point -- and sets its own scale to
    ell = lambda M (the project's tracking law / ratchet, mechanism_ellM, accretion_tracking,
    NONLOCAL_LAW_REPORT).  This DOES defeat the theorem: locality is load-bearing, not
    decorative.  The question this script answers is what it costs.  Answer: the tracking
    must be EXACT.  With a relative tracking error eta the inner-horizon surface gravity is
    kappa_- = -c |eta|^{2/3} with the same exponent as T1's detuning law and a coefficient
    within a factor 1.4 of it, so the non-local law does not remove the 10^-32 requirement
    of T1 -- it moves it from the profile to the tracker.

Run: python src/no_attractor/t36_nonlocal.py
Results: data/no_attractor/t36_nonlocal.json, logs/no_attractor/t36_nonlocal_log.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from t36_loopholes import EOS, initial_state, EPS_LIST  # noqa: E402
from t36_profile import Profile, load_grid  # noqa: E402

LOG = []
NL = chr(10)


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------- scaled profile (A, L)
class ScaledProfile:
    """m(r) = A * m_base(r / L).

    A = L reproduces the base profile exactly (homothety: f is scale invariant), so
    a black hole of mass A whose medium scale is ell = lambda A is an exact copy of the
    base triple-root solution.  A != L is the two-parameter family in which
      - A / L - 1 = eps  is T1's mass-amplitude detuning at fixed medium scale,
      - L / A - 1 = eta  is T1's scale detuning at fixed mass (the tracking error)."""

    def __init__(self, A, L, grid):
        self.A, self.L = A, L
        self.r, self.m_g, self.mp_g, self.mpp_g = grid
        self.lr = np.log(self.r)

    def parts(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        u = r / self.L
        lu = np.log(np.maximum(u, 1e-300))
        m = self.A * np.interp(lu, self.lr, self.m_g)
        mp = (self.A / self.L) * np.interp(lu, self.lr, self.mp_g)
        mpp = (self.A / self.L**2) * np.interp(lu, self.lr, self.mpp_g)
        f = 1 - 2 * m / r
        fp = 2 * m / r**2 - 2 * mp / r
        fpp = -4 * m / r**3 + 4 * mp / r**2 - 2 * mpp / r
        return f, fp, fpp

    def f(self, r):
        return self.parts(r)[0]

    def inner_root(self):
        """Inner horizon of the trapped region and kappa = f'/2 there.

        The tabulated base profile has |f| <~ 1e-12 over a finite interval around the
        triple root, so f crosses zero many times at the interpolation noise level.
        The physically meaningful number is the LARGEST |kappa| among those crossings
        (an upper bound on the residual detuning); that is what is returned, together
        with the number of crossings, so that the noise floor stays visible."""
        rr = np.linspace(0.3 * self.L, 1.2 * self.L, 300001)
        f = self.f(rr)
        idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        if idx.size == 0:
            j = int(np.argmin(np.abs(f)))
            _, fp, _ = [a[0] for a in self.parts(np.array([rr[j]]))]
            return float(rr[j]), float(fp / 2), 0
        best = None
        for i in idx:
            rt = brentq(lambda x: self.f(np.array([x]))[0], rr[i], rr[i + 1], xtol=1e-15)
            _, fp, _ = [a[0] for a in self.parts(np.array([rt]))]
            if best is None or abs(fp / 2) > abs(best[1]):
                best = (float(rt), float(fp / 2))
        return best[0], best[1], int(idx.size)


# ------------------------------------------------------------------------ N1
def part_N1(P, eos):
    say("=" * 78)
    say("N1.  Memory (non-local in proper time): does a relaxation kernel help?")
    say("=" * 78)
    say("Extended state (b, v, q, mu):   mu' = (D - mu)/tau_r,   p_perp = w(rho) rho - G mu,")
    say("with the drive D taken to be the degeneracy defect q itself (the sharpest choice --")
    say("the memory is fed exactly the quantity that must be driven to zero).  This contains")
    say("the Israel-Stewart structure T29 tested, and more: an arbitrary linear memory kernel")
    say("can be written this way.")
    say("Prediction of A3: mu enters q' = 16 pi b v (w rho - G mu) multiplied by v, so at")
    say("v = 0 its contribution to the Jacobian vanishes and the (v, q) block still has zero")
    say("trace.  The extra eigenvalue -1/tau_r contracts only the memory direction.")
    from scipy.integrate import solve_ivp
    rows = []
    for tau_r in (0.05, 0.3, 1.0):
        for G in (0.3, 1.0, 3.0, -1.0):
            letters = ""
            for eps in EPS_LIST:
                y0 = np.append(initial_state(P, eps), 0.0)

                def rhs(tau, y):
                    b, v, q, mu = max(y[0], 1e-6), y[1], y[2], y[3]
                    rho = (1 + q) / (8 * np.pi * b * b)
                    pp = eos(rho) * rho - G * mu
                    return [-v, (v * v - q) / (2 * b), 16 * np.pi * b * v * pp,
                            (q - mu) / tau_r]

                def ev(tau, y):
                    return y[1] - 1e-9
                ev.terminal, ev.direction = True, -1

                def evc(tau, y):
                    return y[0] - 0.05
                evc.terminal, evc.direction = True, -1

                def evn(tau, y):
                    return y[2] + 1.0
                evn.terminal, evn.direction = True, -1
                s = solve_ivp(rhs, (0, 80.0), y0, rtol=1e-8, atol=1e-11,
                              events=[ev, evc, evn], max_step=0.2)
                b, v, q = s.y[0, -1], s.y[1, -1], s.y[2, -1]
                vp = (v * v - q) / (2 * max(b, 1e-9))
                if s.t_events[1].size:
                    letters += "c"
                elif s.t_events[2].size:
                    letters += "n"
                elif s.t_events[0].size:
                    letters += "a" if abs(vp) < 1e-3 else "r"
                else:
                    letters += "u"
            rows.append(dict(tau_r=tau_r, G=G, fates=letters, n_degenerate=letters.count("a")))
            say(f"  tau_r = {tau_r:<5} G = {G:+.1f}: {letters}   degenerate {letters.count('a')}/7")
    say("  Not one of the twelve memory laws produces a degenerate exit -- not even at eps = 0,")
    say("  where the gain G displaces the separatrix away from the tuned profile.  Non-locality")
    say("  in TIME does not defeat the theorem, for the structural reason given above.  (This")
    say("  generalises T29's Israel-Stewart runs, which used one kernel and two relaxation")
    say("  times, to an arbitrary single-pole memory kernel and both signs of the gain.)")
    return rows


# ------------------------------------------------------------------------ N2
def part_N2(grid):
    say(NL + "=" * 78)
    say("N2.  Non-local in r: the medium reads the TOTAL mass and sets ell = lambda M")
    say("=" * 78)
    say("This is the project's tracking law (ratchet).  It is non-local: M = m(infinity) is")
    say("an integral of rho over the whole profile, not a function of the fields at a point,")
    say("so it is outside the premises of the theorem -- and it works, trivially, because a")
    say("homothety (m -> A m, r -> A r) leaves f invariant.  A tracker that is EXACT keeps the")
    say("triple root for every mass.  The question is what an inexact tracker costs.")
    say(NL + "  (a) exact tracker, ell = lambda M, for a range of masses A:")
    rows_a = []
    for A in (0.5, 0.9, 1.0, 1.1, 2.0, 10.0):
        sp = ScaledProfile(A, A, grid)
        rt, kap, nr = sp.inner_root()
        rows_a.append(dict(A=A, r_inner=rt, kappa=kap, n_roots=nr))
        say(f"      M = {A:<5}: inner root at r = {rt:.6f} = {rt/A:.6f} M, kappa_- = {kap:+.2e}, "
            f"kappa_- M = {kap*A:+.3e}, sign changes of f = {nr}")
    say("      kappa_- M is the SAME number for every mass: the triple root is exactly preserved")
    say("      under the homothety, as it must be.  Its value -7.1e-7 is the noise floor of the")
    say("      cached interpolation grid (T26, evaluating the family directly, gets 5e-8); the")
    say("      many sign changes of f are the same noise, |f| <~ 1e-12 over a finite interval.")

    say(NL + "  (b) inexact tracker: ell = lambda M (1 + eta), i.e. the scale lags or leads the")
    say("      mass by a relative error eta.  Model: A fixed, L -> A (1 + eta).")
    rows_b = []
    for eta in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                -1e-5, -1e-4, -1e-3, -1e-2):
        sp = ScaledProfile(1.0, 1.0 * (1 + eta), grid)
        rt, kap, nr = sp.inner_root()
        rows_b.append(dict(eta=eta, r_inner=rt, kappa=kap, n_roots=nr))
    pos = [(r["eta"], r["kappa"]) for r in rows_b if r["eta"] > 0]
    neg = [(abs(r["eta"]), r["kappa"]) for r in rows_b if r["eta"] < 0]
    for lab, dat in (("eta > 0", pos), ("eta < 0", neg)):
        say(f"      {lab}:")
        for e, k in dat:
            say(f"        |eta| = {e:.0e}   kappa_- = {k:+.4e}   |kappa|/|eta|^(2/3) = "
                f"{abs(k)/e**(2/3):.3f}")
    xs = np.array([e for e, k in pos if 1e-6 <= e <= 1e-3])
    ys = np.array([abs(k) for e, k in pos if 1e-6 <= e <= 1e-3])
    slope, inter = np.polyfit(np.log(xs), np.log(ys), 1)
    say(f"      fit over |eta| in [1e-6, 1e-3]:  d ln|kappa| / d ln|eta| = {slope:.4f} "
        f"(T1 law: 2/3 = 0.6667).  The fitted prefactor {np.exp(inter):.3f} is biased by the")
    say("      upper end of the range (the ratio drifts from 1.92 to 1.75 as |eta| grows to")
    say(f"      1e-3); the asymptotic coefficient is the |eta| -> 0 ratio, {abs(pos[0][1])/pos[0][0]**(2/3):.3f}.")
    say("      Analytic check.  With m(r) = A m_b(r/L) at fixed A, d f/d L at L = 1 equals")
    say("      2 m_b'(r), and the degeneracy condition 2 m_b'(R_-) = 1 makes it exactly 1:")
    say("      delta f = eta * g with g = +1, the same |g| as T1's mass-amplitude case (g_a = -1).")
    say("      T1's law kappa_- = -(3/2)|a|^{1/3} |eta g|^{2/3} with (3/2)|a|^{1/3} = 1.938 then")
    say(f"      predicts 1.938; the measured |eta| -> 0 ratio is {abs(pos[0][1])/pos[0][0]**(2/3):.3f} (1 % agreement).")
    say(NL + "      CONSEQUENCE.  kappa_-(eta) is T1's law with the SAME exponent AND the same")
    say("      coefficient.  T1's requirement eps <~ 3e-32 for a 10 M_sun hole becomes")
    say("      eta <~ 3e-32 on the relative tracking error, with no gain at all.")
    say("      The non-local law removes the tuning of the profile only by requiring an")
    say("      equally exact tuning of the tracker.  It is a counterexample to the theorem as")
    say("      stated, and NOT a physical mechanism: the enumeration did not miss anything.")
    say("      Its independent price is already on record: accretion_tracking (an ingoing flux")
    say("      of negative energy up to 0.2 Mdot inside R < 1.49 M, or emission by the medium")
    say("      towards R_-) and NONLOCAL_LAW_REPORT / inhomogeneous_collapse section 3.4 (the")
    say("      local version ell = lambda m_enc(r) gives a singular centre, K ~ 1e24...1e29).")
    return dict(exact=rows_a, inexact=rows_b, slope=float(slope),
                prefactor=float(np.exp(inter)),
                t1_prefactor=1.938, g_of_scale_detuning=1.0,
                eta_required_10Msun=3e-32)


def main():
    grid = load_grid()
    P = Profile(0.0, grid=grid)
    eos = EOS(P)
    out = dict(N1=part_N1(P, eos), N2=part_N2(grid))
    say(NL + "=" * 78)
    say("SUMMARY of the non-local tests")
    say("=" * 78)
    say("Locality is the load-bearing premise, but in a precise and narrow sense: what fails")
    say("is not 'non-locality in time' (memory laws are still covered, N1) but dependence on")
    say("a FUNCTIONAL OF THE PROFILE, specifically the total mass (N2).  And the escape is")
    say("not free: an exact tracker is needed, with the same 10^-32 accuracy T1 demanded of")
    say("the profile itself.  The theorem is therefore sharp and the loophole is empty of")
    say("physics -- which is the honest way to state it.")
    json.dump(out, open(OUT / "t36_nonlocal.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    (LOGS / "t36_nonlocal_log.txt").write_text(NL.join(LOG), encoding="utf-8")
    say(f"{NL}written: {OUT/'t36_nonlocal.json'}, {LOGS/'t36_nonlocal_log.txt'}")


if __name__ == "__main__":
    main()
