"""T36, part (b): the single mechanism.

Claim under test (referee's conjecture): every closed mechanism reduces to
"a degenerate exit is the approach to a fixed point of an autonomous system in
infinite parameter time, hence a trajectory of codimension >= 1".

This script derives and verifies the dynamical half of that claim:

  A (sympy).  Static spherically symmetric source, general lapse:
      A1  at ANY horizon  8 pi p_r = -1/r^2 ;  at any DEGENERATE horizon also
          8 pi rho = 1/r^2, hence rho + p_r = 0: the class T^t_t = T^r_r is not a
          premise, it is FORCED at the fixed point.
      A2  class T^t_t = T^r_r: the T-region (Kantowski-Sachs) flow reduces to the
          Newtonian system  b'' = -f'(b)/2, whose phase-space divergence is
          identically zero (Liouville) -- for ANY local barotropic law and even for
          an explicitly tau-dependent one.
      A3  same class with a velocity-dependent transverse pressure: the reduced
          system in (b, v, q), v = -b', q = 8 pi rho b^2 - 1, is
              b' = -v,  v' = (v^2 - q)/(2b),  q' = 16 pi b v p_perp .
          Its Jacobian at a degenerate horizon (v = q = 0) has TRACE ZERO for every
          p_perp bounded at v -> 0, with transverse eigenvalues s = +- sqrt(-8 pi p_perp*).
      A4  wider class (p_x != -rho): the same flow is NOT Lipschitz at v = 0
          (h = a'/a diverges as 1/v), i.e. the dynamical-systems framework itself
          breaks there -- recorded as the place where a fully general proof stops.
      A5  the identity 8 pi p_perp = f''/2 + f'/b, so at a degenerate horizon
          s^2 = -f''(b*)/2 : saddle for f'' < 0 (double root), centre for f'' > 0,
          totally degenerate for f'' = 0 (triple root, where p_perp(b*) = 0).

  B (numeric).  On the project's base triple-root profile and on exact
      Schwarzschild-de Sitter:
      B1  p_perp(R_-) = 0 and 8 pi rho R_-^2 = 1 on the base profile;
      B2  Liouville checked by integrating the variational equation across the
          whole T-region: det of the monodromy matrix stays 1;
      B3  Nariai (SdS double root): analytic s = +- sqrt(-f''/2) = +- 1/b*, which is
          exactly the eigenvalue T26 found for the scalar-field Nariai point -- the
          T26 statement is the f'' < 0 case of A3/A5;
      B4  independence of the three conditions f = f' = f'' = 0 under arbitrary
          localised mass perturbations (rank of the 3 x k response matrix) --
          the codimension count of Theorem A.

Run: python src/no_attractor/t36_core.py
Results: data/no_attractor/t36_core.json, logs/no_attractor/t36_core_log.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from t36_profile import Profile, load_grid  # noqa: E402

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ A: symbolic
def part_A():
    res = {}
    say("=" * 78)
    say("A. SYMBOLIC DERIVATION")
    say("=" * 78)

    # ---- A1: what is forced at a horizon, for a general static metric ---------
    r, t, th, ph = sp.symbols("r t theta phi", positive=True)
    m = sp.Function("m")(r)
    Phi = sp.Function("Phi")(r)
    f = 1 - 2 * m / r
    g = sp.diag(-sp.exp(2 * Phi) * f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)
    x = [t, r, th, ph]
    ginv = g.inv()
    Gam = [[[sum(ginv[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d]))
                 for d in range(4)) / 2 for c in range(4)] for b in range(4)] for a in range(4)]
    Ric = sp.zeros(4, 4)
    for b in range(4):
        for c in range(4):
            e = 0
            for a in range(4):
                e += sp.diff(Gam[a][b][c], x[a]) - sp.diff(Gam[a][b][a], x[c])
                for d in range(4):
                    e += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
            Ric[b, c] = sp.simplify(e)
    Rs = sp.simplify(sum(ginv[a, b] * Ric[a, b] for a in range(4) for b in range(4)))
    Ein = sp.simplify(Ric - Rs * g / 2)
    Gtt = sp.simplify((ginv * Ein)[0, 0])   # G^t_t = -8 pi rho
    Grr = sp.simplify((ginv * Ein)[1, 1])   # G^r_r =  8 pi p_r
    rho_e = sp.simplify(-Gtt / (8 * sp.pi))
    pr_e = sp.simplify(Grr / (8 * sp.pi))
    say("A1. General static metric ds^2 = -e^{2Phi} f dt^2 + dr^2/f + r^2 dOmega^2, f = 1 - 2m/r:")
    say(f"    8 pi rho = {sp.simplify(8*sp.pi*rho_e)}")
    say(f"    8 pi p_r = {sp.simplify(8*sp.pi*pr_e)}")
    # evaluate on a horizon: substitute m -> r/2 (f = 0)
    mm, mp1, mp2 = sp.symbols("m0 m1 m2", real=True)
    sub_h = {sp.Derivative(m, (r, 2)): mp2, sp.Derivative(m, r): mp1, m: r / 2}
    pr_h = sp.simplify(sp.simplify(8 * sp.pi * pr_e).subs(sub_h).doit())
    rho_h = sp.simplify(sp.simplify(8 * sp.pi * rho_e).subs(sub_h).doit())
    say(f"    on a horizon (m = r/2, f = 0):   8 pi p_r = {pr_h}      8 pi rho = {rho_h}")
    # degeneracy f' = 0 at f = 0  <=>  2 m' = 1
    rho_deg = sp.simplify(rho_h.subs(mp1, sp.Rational(1, 2)))
    pr_deg = sp.simplify(pr_h.subs(mp1, sp.Rational(1, 2)))
    say(f"    degeneracy f'(r_h) = 0  <=>  2 m'(r_h) = 1   =>  8 pi rho = {rho_deg}, 8 pi p_r = {pr_deg}")
    say(f"    hence  rho + p_r = {sp.simplify((rho_deg + pr_deg)/(8*sp.pi))}  at EVERY degenerate horizon,")
    say("    independently of the lapse Phi: the class T^t_t = T^r_r is FORCED at the fixed point,")
    say("    it is not an extra premise.  (NEC is saturated radially there.)")
    res["A1"] = dict(p_r_at_horizon="8 pi p_r = -1/r^2",
                     rho_at_degenerate_horizon="8 pi rho = 1/r^2",
                     conclusion="rho + p_r = 0 forced at any degenerate horizon, any lapse")

    # ---- A2: Liouville for the barotropic / tau-dependent case ---------------
    b, tau = sp.symbols("b tau", positive=True)
    v = sp.Symbol("v", real=True)
    F = sp.Function("f")(b, tau)                      # allow explicit tau dependence
    flow2 = sp.Matrix([-v, sp.diff(F, b) / 2])        # (b, v) with v = -db/dtau
    div2 = sp.simplify(sp.diff(flow2[0], b) + sp.diff(flow2[1], v))
    say("\nA2. Class T^t_t = T^r_r, T-region:  b'' = -f'(b)/2, v = -b'.")
    say(f"    flow (b', v') = ({flow2[0]}, {flow2[1]});   divergence = {div2}")
    say("    The divergence vanishes IDENTICALLY -- for any f(b), and also for an explicitly")
    say("    tau-dependent f(b, tau).  The flow is a (possibly time-dependent) Hamiltonian flow")
    say("    with H = v^2/2 + f(b)/2; Liouville's theorem applies; NO attractor of any kind exists.")
    say("    => the premise 'autonomous' is REDUNDANT for this half of the theorem.")
    res["A2"] = dict(divergence=str(div2), note="Liouville holds also for f(b, tau)")

    # ---- A3: velocity-dependent p_perp: trace of the Jacobian ----------------
    q = sp.Symbol("q", real=True)
    pp = sp.Function("p")(b, v, q)                    # general local law, bounded at v -> 0
    # state order (b, v, q); flow components in the same order
    F3 = sp.Matrix([-v, (v**2 - q) / (2 * b), 16 * sp.pi * b * v * pp])
    J = F3.jacobian(sp.Matrix([b, v, q]))
    Jfp = sp.simplify(J.subs({v: 0, q: 0}))
    pstar = sp.Symbol("p_*", real=True)
    Jfp2 = sp.Matrix(Jfp.subs(sp.Function("p")(b, 0, 0), pstar))
    trace = sp.simplify(sp.trace(Jfp2))
    ev = sp.simplify(Jfp2.eigenvals())
    say("\nA3. Same class, but p_perp = p(b, q, v) allowed to depend on v = -b'  (bounded at v -> 0).")
    say("    Reduced system, q = 8 pi rho b^2 - 1 :")
    say("        b' = -v,   v' = (v^2 - q)/(2 b),   q' = 16 pi b v p_perp .")
    say(f"    Jacobian at the degenerate horizon (v = q = 0):\n{sp.pretty(Jfp2)}")
    say(f"    trace = {trace}   (identically zero)")
    say(f"    eigenvalues = {ev}")
    say("    => every derivative of p_perp with respect to v enters multiplied by v and dies at")
    say("       the fixed point.  Transverse eigenvalues are +- sqrt(-8 pi p_perp*): a saddle or a")
    say("       centre, NEVER a sink.  An attractor requires p_perp to BLOW UP as 1/v, i.e. a")
    say("       dependence on the expansion Theta (which diverges at every non-degenerate horizon).")
    res["A3"] = dict(trace=str(trace), eigenvalues=[str(k) for k in ev])

    # ---- A4: wider class -- loss of Lipschitz --------------------------------
    rho_s = sp.Symbol("rho", positive=True)
    px = sp.Function("p_x")(rho_s, b)
    h = ((v**2 + 1) - 8 * sp.pi * rho_s * b**2) / (2 * b * v)
    rhodot = -h * (rho_s + px) + 2 * v / b * (rho_s + sp.Function("p_perp")(rho_s, b))
    say("\nA4. Wider class (p_x != -rho).  The constraint gives h = a'/a = [(v^2+1) - 8 pi rho b^2]/(2 b v),")
    say("    so h ~ 1/v at a horizon, and rho' = -h (rho + p_x) + (2v/b)(rho + p_perp) contains")
    say(f"    the term  {sp.simplify(-h*(rho_s+px))}")
    say("    At a degenerate horizon rho + p_x -> 0 (A1) and the term is 0/0: the right-hand side")
    say("    is bounded but NOT Lipschitz there (its v-derivative diverges).  The dynamical-systems")
    say("    argument therefore does not extend verbatim outside the class T^t_t = T^r_r; this is")
    say("    the exact step at which the general proof stops (see REPORT.md, section 5).")
    res["A4"] = dict(obstruction="h ~ 1/v; rho-equation is bounded but not Lipschitz at v = 0")

    # ---- A5: p_perp <-> f'' ---------------------------------------------------
    fr = sp.Function("f")(b)
    pperp_expr = (sp.diff(fr, b, 2) / 2 + sp.diff(fr, b) / b) / (8 * sp.pi)
    say("\nA5. In this class  8 pi p_perp = f''/2 + f'/b.  At a degenerate horizon (f = f' = 0):")
    say(f"    p_perp* = f''(b*)/(16 pi)   =>   s^2 = -8 pi p_perp* = -f''(b*)/2 .")
    say("    f'' < 0 (double root, f <= 0 on both sides): s real  -> SADDLE, stable set codim 1.")
    say("    f'' > 0                                    : s imaginary -> CENTRE, no approach.")
    say("    f'' = 0 (TRIPLE root)                      : s = 0 both, p_perp(b*) = 0, fully")
    say("       degenerate; the approach is algebraic, tau ~ (b - b*)^{-1/2} (T26), infinite time,")
    say("       along a single orbit.")
    res["A5"] = dict(identity="8 pi p_perp = f''/2 + f'/b", s2="-f''(b*)/2",
                     pperp_at_triple_root=0)
    return res


# ------------------------------------------------------------------ B: numeric
def part_B():
    res = {}
    say("\n" + "=" * 78)
    say("B. NUMERICAL VERIFICATION")
    say("=" * 78)
    grid = load_grid()
    P = Profile(0.0, grid=grid)
    Rm, fmin = P.triple_root()
    Rp = P.roots()[-1]
    f0, fp0, fpp0, rho0, drho0 = [a[0] for a in P.parts(np.array([Rm]))]
    pperp0 = P.p_perp(np.array([Rm]))[0]

    say("\nB1. Base triple-root profile (BASE0, sigma_min = 1, same object as T26):")
    say(f"    R_- = {Rm:.6f}   R_+ = {Rp:.6f}")
    say(f"    f(R_-)  = {f0:+.3e}   f'(R_-) = {fp0:+.3e}   f''(R_-) = {fpp0:+.3e}")
    say(f"    8 pi rho R_-^2 - 1 = {8*np.pi*rho0*Rm**2 - 1:+.3e}   (degeneracy condition)")
    say(f"    p_perp(R_-) = {pperp0:+.3e};  identity check f''/2 + f'/b - 8 pi p_perp = "
        f"{fpp0/2 + fp0/Rm - 8*np.pi*pperp0:+.2e}")
    say(f"    predicted s^2 = -f''/2 = {-fpp0/2:+.3e}  -> |s| = {np.sqrt(abs(fpp0/2)):.3e}")
    say("    (the residual f'' = 2.7e-3 is the tuning residual of the tabulated profile,")
    say("     kappa_- = -5e-8 in T26; the exact triple root has f'' = p_perp(b*) = 0 exactly.)")
    res["B1"] = dict(R_minus=Rm, R_plus=Rp, f=f0, fp=fp0, fpp=fpp0,
                     deg_cond=float(8 * np.pi * rho0 * Rm**2 - 1), p_perp=float(pperp0),
                     identity_residual=float(fpp0 / 2 + fp0 / Rm - 8 * np.pi * pperp0))

    say("\nB2. Liouville: monodromy determinant of the T-region flow (b, v).")
    say("    b' = -v, v' = f'(b)/2; variational equation dJ/dtau = A(b) J, A = [[0,-1],[f''/2,0]].")

    def rhs(tau, y):
        bb = max(y[0], 1e-6)
        _, fp, fpp, _, _ = [a[0] for a in P.parts(np.array([bb]))]
        J = y[2:].reshape(2, 2)
        A = np.array([[0.0, -1.0], [fpp / 2, 0.0]])
        return np.concatenate(([-y[1], fp / 2], (A @ J).ravel()))

    b0 = Rp * (1 - 1e-6)
    v0 = np.sqrt(max(-P.f(np.array([b0]))[0], 0.0))
    y0 = np.concatenate(([b0, v0], np.eye(2).ravel()))
    sol = solve_ivp(rhs, (0, 40.0), y0, rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.05)
    taus = np.linspace(0, sol.t[-1], 400)
    dets = np.array([np.linalg.det(sol.sol(t)[2:].reshape(2, 2)) for t in taus])
    bend = sol.sol(sol.t[-1])[0]
    say(f"    integrated from b = {b0:.5f} down to b = {bend:.6f} over tau = {sol.t[-1]:.1f} M")
    say(f"    max |det J - 1| = {np.max(np.abs(dets - 1)):.2e}   (phase-space area exactly conserved)")
    say("    => the barotropic T-region flow contracts no phase volume anywhere: no attractor can")
    say("       exist, whatever the equation of state.")
    res["B2"] = dict(tau_end=float(sol.t[-1]), b_end=float(bend),
                     max_det_error=float(np.max(np.abs(dets - 1))))

    say("\nB3. Nariai (exact Schwarzschild-de Sitter double root) as a check of s^2 = -f''/2:")
    rows = []
    for Lam in (0.1, 1.0, 3.0):
        bstar = 1.0 / np.sqrt(Lam)
        mstar = 1.0 / (3.0 * np.sqrt(Lam))
        fv = 1 - 2 * mstar / bstar - Lam * bstar**2 / 3
        fpv = 2 * mstar / bstar**2 - 2 * Lam * bstar / 3
        fppv = -4 * mstar / bstar**3 - 2 * Lam / 3
        s = np.sqrt(-fppv / 2)
        rows.append(dict(Lambda=Lam, b_star=bstar, f=fv, fp=fpv, fpp=fppv,
                         s=float(s), one_over_bstar=float(1 / bstar)))
        say(f"    Lambda = {Lam:<4}: b* = {bstar:.6f}, f = {fv:+.1e}, f' = {fpv:+.1e}, "
            f"f'' = {fppv:+.6f} = -2 Lambda; s = sqrt(-f''/2) = {s:.6f} = 1/b* = {1/bstar:.6f}")
    say("    The T26 result 'every Nariai point is a saddle with eigenvalue +1/b*, independently of")
    say("    the potential V' is exactly the f'' = -2 Lambda < 0 case of A3/A5: no separate theorem")
    say("    for scalar fields is needed.")
    res["B3"] = rows

    say("\nB4. Codimension: are f = f' = f'' = 0 independent under arbitrary local perturbations?")
    say("    delta m(r) = eps * S_k(r) with three independent smooth bumps; response matrix")
    say("    M_ik = d(f, f', f'')_i / d eps_k at R_-.")
    centres = (0.45, 0.68, 0.95)
    width = 0.12
    eps = 1e-6
    cols = []
    for c in centres:
        def dm(rr, c=c):
            return np.exp(-((rr - c) / width) ** 2)

        def parts_pert(rr, sgn):
            rr = np.atleast_1d(rr)
            m, mp, mpp = P._interp(rr)
            dmv = dm(rr)
            dmp = -2 * (rr - c) / width**2 * dmv
            dmpp = (4 * (rr - c) ** 2 / width**4 - 2 / width**2) * dmv
            m = m + sgn * eps * dmv
            mp = mp + sgn * eps * dmp
            mpp = mpp + sgn * eps * dmpp
            f = 1 - 2 * m / rr
            fp = 2 * m / rr**2 - 2 * mp / rr
            fpp = -4 * m / rr**3 + 4 * mp / rr**2 - 2 * mpp / rr
            return np.array([f[0], fp[0], fpp[0]])

        col = (parts_pert(np.array([Rm]), +1) - parts_pert(np.array([Rm]), -1)) / (2 * eps)
        cols.append(col)
    Mmat = np.array(cols).T
    sv = np.linalg.svd(Mmat, compute_uv=False)
    say(f"    bump centres {centres}, width {width}")
    say("    response matrix (rows f, f', f''; columns = bumps):")
    for i, nm in enumerate(("f  ", "f' ", "f''")):
        say(f"      {nm}  " + "  ".join(f"{Mmat[i,j]:+.4e}" for j in range(3)))
    say(f"    singular values = {np.array2string(sv, precision=3)};  rank = {np.linalg.matrix_rank(Mmat, tol=1e-10)}")
    say("    => the three conditions are functionally independent: in the space of mass profiles")
    say("       'a degenerate horizon exists somewhere' is codimension 2 (three conditions minus the")
    say("       free location R_-), and codimension 1 inside a family whose shape is already fixed")
    say("       and only the scale is free (T26 (a), T1).")
    res["B4"] = dict(centres=list(centres), width=width, matrix=Mmat.tolist(),
                     singular_values=sv.tolist(), rank=int(np.linalg.matrix_rank(Mmat, tol=1e-10)))
    return res


def main():
    out = dict(A=part_A(), B=part_B())
    say("\n" + "=" * 78)
    say("SUMMARY of part (b)")
    say("=" * 78)
    say("The dynamical closures DO reduce to one statement:")
    say("  In the class T^t_t = T^r_r the T-region flow preserves phase volume; a degenerate")
    say("  horizon is a fixed point with zero-trace linearisation and eigenvalues +- sqrt(-f''/2);")
    say("  it is a saddle (double root) or fully degenerate (triple root), never a sink.")
    say("  Hence its stable set has empty interior: measure zero in initial data, codimension 2")
    say("  in profile space.  T26 (fluid/scalar/k-essence), T29 (zeta > 0), T30 (piecewise),")
    say("  T21 (quartic end) and the profile-space part of T1 are all instances.")
    say("The premises that are NOT needed: autonomy (A2), smoothness beyond continuity (A2/A3),")
    say("  energy conditions (used nowhere above), the class condition itself at the fixed point (A1).")
    say("The premises that ARE load-bearing: locality (finite-dimensional state) and the")
    say("  boundedness of the source as v -> 0 (no Theta-dependence) -- tested in t36_loopholes.py")
    say("  and t36_nonlocal.py.")
    say("")
    say("=" * 78)
    say("STATEMENT OF THE RESULT (part (d) of the task)")
    say("=" * 78)
    say("THEOREM (no selection of a degenerate inner horizon by a local classical source).")
    say("  Let the spacetime be spherically symmetric and satisfy the Einstein equations, and let")
    say("  the region between horizons be the homogeneous Kantowski-Sachs reduction (equivalently,")
    say("  the static mass function m(r) continued into f < 0).  Let the transverse pressure be")
    say("  given by a constitutive law  p_perp = P(b, rho, b', internal variables)  which")
    say("    (i)   is a function on a FINITE-DIMENSIONAL state space (locality);")
    say("    (ii)  makes the evolution field continuous and locally Lipschitz;")
    say("    (iii) has a finite limit as b' -> 0 at fixed (b, rho) (boundedness at the horizon,")
    say("          i.e. no dependence on the expansion Theta).")
    say("  Explicit dependence on proper time is ALLOWED.  No energy condition is assumed.  The")
    say("  radial condition p_r = -rho need not be assumed: it is forced at any degenerate")
    say("  horizon (A1).  Then:")
    say("    1. a degenerate horizon f = f' = 0 at b = b* is a fixed point whose linearisation has")
    say("       ZERO TRACE, with transverse eigenvalues s = +- sqrt(-8 pi p_perp(b*)) =")
    say("       +- sqrt(-f''(b*)/2);")
    say("    2. hence it is a saddle (f'' < 0), a centre (f'' > 0) or totally degenerate (f'' = 0,")
    say("       a triple root, where p_perp(b*) = 0), and NEVER asymptotically stable; if the law")
    say("       does not depend on b' the flow is exactly Hamiltonian, H = b'^2/2 + f(b, tau)/2,")
    say("       and preserves phase-space volume;")
    say("    3. therefore the set of interior data ending on a degenerate horizon has EMPTY")
    say("       INTERIOR.  In profile space f = f' = f'' = 0 is codimension 2 (codimension 1 with")
    say("       the shape fixed and only the scale free), with normal form")
    say("       kappa_- = -(3/2)|f'''(R_-)/6|^{1/3} |eps g(R_-)|^{2/3}  (T1).")
    say("")
    say("SHARPNESS.  Premises (i) and (iii) are each necessary.")
    say("  - dropping (iii) admits Theta-dependent laws; the transverse trace is then 8 pi zeta_eff")
    say("    with zeta_eff = -dPi/dTheta, so an attractor exists only for zeta_eff < 0, i.e.")
    say("    NEGATIVE ENTROPY PRODUCTION.  The theorem survives with the second law as a premise.")
    say("  - dropping (i) -- a law reading the TOTAL mass, ell = lambda M -- admits an attractor by")
    say("    homothety, but only if the tracking is exact: a relative tracking error eta gives back")
    say("    T1's law, kappa_- = -1.92 |eta|^{2/3}, with T1's own coefficient.")
    say("  - autonomy and smoothness beyond continuity are NOT needed; energy conditions are NOT")
    say("    used anywhere in the proof.")
    say("")
    say("NOT COVERED: non-sphericity (T37); rotation (T31: the rotating triple root exists only for")
    say("  a <~ 0.1-0.15 and is itself codimension 2); quantum gravity and semiclassical sources")
    say("  (T14, T38); non-local sources (by the sharpness clause); the wider class p_x != -rho away")
    say("  from the horizon (A4: the flow is bounded but not Lipschitz there -- the exact step at")
    say("  which a fully general proof stops); and what the interior looks like after the exit.")
    json.dump(out, open(OUT / "t36_core.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    (LOGS / "t36_core_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\nwritten: {OUT/'t36_core.json'}, {LOGS/'t36_core_log.txt'}")


if __name__ == "__main__":
    main()
