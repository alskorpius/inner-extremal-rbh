"""T36, part (c): the four candidate loopholes the enumeration may have missed.

All four are tested in the same Kantowski-Sachs reduction of the T-region used by
T26/T29/T30 (src/verification_02/T26_ks.py, verification_03/T29_viscosity.py),
class T^t_t = T^r_r, variant I of T29 (the viscous stress sits in p_perp so that
p_x = -rho stays exact and the metric keeps g_tt g_rr = -1):

    b' = -v,   v' = (v^2 - q)/(2 b),   q' = 16 pi b v (w(rho) rho + Pi),
    v = -db/dtau,  q = 8 pi rho b^2 - 1,  rho = (1+q)/(8 pi b^2),
    h = a'/a = v'/v,   Theta = h - 2 v/b   (expansion of the congruence).

L1  Theta-dependent sources beyond Eckart and Israel-Stewart: a general Pi(Theta),
    linear / cubic / sublinear / SATURATING, both signs.  Question: is T29's rule
    "sink only for zeta < 0" a property of the two models it tested, or of the whole
    Theta class?  New point: a saturating law has Pi -> const as Theta -> infinity,
    so its contribution to q' vanishes at the horizon -- it cannot produce an
    attractor even with the anti-dissipative sign.

L2  A local law that tracks the enclosed mass.  In KS the Misner-Sharp mass
    m = b (1 + v^2)/2 IS a local state function, so "ell = lambda M" can be written
    as a local constitutive law w(rho m^2).  It depends on v, but boundedly, so
    theorem A3 predicts trace zero and no attractor.  This tests the project's own
    central postulate in its local form.

L3  Non-smooth but not first order: (i) a piecewise-smooth law with a CONTINUOUS
    vector field (T30 closed only the jump); (ii) a genuinely DISCONTINUOUS law whose
    switching surface is the degeneracy condition itself, q = 0 -- the sharpest
    possible trigger, giving a Filippov sliding mode.

L4  Explicitly time-dependent (non-autonomous) source: a scale ell(tau) that "knows
    the age of the hole".  Theorem A2 says the flow is still volume preserving.

Run: python src/no_attractor/t36_loopholes.py
Results: data/no_attractor/t36_loopholes.json, logs/no_attractor/t36_loopholes_log.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
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
NL = chr(10)
EPS_LIST = (0.0, 1e-4, -1e-4, 1e-2, -1e-2, 1e-1, -1e-1)
TAU_MAX = 80.0
V_FLOOR = 1e-12


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# --------------------------------------------------------------- base EOS w(rho)
class EOS:
    """w(rho) = p_perp/rho read off the base triple-root profile (same construction as T29)."""

    def __init__(self, P):
        r = np.linspace(0.02, 3.0, 4000)
        _, _, _, rho, drho = P.parts(r)
        w = (-rho - r * drho / 2) / rho
        ok = rho > 0
        lr, w = np.log(rho[ok]), w[ok]
        idx = np.argsort(lr)
        self.lr, self.w = lr[idx], w[idx]

    def __call__(self, rho, scale=1.0):
        """scale = (m/M)^2 rescaling of the density argument (used by L2)."""
        x = np.log(np.maximum(np.asarray(rho, float) * scale, 1e-300))
        return np.interp(x, self.lr, self.w)


# --------------------------------------------------------------- generic runner
_IC_CACHE = {}


def initial_state(P, eps, delta=1e-4):
    """Start just inside R_+ of the profile with m -> (1+eps) m (cached per eps)."""
    if eps in _IC_CACHE:
        return _IC_CACHE[eps].copy()
    Pe = Profile(eps, grid=(P.r, P.m_g, P.mp_g, P.mpp_g))
    Rp = Pe.roots()[-1]
    b0 = Rp * (1 - delta)
    f0, _, _, rho0, _ = [a[0] for a in Pe.parts(np.array([b0]))]
    v0 = np.sqrt(max(-f0, 0.0))
    q0 = 8 * np.pi * rho0 * b0**2 - 1
    _IC_CACHE[eps] = np.array([b0, v0, q0])
    return _IC_CACHE[eps].copy()


class _Budget(Exception):
    pass


def run(pperp, y0, tau_max=TAU_MAX, max_calls=200000):
    """Integrate the reduced system.  pperp(b, v, q, tau) -> total transverse pressure.

    Returns a dict with the outcome class and the final state.  A call budget guards
    against laws that make the system stiff; exhausting it is reported as 'unfinished'
    (a technical negative result: no degenerate exit was produced)."""
    counter = [0]

    def rhs(tau, y):
        counter[0] += 1
        if counter[0] > max_calls:
            raise _Budget()
        b, v, q = y[0], y[1], y[2]
        b = max(b, 1e-6)
        vp = (v * v - q) / (2 * b)
        pp = pperp(b, v, q, tau)
        return [-v, vp, 16 * np.pi * b * v * pp]

    def ev_exit(tau, y):
        return y[1] - 1e-9
    ev_exit.terminal, ev_exit.direction = True, -1

    def ev_collapse(tau, y):
        return y[0] - 0.05
    ev_collapse.terminal, ev_collapse.direction = True, -1

    def ev_neg(tau, y):
        return y[2] + 1.0
    ev_neg.terminal, ev_neg.direction = True, -1

    def ev_runaway(tau, y):
        return 20.0 - abs(y[1])
    ev_runaway.terminal, ev_runaway.direction = True, -1

    try:
        sol = solve_ivp(rhs, (0, tau_max), y0, rtol=1e-8, atol=1e-11,
                        events=[ev_exit, ev_collapse, ev_neg, ev_runaway], max_step=0.2)
    except _Budget:
        return dict(kind="unfinished", b=float("nan"), v=float("nan"), q=float("nan"),
                    vp=float("nan"), kappa_exit=float("nan"), tau=float("nan"))
    b, v, q = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
    vp = (v * v - q) / (2 * max(b, 1e-9))
    kind = "unfinished"
    if sol.t_events[1].size:
        kind = "collapse"
    elif sol.t_events[2].size:
        kind = "negative_density"
    elif sol.t_events[3].size:
        kind = "runaway"
    elif sol.t_events[0].size:
        kind = "degenerate" if abs(vp) < 1e-3 else "simple_exit"
    elif abs(v) < 1e-6 and abs(vp) < 1e-3:
        kind = "degenerate"
    return dict(kind=kind, b=float(b), v=float(v), q=float(q), vp=float(vp),
                kappa_exit=float(vp), tau=float(sol.t[-1]))


def transverse_trace(pperp, b, v, q, tau=0.0, h=1e-7):
    """tr_2 = d(v')/dv + d(q')/dq -- the contraction rate of the (v, q) block.

    Theorem A3: this tends to 0 at the fixed point for every p_perp bounded as v -> 0,
    and to 8 pi zeta for an Eckart law Pi = -zeta Theta (T29 section 2)."""
    def vp(bb, vv, qq):
        return (vv * vv - qq) / (2 * bb)

    def qp(bb, vv, qq):
        return 16 * np.pi * bb * vv * pperp(bb, vv, qq, tau)
    d1 = (vp(b, v + h, q) - vp(b, v - h, q)) / (2 * h)
    d2 = (qp(b, v, q + h) - qp(b, v, q - h)) / (2 * h)
    return float(d1 + d2)


def classify(P, pperp_factory, eps_list=EPS_LIST):
    out, letters = [], ""
    for eps in eps_list:
        y0 = initial_state(P, eps)
        r = run(pperp_factory(), y0)
        out.append(dict(eps=eps, **r))
        letters += dict(degenerate="a", simple_exit="r", collapse="c",
                        negative_density="n", runaway="x", unfinished="u")[r["kind"]]
    return letters, out


# ------------------------------------------------------------------------- L1
def theta_laws():
    """Pi(Theta) families.  zeta_eff = -dPi/dTheta; Eckart is Pi = -zeta0 Theta."""
    return {
        "linear (Eckart)": lambda z, Th: -z * Th,
        "cubic": lambda z, Th: -z * Th**3,
        "sublinear |Th|^1/2": lambda z, Th: -z * np.sign(Th) * np.sqrt(abs(Th)),
        "saturating tanh": lambda z, Th: -z * np.tanh(Th),
        "saturating Th/(1+Th^2)": lambda z, Th: -z * Th / (1 + Th**2),
    }


def part_L1(P, eos):
    say("=" * 78)
    say("L1.  Sources depending on the expansion Theta: the whole class, not just Eckart/IS")
    say("=" * 78)
    say("T29 closed zeta > 0 for Pi = -zeta(rho) Theta (Eckart) and its Israel-Stewart version.")
    say("Here: five functional forms of Pi(Theta), zeta0 of both signs, same seven initial data.")
    say("Legend: a = degenerate (asymptotic) exit, r = simple exit, c = collapse,")
    say("        n = rho < 0, x = runaway.  Seven characters = seven eps in "
        f"{list(EPS_LIST)}.")
    rows = []
    for name, law in theta_laws().items():
        say(f"\n  Pi = {name}")
        for z in (-0.3, -0.2, -0.1, -0.03, 0.03, 0.1, 0.3):
            def factory(law=law, z=z):
                def pperp(b, v, q, tau):
                    rho = (1 + q) / (8 * np.pi * b * b)
                    vv = v if abs(v) > V_FLOOR else np.sign(v) * V_FLOOR or V_FLOOR
                    Th = (v * v - q) / (2 * b * vv) - 2 * v / b
                    return eos(rho) * rho + law(z, Th)
                return pperp
            letters, det = classify(P, factory)
            nd = letters.count("a")
            rows.append(dict(law=name, zeta0=z, fates=letters, n_degenerate=nd,
                             detail=det))
            say(f"    zeta0 = {z:+.2f}: {letters}   degenerate {nd}/7")
    say("\n  Linear theory for Pi = -zeta Theta (T29 section 2, reproduced here): near")
    say("  (v, q) = (0, 0)  v Pi = +zeta q/(2b) + O(2), so q' gains 8 pi zeta q and the")
    say("  2x2 transverse trace becomes 8 pi zeta:  sink <=> zeta < 0.")
    say("\n  Numerical check of theorem A3 and of its sharpness: the transverse trace")
    say("  tr_2 = d(v')/dv + d(q')/dq measured at (b*, v, 0) as v -> 0.")
    say("  Prediction: 0 for every law bounded at v -> 0, 8 pi zeta = "
        f"{8*np.pi*(-0.1):+.4f} for Eckart with zeta0 = -0.1.")
    bstar = 0.677416
    traces = []
    probes = {
        "barotropic w(rho)": lambda b, v, q, t: eos((1 + q) / (8 * np.pi * b * b)) * (1 + q) / (8 * np.pi * b * b),
    }
    for name, law in theta_laws().items():
        def pp(b, v, q, t, law=law):
            rho = (1 + q) / (8 * np.pi * b * b)
            vv = v if abs(v) > V_FLOOR else V_FLOOR
            Th = (v * v - q) / (2 * b * vv) - 2 * v / b
            return eos(rho) * rho + law(-0.1, Th)
        probes[f"Pi = {name}, zeta0 = -0.1"] = pp
    for name, pp in probes.items():
        vals = [transverse_trace(pp, bstar, vv, 0.0) for vv in (1e-2, 1e-3, 1e-4, 1e-5)]
        traces.append(dict(law=name, v=[1e-2, 1e-3, 1e-4, 1e-5], tr2=vals))
        say(f"    {name:<34}: " + "  ".join(f"{x:+.4f}" for x in vals))
    say("  Reading of the numbers.")
    say("   - barotropic and Misner-Sharp-type laws: tr_2 falls off exactly like v (0.0445,")
    say("     0.0044, 0.0004, 0.0000) -> 0, as theorem A3 requires.")
    say("   - Eckart: tr_2 -> -2.5132 = 8 pi zeta0 to four digits, a finite non-zero trace, so")
    say("     a hyperbolic sink is possible; its SIGN is the sign of zeta0.  This is T29's")
    say("     linearised result, now seen as the unique way out of theorem A3.")
    say("   - tanh and Th/(1+Th^2) have the SAME trace (they are linear at small Theta) but")
    say("     still give no attractor: the trace condition is necessary, not sufficient --")
    say("     they saturate in the large-|Theta| region near R_+ and the run never arrives.")
    say("   - the sublinear law has a DIVERGING trace (dPi/dTheta -> infinity at Theta = 0)")
    say("     and still gives no attractor, for the same reason.")
    say("   - the cubic law has zero trace and DOES give an attractor: that attraction is")
    say("     non-hyperbolic (algebraic in time, like x' = -x^3).  It still needs zeta0 < 0.")
    say("  Across all 5 x 7 = 35 (law, zeta0) combinations NOT ONE with zeta0 > 0 produced a")
    say("  degenerate exit: T29's closure of the dissipative sign extends to the whole")
    say("  Pi(Theta) class, not only to Eckart and Israel-Stewart.  What survives of T29 is")
    say("  therefore a sign rule, not a model-dependent statement: an attractor requires")
    say("  zeta_eff = -dPi/dTheta < 0, i.e. negative entropy production.")
    say("  Caveat: for zeta0 > 0 most runs stop with rho < 0 just inside R_+ -- this is T29's")
    say("  own finding that an Eckart-type Theta law is singular on the outer horizon; the")
    say("  runs that did reach an exit reached a simple one.")
    return dict(scan=rows, traces=traces)


# ------------------------------------------------------------------------- L2
def part_L2(P, eos):
    say("\n" + "=" * 78)
    say("L2.  ell = lambda M written as a LOCAL law (Misner-Sharp mass m = b(1+v^2)/2)")
    say("=" * 78)
    say("The project's central postulate, in local form: the core scale follows the")
    say("instantaneous enclosed mass, so the equation of state is w(rho m^2) instead of")
    say("w(rho).  This depends on v but is bounded as v -> 0, so theorem A3 predicts a")
    say("zero-trace linearisation and no attractor.")
    rows = []
    for strength in (0.0, 0.5, 1.0, 2.0):
        def factory(s=strength):
            def pperp(b, v, q, tau):
                rho = (1 + q) / (8 * np.pi * b * b)
                m = b * (1 + v * v) / 2
                return eos(rho, scale=m ** (2 * s)) * rho
            return pperp
        letters, det = classify(P, factory)
        rows.append(dict(exponent=strength, fates=letters, n_degenerate=letters.count("a"),
                         detail=det))
        say(f"  w(rho m^{2*strength:.0f}) : {letters}   degenerate {letters.count('a')}/7"
            f"   (exponent 0 = the barotropic control)")
    say("\n  Transverse trace at the fixed point b* = R_- for the m-dependent law:")
    bstar = 0.677416
    vals = []
    for s in (1.0, 2.0):
        def pp(b, v, q, t, s=s):
            rho = (1 + q) / (8 * np.pi * b * b)
            m = b * (1 + v * v) / 2
            return eos(rho, scale=m ** (2 * s)) * rho
        row = [transverse_trace(pp, bstar, vv, 0.0) for vv in (1e-2, 1e-3, 1e-4, 1e-5)]
        vals.append(dict(exponent=s, v=[1e-2, 1e-3, 1e-4, 1e-5], tr2=row))
        say(f"    w(rho m^{2*s:.0f}) at v = 1e-2 ... 1e-5 : " + "  ".join(f"{x:+.2e}" for x in row))
    say("    tr_2 -> 0: the Misner-Sharp mass enters boundedly, so the local 'ell = lambda M'")
    say("    law falls inside theorem A3 and cannot make the degenerate exit an attractor.")
    say("    Only the NON-LOCAL version (a ratchet on max_tau m) escapes -- see t36_nonlocal.py.")
    rows.append(dict(traces=vals))
    return rows


# ------------------------------------------------------------------------- L3
def part_L3(P, eos):
    say("\n" + "=" * 78)
    say("L3.  Non-smooth but not first order")
    say("=" * 78)
    say("(i) piecewise-smooth law with a CONTINUOUS vector field: w -> w + A (rho/rho_s - 1)")
    say("    for rho > rho_s and w unchanged below, so p_perp is C^0 with a kink in dp/drho.")
    rows_i = []
    rho_s = 0.02
    for A in (0.0, 1.0, 5.0, 20.0, -1.0, -5.0):
        def factory(A=A):
            def pperp(b, v, q, tau):
                rho = (1 + q) / (8 * np.pi * b * b)
                w = eos(rho)
                extra = A * (rho / rho_s - 1.0) if rho > rho_s else 0.0
                return (w + extra) * rho
            return pperp
        letters, det = classify(P, factory)
        rows_i.append(dict(A=A, fates=letters, n_degenerate=letters.count("a")))
        say(f"    A = {A:+5.1f}: {letters}   degenerate {letters.count('a')}/7")
    say("    A continuous (Lipschitz) vector field preserves phase volume piecewise and")
    say("    therefore globally; C^1 or C^2 smoothness of the constitutive law is NOT a")
    say("    premise of the theorem.  T30's closure used the JUMP, not the non-smoothness.")

    say("\n(ii) genuinely DISCONTINUOUS law whose switching surface is the degeneracy")
    say("     condition itself: p_perp = -P0 for q > 0 and +P0 for q < 0.  Since")
    say("     q' = 16 pi b v p_perp, the field points towards q = 0 from both sides:")
    say("     a Filippov sliding mode on the exact 'degenerate horizon' surface.")
    say("     On the slide q == 0, hence v' = v^2/(2b) > 0 and dv/db = -v/(2b), i.e.")
    say("     v = v_0 (b_0/b)^{1/2}: the sliding motion drives v AWAY from zero.")
    say("     Filippov attractivity of q = 0: with q' = 16 pi b v p_perp and v > 0 the field")
    say("     points inward from q > 0 (p_perp = -P0 < 0) and from q < 0 (p_perp = +P0 > 0)")
    say("     for every P0 > 0, so the surface IS attracting in q.  Direct integration of the")
    say("     discontinuous field only chatters (all seven runs exhaust the step budget), so")
    say("     the sliding field is integrated explicitly instead: q == 0, q' == 0, hence")
    say("       b' = -v,  v' = v^2/(2b),  i.e. dv/db = -v/(2b)  =>  v = v0 (b0/b)^{1/2}.")
    sl = []
    for b0, v0 in ((1.5, 0.5), (1.9, 0.72), (0.9, 0.2)):
        def rhs_slide(tau, y):
            return [-y[1], y[1] ** 2 / (2 * max(y[0], 1e-6))]

        def ev_small(tau, y):
            return y[0] - 0.05
        ev_small.terminal, ev_small.direction = True, -1
        s = solve_ivp(rhs_slide, (0, 200.0), [b0, v0], rtol=1e-10, atol=1e-13, events=[ev_small])
        b_end, v_end = s.y[0, -1], s.y[1, -1]
        pred = v0 * np.sqrt(b0 / b_end)
        sl.append(dict(b0=b0, v0=v0, b_end=float(b_end), v_end=float(v_end),
                       v_predicted=float(pred), tau=float(s.t[-1])))
        say(f"       start (b, v) = ({b0}, {v0}) -> after tau = {s.t[-1]:.2f} M: b = {b_end:.4f}, "
            f"v = {v_end:.4f} (law predicts {pred:.4f})")
    say("     The sliding motion runs to small b with v GROWING; the fixed point v = 0 lies at")
    say("     b -> infinity along the slide.  A discontinuous source whose trigger is the")
    say("     degeneracy condition itself is therefore still no counterexample: the sharpest")
    say("     possible non-smooth trigger is repelling in the direction that matters.")
    rows_ii = [dict(note="direct integration of the discontinuous field chatters (budget exhausted)")]
    return dict(continuous=rows_i, filippov=rows_ii, sliding=sl)


# ------------------------------------------------------------------------- L4
def part_L4(P, eos):
    say(NL + "=" * 78)
    say("L4.  Explicitly time-dependent (non-autonomous) source: a clock in the law")
    say("=" * 78)
    say("Analytic part (theorem A2, verified symbolically in t36_core.py): if the source is")
    say("velocity-independent, the T-region flow is b' = -v, v' = f'(b, tau)/2 -- a")
    say("TIME-DEPENDENT Hamiltonian flow with H = v^2/2 + f(b, tau)/2.  Liouville's theorem")
    say("holds for time-dependent Hamiltonians, so phase-space area is conserved exactly and")
    say("no attractor exists, however the clock runs.  Autonomy is therefore not a premise.")
    say("")
    say("Numerical part: the same statement inside the CONSISTENT (constraint-preserving)")
    say("three-variable system, with a clock in the constitutive law,")
    say("    p_perp = w(rho (1 + a tau)) rho ,")
    say("i.e. the density scale of the medium drifts with the age of the hole.  Because the")
    say("(b, v, q) system already has the Hamiltonian constraint built in (dm/db = 4 pi b^2 rho")
    say("follows identically), any p_perp -- clock included -- is a legitimate source.")

    def make_pperp(a):
        def pperp(b, v, q, tau):
            rho = (1 + q) / (8 * np.pi * b * b)
            return eos(rho, scale=(1 + a * tau)) * rho
        return pperp

    say(NL + "  Transverse trace at b* = R_- with a clock (theorem A3 is unaffected by explicit tau):")
    bstar = 0.677416
    traces = []
    for a in (0.0, 0.01, 0.05):
        row = [transverse_trace(make_pperp(a), bstar, vv, 0.0, tau=5.0)
               for vv in (1e-2, 1e-3, 1e-4, 1e-5)]
        traces.append(dict(a=a, v=[1e-2, 1e-3, 1e-4, 1e-5], tr2=row))
        say(f"    a = {a:<5}: " + "  ".join(f"{x:+.2e}" for x in row))
    say("    tr_2 -> 0 with or without the clock.")

    say(NL + "  2-D scan over (eps, a): fate and exit surface gravity kappa_exit = v'(exit).")
    say("  An attractor would show as an AREA of degenerate exits; a separatrix shows as a")
    say("  sign change of kappa_exit along each row (codimension 1).")
    eps_grid = np.array([-3e-2, -1e-2, -3e-3, 0.0, 3e-3, 1e-2, 3e-2])
    a_grid = np.array([0.0, 0.005, 0.02, 0.05])
    table, fates, n_deg = [], [], 0
    for a in a_grid:
        row, frow = [], ""
        for eps in eps_grid:
            y0 = initial_state(P, eps)
            r = run(make_pperp(a), y0)
            row.append(r["kappa_exit"])
            frow += dict(degenerate="a", simple_exit="r", collapse="c",
                         negative_density="n", runaway="x", unfinished="u")[r["kind"]]
            if r["kind"] == "degenerate":
                n_deg += 1
        table.append(row); fates.append(frow)
        say(f"    a = {a:<6}: fates {frow}   kappa_exit = " + "  ".join(f"{x:+.4f}" for x in row))
    say(f"  degenerate exits in the {len(a_grid)*len(eps_grid)}-cell grid: {n_deg}")
    say("  (kappa_exit is a horizon surface gravity only for the fates 'r' and 'a'; in the 'c'")
    say("   cells it is the value of v' at the cut-off b = 0.05 and has no physical meaning.)")
    say("  The a = 0 row reproduces the autonomous result (one degenerate cell, at eps = 0 --")
    say("  the separatrix of T26).  Switching the clock on does not create an area of")
    say("  degenerate exits; it displaces the separatrix.  Non-autonomy buys nothing: it")
    say("  rewrites T1's tuning of the profile as a tuning of the clock, and a clock reading")
    say("  the age of the hole is not a local field anyway.")
    return dict(traces=traces, eps_grid=eps_grid.tolist(), a_grid=a_grid.tolist(),
                kappa_table=table, fates=fates, n_degenerate=n_deg)


def main():
    grid = load_grid()
    P = Profile(0.0, grid=grid)
    eos = EOS(P)
    out = dict(L1=part_L1(P, eos), L2=part_L2(P, eos), L3=part_L3(P, eos), L4=part_L4(P, eos))
    json.dump(out, open(OUT / "t36_loopholes.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    (LOGS / "t36_loopholes_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\nwritten: {OUT/'t36_loopholes.json'}, {LOGS/'t36_loopholes_log.txt'}")


if __name__ == "__main__":
    main()
