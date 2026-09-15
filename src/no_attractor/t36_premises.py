"""T36, part (a): formalise the premises and audit which closure uses which.

No physics is computed here; this is the bookkeeping that turns the enumeration into
a statement with explicit hypotheses.  It emits two machine-readable tables:

  premises  -- the formal definition of each premise;
  usage     -- for each closed class, the premises its published argument actually uses,
               plus whether the class is an instance of the fixed-point statement of
               part (b) (theorem A/B in t36_core.py).

Sources for the "uses" column are the reports themselves, cited by path in REPORT.md.

Run: python src/no_attractor/t36_premises.py
Results: data/no_attractor/t36_premises.json, logs/no_attractor/t36_premises_log.txt
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / HERE.name
LOGS = ROOT / "logs" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
LOG = []
NL = chr(10)


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


PREMISES = {
    "P1": dict(
        name="spherical symmetry + GR",
        statement="The geometry is spherically symmetric and obeys the Einstein equations; "
                  "the region between horizons is the homogeneous Kantowski-Sachs T-region "
                  "ds^2 = -dtau^2 + a^2 dx^2 + b^2 dOmega^2 (equivalently, the static profile "
                  "f(r) continued into f < 0)."),
    "P2": dict(
        name="locality",
        statement="The source at a point is a function of the fields and of finitely many of "
                  "their derivatives AT THAT POINT; equivalently, the constitutive law is a "
                  "function on a finite-dimensional state space (internal memory variables with "
                  "bounded dynamics are allowed and do not break this)."),
    "P3": dict(
        name="autonomy",
        statement="The constitutive law contains no explicit dependence on the coordinates "
                  "(no clock, no preferred radius)."),
    "P4": dict(
        name="smoothness",
        statement="The right-hand side of the evolution system is C^k with k >= 1."),
    "P5": dict(
        name="source class T^t_t = T^r_r",
        statement="p_r = -rho, equivalently g_tt g_rr = -1 for the static form."),
    "P6": dict(
        name="boundedness at the horizon",
        statement="The constitutive law has a finite limit as v = -db/dtau -> 0 at fixed (b, rho); "
                  "equivalently it does not depend on the expansion Theta, which diverges at "
                  "every non-degenerate horizon."),
    "P7": dict(
        name="energy conditions",
        statement="NEC (rho + p_i >= 0) and/or WEC hold on the solution."),
    "P8": dict(
        name="second law",
        statement="Entropy production is non-negative: for a bulk-viscous law, zeta >= 0."),
    "P9": dict(
        name="Lagrangian origin",
        statement="The source derives from an action (perfect fluid with an equation of state, "
                  "scalar field, k-essence, ...)."),
    "P10": dict(
        name="quantitative input",
        statement="An external magnitude or time scale enters the argument (hbar, an accretion "
                  "rate, the age of the universe, a curvature bound).  Not a structural premise."),
}

USAGE = [
    dict(cls="T1", what="tuning: kappa_- ~ eps^{2/3}", where="experiments/verification_01/T1_REPORT.md",
         uses=["P1", "P5"], not_used=["P3", "P4", "P6", "P7", "P8", "P9"],
         layer="kinematic (profile space)", instance_of_fixed_point="yes, as the normal form",
         note="Pure codimension statement: three conditions f = f' = f'' = 0 at one point. "
              "It is theorem A (codimension 2 in profile space) made quantitative; the exponent "
              "2/3 is the normal form of a cubic root under a linear perturbation."),
    dict(cls="T2", what="classical back-reaction of mass inflation",
         where="experiments/verification_01/T2_REPORT.md",
         uses=["P1", "P10"], not_used=["P4", "P6", "P7", "P8", "P9"],
         layer="dynamical, but in a DIFFERENT system (Ori two-stream null shell)",
         instance_of_fixed_point="partly",
         note="kappa_- = 0 is a fixed point of the back-reaction map kappa -> kappa', and the "
              "content is that it REPELS (d|kappa_-|/dN_e > 0). That is a fixed-point statement, "
              "but not the Liouville statement of theorem B: it lives in the space of kappa under "
              "accreted e-folds, not in the KS phase plane, and the sign is an independent input."),
    dict(cls="T14", what="semiclassical flux at R_- (2D Polyakov, Unruh)",
         where="experiments/verification_01/T14_REPORT.md",
         uses=["P1", "P10"], not_used=["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
         layer="magnitude", instance_of_fixed_point="no",
         note="Two independent contents: (i) the power is 6e-61 of what is needed -- a pure "
              "magnitude; (ii) the fixed point of the semiclassical feedback is |kappa_-| = kappa_+, "
              "not 0. Neither is the Liouville statement. Also it is not a classical source at all, "
              "so it is outside the theorem's class by construction."),
    dict(cls="T20", what="no protecting conserved quantity", where="VERIFICATION_ANSWERS.md, T20",
         uses=["P1", "P5"], not_used=["P3", "P4", "P6", "P7", "P8", "P9"],
         layer="kinematic (profile space)", instance_of_fixed_point="yes",
         note="Says the three conditions are not a first integral. In the language of theorem B: "
              "the fixed point is not protected by a conserved quantity of the flow, and indeed "
              "the only conserved quantity of the barotropic flow is H = v^2/2 + f/2, whose zero "
              "level set is the whole T-region, not the fixed point."),
    dict(cls="T21", what="extremal end is a branch boundary, not an attractor",
         where="experiments/verification_01/T21_REPORT.md",
         uses=["P1", "P2", "P5"], not_used=["P3", "P4", "P6", "P7", "P8", "P9"],
         layer="kinematic + dynamical", instance_of_fixed_point="yes",
         note="The quartic root has f'' = f''' = 0 as well; theorem A5 puts it at the boundary "
              "between the saddle (f'' < 0) and centre (f'' > 0) cases, which is exactly the "
              "observed split '+delta M -> two horizons / -delta M -> horizonless'."),
    dict(cls="E4 (BH-P)", what="curvature towers h = sum alpha_n psi^n, alpha_n >= 0",
         where="experiments/audit_bhp/REPORT.md, row E4",
         uses=["P1", "P5", "positivity alpha_n >= 0"],
         not_used=["P3", "P4", "P6", "P7", "P8"],
         layer="kinematic (profile space), algebraic", instance_of_fixed_point="no",
         note="Stronger than codimension: inside that ansatz a triple root is IMPOSSIBLE, not "
              "merely rare, because 2 psi h'' - h' >= 1 on a root. The positivity alpha_n >= 0 is "
              "used by no other closure: it is load-bearing for E4's stronger claim and is NOT a "
              "premise of the general theorem."),
    dict(cls="T26", what="smooth Lagrangians in the KS minisuperspace",
         where="experiments/verification_02/T26_REPORT.md",
         uses=["P1", "P2", "P3", "P4", "P5", "P6", "P9"], not_used=["P7", "P8"],
         layer="dynamical (the anchor)", instance_of_fixed_point="yes -- it IS the statement",
         note="Theorem B in its original form. Part (b) shows that of its seven premises only "
              "P1, P2, P6 are load-bearing: P3 is redundant (A2), P4 weakens to continuity (L3i), "
              "P5 is forced at the fixed point (A1), P9 is redundant (A3 covers non-Lagrangian "
              "bounded laws)."),
    dict(cls="T29", what="bulk viscosity zeta > 0 (Eckart, Israel-Stewart)",
         where="experiments/verification_03/T29_REPORT.md",
         uses=["P1", "P2", "P5", "P8"], not_used=["P4", "P6", "P7", "P9"],
         layer="dynamical, P6 dropped on purpose", instance_of_fixed_point="yes, as the boundary case",
         note="The ONLY closure that drops P6, and the only one that needs P8. Its content is the "
              "sign rule: the transverse trace at the fixed point is 8 pi zeta, so a sink requires "
              "zeta < 0. P8 is therefore load-bearing AND used by exactly one class -- the audit "
              "flag asked for in the task."),
    dict(cls="T30", what="first-order phase transition at eps_c",
         where="experiments/verification_03/T30_REPORT.md",
         uses=["P1", "P2", "P3", "P5"], not_used=["P4", "P6", "P7", "P8", "P9"],
         layer="dynamical, P4 dropped", instance_of_fixed_point="yes",
         note="Each phase is a Hamiltonian flow; the transition is a codimension-1 matching "
              "surface. L3 shows what was really used was the JUMP (a discontinuous vector field), "
              "not non-smoothness: a C^0 vector field with a kink changes nothing."),
    dict(cls="T32", what="conversion of the inflow by F(K) on the seam",
         where="experiments/cycle2/T32_REPORT.md",
         uses=["P1", "P10"], not_used=["P3", "P4", "P6", "P7", "P8", "P9"],
         layer="curvature identity", instance_of_fixed_point="no",
         note="Content is the algebraic decomposition K = C^2 + 2 R_ab R^ab - R^2/3 with C^2 fixed "
              "by the enclosed mass and the Ricci part non-negative, so conversion cannot lower K. "
              "That is an identity about the Kretschmann scalar, not a statement about fixed points."),
    dict(cls="T33", what="finite future of the external universe as a cutoff",
         where="experiments/cycle2/T33_REPORT.md",
         uses=["P1", "P10"], not_used=["P2", "P3", "P4", "P6", "P7", "P8", "P9"],
         layer="time scale", instance_of_fixed_point="no",
         note="A comparison of two numbers (time to Planck curvature on the seam vs any end of "
              "the universe). It reduces to T1's inequality with tau = t_evaporation, so it is "
              "downstream of T1, not an independent structural closure."),
]


def main():
    say("=" * 78)
    say("T36 (a).  PREMISES, FORMALISED")
    say("=" * 78)
    for k, v in PREMISES.items():
        say(f"{k}  {v['name']}")
        say(f"    {v['statement']}")
    say(NL + "=" * 78)
    say("T36 (a).  WHICH CLOSURE USES WHICH PREMISE")
    say("=" * 78)
    say(f"{'class':<11} {'uses':<34} {'fixed-point instance?':<26} layer")
    say("-" * 100)
    for u in USAGE:
        say(f"{u['cls']:<11} {','.join(u['uses']):<34} {u['instance_of_fixed_point']:<26} {u['layer']}")
    say(NL + "Per-class notes:")
    for u in USAGE:
        say(f"  {u['cls']} ({u['where']}): {u['note']}")

    say(NL + "=" * 78)
    say("PREMISES USED BY EXACTLY ONE CLASS (the audit the task asks for)")
    say("=" * 78)
    counts = {}
    for u in USAGE:
        for p in u["uses"]:
            counts[p] = counts.get(p, 0) + 1
    singles = {p: c for p, c in counts.items() if c == 1}
    for p, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        nm = PREMISES.get(p, dict(name=p))["name"]
        say(f"  {p:<6} {nm:<32} used by {c} class(es)")
    say("")
    say("  P8 (second law) -- used by T29 alone.  LOAD-BEARING: it is the premise that closes")
    say("     the only escape a local source has (a Theta-dependent law), and nothing else in")
    say("     the programme rests on it.  Without it the no-go is false: T29's zeta < 0 medium")
    say("     and t36_loopholes L1's cubic law both produce genuine attractors.")
    say("  positivity alpha_n >= 0 -- used by BH-P's E4 alone.  Load-bearing for E4's stronger")
    say("     claim (impossibility, not rarity) and NOT needed by the general theorem.")
    say("  P9 (Lagrangian origin) -- used by T26 alone.  REDUNDANT: theorem A3 covers every")
    say("     bounded constitutive law, Lagrangian or not.")
    say("  P7 (energy conditions) -- used by NO closure of the selection question.  The no-go")
    say("     holds without any energy condition; NEC is used elsewhere in the programme (T24,")
    say("     the existence of monotone profiles), not here.  This STRENGTHENS the theorem.")
    say("  P3 (autonomy) -- stated by T26 and T30, shown redundant in t36_core A2 and")
    say("     t36_loopholes L4.")
    say("  P4 (smoothness) -- stated by T26, dropped by T30.  Shown in L3 to weaken to")
    say("     'the vector field is continuous'; C^1 or C^2 of the constitutive law is not needed.")

    say(NL + "=" * 78)
    say("VERDICT ON THE REFEREE'S CONJECTURE")
    say("=" * 78)
    yes = [u["cls"] for u in USAGE if u["instance_of_fixed_point"].startswith("yes")]
    partly = [u["cls"] for u in USAGE if u["instance_of_fixed_point"].startswith("partly")]
    no = [u["cls"] for u in USAGE if u["instance_of_fixed_point"] == "no"]
    say(f"  instances of the fixed-point statement : {', '.join(yes)}")
    say(f"  partly (fixed point, different space)  : {', '.join(partly)}")
    say(f"  NOT instances                          : {', '.join(no)}")
    say("  So the conjecture is true for the six dynamical/kinematic closures and false for")
    say("  four: T14 and T33 are magnitude arguments, T32 is a curvature identity, E4 is an")
    say("  algebraic impossibility inside a restricted ansatz.  A single theorem covers the")
    say("  first group; the second group is not unified and does not need to be -- those are")
    say("  quantitative closures of specific proposals, not structural statements.")
    json.dump(dict(premises=PREMISES, usage=USAGE, single_use=singles,
                   fixed_point_instances=yes, partly=partly, not_instances=no),
              open(OUT / "t36_premises.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    (LOGS / "t36_premises_log.txt").write_text(NL.join(LOG), encoding="utf-8")
    say(f"{NL}written: {OUT/'t36_premises.json'}, {LOGS/'t36_premises_log.txt'}")


if __name__ == "__main__":
    main()
