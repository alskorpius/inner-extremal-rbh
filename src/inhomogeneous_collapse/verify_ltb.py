"""Independent symbolic check of the formulas used in ltb_bounce.py (sympy).

Metric in synchronous comoving coordinates: ds^2 = -dtau^2 + A(tau,r)^2 dr^2 + R(tau,r)^2 dOmega^2, G = c = 1.
We check:
 1. For A = R'/sqrt(1+2E(r)) the component G_{tau r} = 0 identically (no energy flux in the comoving frame).
 2. Misner-Sharp mass m = R(1 + Rdot^2 - R'^2/A^2)/2 = R(Rdot^2 - 2E)/2, and
    G^tau_tau = -8 pi rho with rho = m'/(4 pi R^2 R');  G^r_r = 8 pi p_r with p_r = -mdot/(4 pi R^2 Rdot);
    G^theta_theta = 8 pi p_perp with p_perp = p_r + R p_r'/(2 R').
 3. Kretschmann invariant for ARBITRARY A, R: K = 48 Psi2^2 + 2 R_ab R^ab - Rs^2/3, where
    Psi2 = -m/R^3 + (4 pi/3)(rho - p_r + p_perp), rho, p_r, p_perp -- from the Einstein tensor (numerical check at random points).
 4. Hayward-Kodama dynamical surface gravity kappa = (1/2) Box_2 R = (1/2)(-Rddot - Rdot Rdot'/R' + E'/R')
    and its static limit f'/2 for the metric f = 1 - 2m(R)/R written in geodesic coordinates (check via the Schwarzschild
    limit: a shell falling freely from rest in Schwarzschild: E = -M/r, R(tau,r) -- a cycloid).
Run: python src/inhomogeneous_collapse/verify_ltb.py
"""
import sys
from pathlib import Path

import sympy as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
LOGS = Path(__file__).resolve().parents[2] / "logs" / Path(__file__).resolve().parent.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
log = []


def say(s):
    print(s)
    log.append(s)


t, r, th, ph = sp.symbols("tau r theta phi", real=True)
A = sp.Function("A")(t, r)
R = sp.Function("R")(t, r)
coords = [t, r, th, ph]
g = sp.diag(-1, A**2, R**2, R**2 * sp.sin(th) ** 2)
ginv = g.inv()


def christoffel(g, ginv):
    n = 4
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                G[a][b][c] = sp.simplify(sum(ginv[a, d] * (sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b]) - sp.diff(g[b, c], coords[d])) for d in range(n)) / 2)
    return G


def riemann(G):
    n = 4
    Rm = [[[[0] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    expr = sp.diff(G[a][b][d], coords[c]) - sp.diff(G[a][b][c], coords[d])
                    expr += sum(G[a][c][e] * G[e][b][d] - G[a][d][e] * G[e][b][c] for e in range(n))
                    Rm[a][b][c][d] = sp.simplify(expr)
    return Rm


say("Computing the connection and Riemann tensor for the general metric -dtau^2 + A^2 dr^2 + R^2 dOmega^2 ...")
Gam = christoffel(g, ginv)
Rm = riemann(Gam)
n = 4
Ric = sp.zeros(4)
for b in range(n):
    for d in range(n):
        Ric[b, d] = sp.simplify(sum(Rm[a][b][a][d] for a in range(n)))
Rs = sp.simplify(sum(ginv[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
Gmix = sp.zeros(4)  # G^a_b
for a in range(n):
    for b in range(n):
        Gmix[a, b] = sp.simplify(sum(ginv[a, c] * Ric[c, b] for c in range(n)) - sp.Rational(1, 2) * Rs * (1 if a == b else 0))
Gdown_tr = sp.simplify(sum(g[0, c] * Gmix[c, 1] for c in range(n)))

# Kretschmann: K = R_{abcd} R^{abcd}
Rdown = [[[[sum(g[a, e] * Rm[e][b][c][d] for e in range(n)) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
K = 0
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                Rup = sum(ginv[b, f1] * ginv[c, f2] * ginv[d, f3] * Rm[a][f1][f2][f3] for f1 in range(n) for f2 in range(n) for f3 in range(n))
                K += Rdown[a][b][c][d] * Rup
K = sp.simplify(K)
say("done.")

# --- 1, 2: LTB gauge
E = sp.Function("E")(r)
A_ltb = sp.diff(R, r) / sp.sqrt(1 + 2 * E)
subs_ltb = {A: A_ltb}
Gtr = sp.simplify(Gdown_tr.subs(subs_ltb).doit())
say(f"1. G_(tau r) for A = R'/sqrt(1+2E): {Gtr}")
m = R * (sp.diff(R, t) ** 2 - 2 * E) / 2
rho = sp.diff(m, r) / (4 * sp.pi * R**2 * sp.diff(R, r))
p_r = -sp.diff(m, t) / (4 * sp.pi * R**2 * sp.diff(R, t))
p_perp = p_r + R * sp.diff(p_r, r) / (2 * sp.diff(R, r))
c1 = sp.simplify((Gmix[0, 0].subs(subs_ltb).doit() + 8 * sp.pi * rho))
c2 = sp.simplify((Gmix[1, 1].subs(subs_ltb).doit() - 8 * sp.pi * p_r))
c3 = sp.simplify((Gmix[2, 2].subs(subs_ltb).doit() - 8 * sp.pi * p_perp))
say(f"2. G^t_t + 8 pi m'/(4 pi R^2 R') = {c1}")
say(f"   G^r_r - 8 pi (-mdot/(4 pi R^2 Rdot)) = {c2}")
say(f"   G^th_th - 8 pi (p_r + R p_r'/(2R')) = {c3}")
# Misner-Sharp mass via the general formula
m_gen = R * (1 - sum(ginv[a, b] * sp.diff(R, coords[a]) * sp.diff(R, coords[b]) for a in range(n) for b in range(n))) / 2
say(f"   m_MS(general) - R(Rdot^2-2E)/2 in the LTB gauge: {sp.simplify(m_gen.subs(subs_ltb).doit() - m)}")

# --- 3: Kretschmann formula via Psi2 for arbitrary A, R -- numerical check at random points
rho_g = -Gmix[0, 0] / (8 * sp.pi)
pr_g = Gmix[1, 1] / (8 * sp.pi)
pp_g = Gmix[2, 2] / (8 * sp.pi)
Psi2 = -m_gen / R**3 + sp.Rational(4, 3) * sp.pi * (rho_g - pr_g + pp_g)
RicMix = sp.zeros(4)
for a in range(n):
    for b in range(n):
        RicMix[a, b] = sum(ginv[a, c] * Ric[c, b] for c in range(n))
RabRab = sum(RicMix[a, b] * RicMix[b, a] for a in range(n) for b in range(n))
K_formula = 48 * Psi2**2 + 2 * RabRab - Rs**2 / 3
import random
random.seed(1)
maxdev = 0.0
for k in range(6):
    a0, a1, a2, b0, b1, b2 = [random.uniform(0.5, 2.0) for _ in range(6)]
    Atest = a0 + a1 * sp.sin(a2 * t + r) + sp.Rational(1, 3) * r**2 * sp.exp(-t / 3)
    Rtest = b0 * r + b1 * sp.cos(b2 * t) * r**2 + sp.Rational(1, 5) * t * r
    pt = {t: random.uniform(0.3, 1.5), r: random.uniform(0.5, 1.5), th: 0.7}
    Kv = K.subs({A: Atest, R: Rtest}).doit().subs(pt).evalf(30)
    Kf = K_formula.subs({A: Atest, R: Rtest}).doit().subs(pt).evalf(30)
    dev = abs((Kv - Kf) / Kv)
    maxdev = max(maxdev, float(dev))
    say(f"3. random point {k}: K(Riemann) = {sp.N(Kv, 12)}, K(48Psi2^2+2RabRab-Rs^2/3) = {sp.N(Kf, 12)}, relative difference {float(dev):.1e}")
say(f"   max relative difference over 6 points: {maxdev:.1e}")

# --- 4: Box_2 R and the surface gravity
box2 = (1 / A) * (-sp.diff(A * sp.diff(R, t), t) + sp.diff(sp.diff(R, r) / A, r))
box2_ltb = sp.simplify(box2.subs(subs_ltb).doit())
target = -sp.diff(R, t, 2) - sp.diff(R, t) * sp.diff(R, t, r) / sp.diff(R, r) + sp.diff(E, r) / sp.diff(R, r)
say(f"4. Box_2 R - (-Rddot - Rdot Rdot'/R' + E'/R') in the LTB gauge: {sp.simplify(box2_ltb - target)}")
# Schwarzschild in geodesic coordinates (infall from rest with R(0,r)=r): E=-M/r, cycloid parametrized by eta.
# Numerical check: kappa = Box_2 R / 2 at R = 2M equals 1/(4M).
Msym = sp.Symbol("M", positive=True)
eta = sp.Symbol("eta")
# R = (r/2)(1+cos eta), tau = (r^{3/2}/(2 sqrt(2M))) (eta + sin eta)   [from R = M/(-2E)(1 - cos eta') at eta' = pi - eta]
Rcyc = (r / 2) * (1 + sp.cos(eta))
taucyc = r ** sp.Rational(3, 2) / (2 * sp.sqrt(2 * Msym)) * (eta + sp.sin(eta))
# derivatives with respect to tau at fixed r and with respect to r at fixed tau, via the Jacobian (tau, r) <- (eta, r)
dtau_deta = sp.diff(taucyc, eta)
dtau_dr = sp.diff(taucyc, r)
Rdot = sp.diff(Rcyc, eta) / dtau_deta
Rprime = sp.diff(Rcyc, r) - Rdot * dtau_dr          # d/dr|_tau = d/dr|_eta - (dtau/dr|_eta) d/dtau
Rddot = sp.diff(Rdot, eta) / dtau_deta
Rdotprime = sp.diff(Rdot, r) - Rddot * dtau_dr
Esch = -Msym / r
box_s = -Rddot - Rdot * Rdotprime / Rprime + sp.diff(Esch, r) / Rprime
# point: horizon R = 2M on the shell r = 6M: cos eta = 2*2M/6M - 1 = -1/3
vals = {Msym: 1, r: 6, eta: sp.acos(sp.Rational(-1, 3))}
kap_s = sp.N(box_s.subs(vals) / 2, 15)
say(f"   Schwarzschild, shell r=6M at R=2M: kappa = Box_2 R/2 = {kap_s} (expected 1/(4M) = 0.25); "
    f"E(r) equation: Rdot^2 - 2M/R - 2E = {sp.N((Rdot**2 - 2*Msym/Rcyc - 2*Esch).subs(vals), 10)}")
say(f"   m_MS at the same point: {sp.N((Rcyc*(Rdot**2-2*Esch)/2).subs(vals), 12)} (expected M=1)")
(LOGS / "verify_ltb.txt").write_text("\n".join(log), encoding="utf-8")
print("->", OUT / "verify_ltb.txt")
