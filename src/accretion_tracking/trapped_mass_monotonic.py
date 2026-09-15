"""General class (double-null coordinates): ds^2 = -2 e^{-F(u,v)} du dv + R(u,v)^2 dOmega^2.
Question: can the Misner-Sharp mass on a sphere of fixed R decrease in time inside the trapped region
(R_u < 0, R_v < 0) without violating the energy conditions? We compute d m/dv|_R = m_v - m_u R_v/R_u and express it via
T_uu, T_vv, T_uv (G = c = 1). Run: python src/accretion_tracking/trapped_mass_monotonic.py
"""
import sys
from pathlib import Path

import sympy as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
LOGS = Path(__file__).resolve().parents[2] / "logs" / Path(__file__).resolve().parent.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
u, v, th, ph = sp.symbols("u v theta phi", real=True)
F = sp.Function("F")(u, v)
R = sp.Function("R")(u, v)
x = [u, v, th, ph]
g = sp.Matrix([[0, -sp.exp(-F), 0, 0], [-sp.exp(-F), 0, 0, 0], [0, 0, R**2, 0], [0, 0, 0, R**2 * sp.sin(th) ** 2]])
gi = g.inv()
n = 4
Gam = [[[sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d])) for d in range(n)) / 2) for c in range(n)] for b in range(n)] for a in range(n)]
Ric = sp.zeros(4)
for b in range(n):
    for d in range(n):
        Ric[b, d] = sp.simplify(sum(sp.diff(Gam[a][b][d], x[a]) - sp.diff(Gam[a][b][a], x[d]) + sum(Gam[a][a][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][a] for e in range(n)) for a in range(n)))
Rs = sp.simplify(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
G = sp.simplify(Ric - Rs * g / 2)
m = R * (1 - sum(gi[a, b] * sp.diff(R, x[a]) * sp.diff(R, x[b]) for a in range(n) for b in range(n))) / 2
m = sp.simplify(m)
Ru, Rv = sp.diff(R, u), sp.diff(R, v)
m_u, m_v = sp.diff(m, u), sp.diff(m, v)
Tuu, Tvv, Tuv = G[0, 0] / (8 * sp.pi), G[1, 1] / (8 * sp.pi), G[0, 1] / (8 * sp.pi)
# known identities: m_u = 4 pi R^2 e^{F} (T_uv R_u - T_uu R_v), m_v = 4 pi R^2 e^{F} (T_uv R_v - T_vv R_u)  (we verify)
c1 = sp.simplify(m_u - 4 * sp.pi * R**2 * sp.exp(F) * (Tuv * Ru - Tuu * Rv))
c2 = sp.simplify(m_v - 4 * sp.pi * R**2 * sp.exp(F) * (Tuv * Rv - Tvv * Ru))
print("m_u - 4 pi R^2 e^F (T_uv R_u - T_uu R_v) =", c1)
print("m_v - 4 pi R^2 e^F (T_uv R_v - T_vv R_u) =", c2)
dm_dv_at_R = sp.simplify(m_v - m_u * Rv / Ru)
expr = sp.simplify(dm_dv_at_R / (4 * sp.pi * R**2 * sp.exp(F)))
# expectation: (T_uu R_v^2 - T_vv R_u^2)/R_u  -> inside the trapped region (R_u<0, R_v<0): = -(T_uu R_v^2 + T_vv R_u^2)/|R_u| <= 0 under NEC ?!
target = (Tuu * Rv**2 - Tvv * Ru**2) / Ru
print("dm/dv|_R /(4 pi R^2 e^F) - (T_uu R_v^2 - T_vv R_u^2)/R_u =", sp.simplify(expr - target))
txt = f"""Double-null coordinates ds^2 = -2 e^-F du dv + R^2 dOmega^2, m = R(1 + 2 e^F R_u R_v)/2; u is retarded
(d_u is tangent to INGOING rays v = const), v is advanced (d_v is tangent to OUTGOING rays u = const).
T_vv is the ingoing flux (propagates along d_u), T_uu is the outgoing flux (propagates along d_v).
Identities (verified with sympy, residuals {c1}, {c2}):
  m_u = 4 pi R^2 e^F (T_uv R_u - T_uu R_v),   m_v = 4 pi R^2 e^F (T_uv R_v - T_vv R_u).
Change of the Misner-Sharp mass on a sphere of fixed R along advanced time:
  dm/dv|_R = m_v - m_u R_v/R_u = 4 pi R^2 e^F (T_uu R_v^2 - T_vv R_u^2)/R_u   (residual {sp.simplify(expr - target)}); T_uv drops out.
Everywhere R_u < 0 (both outside and inside the trapped region):
  dm/dv|_R = 4 pi R^2 e^F (T_vv R_u^2 - T_uu R_v^2)/|R_u|:
an ingoing flux (T_vv >= 0) increases m inside the sphere R, an outgoing one (T_uu >= 0) decreases it -- also inside the trapped region.
Consequence for a tracking law: the required decrease of m at fixed R < R* can be arranged WITHOUT negative energy --
by positive outgoing emission of the medium T_uu R_v^2 > T_vv R_u^2 (the ingoing Vaidya class, T_uu = 0, allows only T_vv < 0).
But inside the trapped region outgoing rays move toward smaller R and asymptotically approach the inner horizon R_- from outside:
the emitted energy does not leave the region and accumulates near R_-. The claim that "negative energy is unavoidable inside the
trapped region" from the first version of REPORT.md is INCORRECT in the general class; what holds instead is: either a negative
ingoing flux (Vaidya class), or a positive outgoing emission of the medium accumulating at the inner horizon (a two-flux class; self-consistency not checked).
"""
(LOGS / "trapped_mass_monotonic.txt").write_text(txt, encoding="utf-8")
print(txt)
