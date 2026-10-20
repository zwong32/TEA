"""
Internal partition functions for 16O2 and 16O2+ from Huber & Herzberg constants.
Energies in cm^-1. Reference: Q_int is measured from the X-state potential minimum
(Te scale). Multiply by exp(+C2*E0/T) to shift to the X v=0 reference.

X, a, b of O2 are from p. 498 of the table.
"""
import numpy as np

C2 = 1.438777  # hc/k, K per cm^-1
EV = 8065.544  # cm^-1 per eV

def G(x, we, wexe, weye=0.0):
    return we*x - wexe*x**2 + weye*x**3  # x = v + 1/2

def qrot(T, Bv, sigma=2):
    th = C2*Bv
    return T/(sigma*th)*(1 + th/(3*T) + th**2/(15*T**2))

# name: (g_e, Te, we, wexe, weye, Be, ae, limit_key)
O2_STATES = {
    "X 3Sg-": (3,     0.0,    1580.193, 11.981, 0.0,  1.44563, 0.01593, "gnd"),  # p.498
    "a 1Dg":  (2,  7918.1,    1483.50, 12.9,  0.0,    1.4264,  0.0171,  "gnd"),  # p.498 (wexe in parens)
    "b 1Sg+": (1, 13195.1,    1432.77, 14.00, 0.0,    1.40037, 0.01820, "gnd"),  # p.498
    "c 1Su-": (1, 33057.3,     794.29, 12.736, 0.0,   0.9155,  0.01391, "gnd"),
    "A' 3Du": (6, 34690.0,     850.0,  20.0,  0.0,    0.96,    0.0262,  "gnd"),
    "A 3Su+": (3, 35397.8,     799.07, 12.16, -0.550, 0.9106,  0.01416, "gnd"),
    "B 3Su-": (3, 49793.28,    709.31, 10.65, -0.139, 0.81902, 0.01206, "B"),
}
G0_X = G(0.5, 1580.193, 11.981)
LIMITS_O2 = {"gnd": 41260.0 + G0_X,      # D0 = 5.1156 eV = 41260 cm^-1 above X v=0
             "B":   57127.5 + G0_X}      # B-state convergence limit above X v=0

# O2+: (g_e, Te, we, wexe, weye, Be, ae); spin-orbit handled for X only
O2P_STATES = {
    "X 2Pg":  (4,     0.0,   1904.77, 16.259, 0.0, 1.6913,  0.01976),
    "a 4Piu": (8, 32964.0,   1035.69, 10.39,  0.0, 1.10466, 0.01575),
    "A 2Piu": (4, 40669.3,    898.25, 13.573, 0.0, 1.0617,  0.01936),
    "b 4Sg-": (4, 49552.0,   1196.77, 17.09,  0.0, 1.28729, 0.02206),
}
G0_XP = G(0.5, 1904.77, 16.259)
LIMIT_O2P = 6.663*EV + G0_XP                       # D0 = 6.663 eV, above O2+ X minimum
E_ION = 12.071*EV - G0_XP + G0_X                   # O2+ X minimum above O2 X minimum

def q_state(T, g, Te, we, wexe, weye, Be, ae, limit, sigma=2, so=None):
    """Sum over v with analytic rotational sum. Stops at dissociation limit
    or where the polynomial turns over."""
    q, v, Gprev = 0.0, 0, -1.0
    while v < 300:
        x = v + 0.5
        Gv = G(x, we, wexe, weye)
        if Te + Gv > limit or Gv <= Gprev:
            break
        Bv = Be - ae*x
        if Bv <= 0:
            break
        q += qrot(T, Bv, sigma)*np.exp(-C2*(Te + Gv)/T)
        Gprev, v = Gv, v + 1
    return g*q, v - 1  # v_max returned for inspection

def Q_O2(T, ref="min", detail=False):
    parts = {}
    for name, (g, Te, we, wexe, weye, Be, ae, lk) in O2_STATES.items():
        parts[name], vmax = q_state(T, g, Te, we, wexe, weye, Be, ae, LIMITS_O2[lk])
    Q = sum(parts.values())
    if ref == "v0":
        Q *= np.exp(C2*G0_X/T)
        parts = {k: v*np.exp(C2*G0_X/T) for k, v in parts.items()}
    return (Q, parts) if detail else Q

def Q_O2plus(T, ref="O2min"):
    """Referenced to the O2 X minimum (includes the ionization offset E_ION)."""
    Q = 0.0
    for name, (g, Te, we, wexe, weye, Be, ae) in O2P_STATES.items():
        if name == "X 2Pg":  # split into 2Pi_1/2 (0) and 2Pi_3/2 (+197.3 cm^-1)
            for off in (0.0, 197.3):
                q, _ = q_state(T, 2, Te + off, we, wexe, weye, Be, ae, LIMIT_O2P)
                Q += q#*np.exp(-C2*E_ION/T)
        else:
            q, _ = q_state(T, g, Te, we, wexe, weye, Be, ae, LIMIT_O2P)
            Q += q#*np.exp(-C2*E_ION/T)
    return Q

if __name__ == "__main__":
    print(f"G0(X)={G0_X:.1f}  E_ion(min-min)={E_ION:.0f} cm^-1")
    for T in (300, 1000, 2000, 4000, 6000, 8000, 10000):
        Q, parts = Q_O2(T, detail=True)
        top = max(parts, key=parts.get)
        excited = 1 - parts["X 3Sg-"]/Q
        print(f"T={T:6d}  Q_O2={Q:12.4e}  excited-state share={excited:6.3%}  "
              f"Q_O2+={Q_O2plus(T):10.3e}")
