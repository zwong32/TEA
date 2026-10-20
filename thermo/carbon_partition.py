"""
Electronic partition functions for C I ... C VI from Moore, Atomic Energy Levels (vol. I).

Q_el(T) = sum_i g_i exp(-hc E_i / kT), with E_i (cm^-1) measured from each species' OWN ground
level (so add the ionization energy when forming Saha / equilibrium constants).

Construction
  1. Explicit levels: transcribed from the tables, g = sum(2J+1) over J sharing an energy.
     Only terms that are COMPLETE (every level of the shell present) are used explicitly.
  2. Rydberg shells above that: quantum-defect model
        E(n,l) = IP - Z^2 R / (n - delta_l)^2,   g(n,l) = g_core * 2(2l+1)
     with delta_l fitted to the listed levels (see DELTAS comments). Replaces the incomplete /
     cropped high-n portions of the tables.
  3. Cutoff: only states with E < IP - dchi are counted (dchi = ionization-potential lowering).
Autoionizing states above the IP (e.g. C I 2p3 1D, 1S, 1P) are excluded.
"""
import numpy as np

C2 = 1.438777           # hc/k  [K cm]
R_C = 109732.286        # Rydberg constant for 12C [cm^-1] (C VI page)

# ---------------- explicit levels: (E [cm^-1], g) ----------------
C1 = [(0.0,1),(16.4,3),(43.5,5),(10193.70,5),(21648.4,1),(33735.2,5),
      (60333.80,1),(60353.00,3),(60393.52,5),(61982.20,3),
      (64088.56,7),(64093.19,5),(64092.01,3),
      (68858,3),(69689.79,3),(69710.99,5),(69744.40,7),(70744.26,3),
      (71352.81,1),(71365.23,3),(71385.70,5),(72611.06,5),(73976.23,1),
      (75256.3,9)]
C2L = [(0.0,2),(64.0,4),(43000.2,2),(43021.8,4),(43050.7,6),(74930.9,6),(74933.2,4),
       (96494.1,2),(110625.1,2),(110666.3,4),(116537.88,2),(131724.68,2),(131735.81,4),
       (142024.4,4),(145549.99,4),(145551.44,6),(150462.8,6),(150467.9,4),(157234.43,2),
       (162518.70,2),(162524.62,4),(166964.70,2),(166988.46,4),(167033.43,6),
       (168123.92,4),(168124.33,6),(168731.6,2),(168750.2,4),(168979.05,14)]
C3 = [(0.0,1),(52315.0,1),(52338.0,3),(52394.8,5),(102351.4,3),
      (137374.0,1),(137403.4,3),(137450.5,5),(145875.1,5),(182520.2,1),
      (238160.7,3),(247169.5,1),(258931.4,3),(259653.8,1),(259659.3,3),(259672.1,5),
      (269957.6,3),(269959.7,5),(269962.9,7),(276482.7,5)]   # triplet offset x taken as 0
C4 = [(0.0,2),(64484.2,2),(64591.3,4),(302847.9,2),(320048.5,2),(320080.0,4),
      (324880.2,4),(324890.9,6),(401346.7,2),(408308.9,2),(408322.2,4),
      (410333.8,4),(410338.2,6),(410434.1,14)]

# ---------------- species table ----------------
# name: dict(z, IP, levels, core_g, deltas{l:delta}, n_start(l))
SPECIES = {
    # CI, CII etc is called "sepctroscopic notation" for the species, e.g. C I = neutral carbon, C II = singly ionized carbon, etc. blehhhhhh
    
    "C I":   dict(z=0, IP=90878.3,  lev=C1,  core_g=6, dl={0:1.05,1:0.72,2:0.07}, # C neutral cabon
                  n0=lambda l: 3 if l == 2 else 4),
    "C II":  dict(z=1, IP=196659.0, lev=C2L, core_g=1, dl={0:0.66,1:0.41,2:0.08,3:0.02}, # C+ singly ionized carbon
                  n0=lambda l: 5),
    "C III": dict(z=2, IP=386159.7, lev=C3,  core_g=2, dl={0:0.42,1:0.20,2:0.05}, # C2+ doubly ionized carbon
                  n0=lambda l: 4),
    "C IV":  dict(z=3, IP=520177.8, lev=C4,  core_g=1, dl={0:0.157,1:0.04}, # C3+ triply ionized carbon
                  n0=lambda l: 5),
    "C V":   dict(z=4, IP=3162450.0, lev=[(0.0,1)], core_g=None, dl={}, n0=None), # C4+ quatriply ionized carbon
    "C VI":  dict(z=5, IP=3951950.0, lev=[(0.0,2)], core_g=None, dl={}, n0=None),# C5+ etc. 
}

def shells(sp, dchi):
    """Yield (E, g) for the quantum-defect Rydberg shells that pass the cutoff."""
    if sp["core_g"] is None:
        return
    Z = sp["z"] + 1
    IP = sp["IP"]
    nmax = min(400, int(Z*np.sqrt(R_C/max(dchi, 1.0))) + 3)
    for n in range(3, nmax + 1):
        for l in range(0, n):
            if n < sp["n0"](l):
                continue
            d = sp["dl"].get(l, 0.0)
            E = IP - Z**2*R_C/(n - d)**2
            if E < IP - dchi:
                yield E, sp["core_g"]*2*(2*l + 1)

def Q_el(name, T, dchi=None, d1=1000.0, parts=False):
    """dchi: IP lowering [cm^-1]; default (z+1)*d1 (Debye-type scaling with ion charge)."""
    sp = SPECIES[name]
    if dchi is None:
        dchi = (sp["z"] + 1)*d1
    cut = sp["IP"] - dchi
    q_exp = sum(g*np.exp(-C2*E/T) for E, g in sp["lev"] if E < cut)
    q_ryd = sum(g*np.exp(-C2*E/T) for E, g in shells(sp, dchi))
    return (q_exp, q_ryd) if parts else q_exp + q_ryd

def debye_lowering(T, ne_m3, z):
    """Debye-Huckel IP lowering [cm^-1] for a species of charge z (electrons + equal ion density)."""
    e2_4pieps0 = 2.307077e-28            # J m
    kB = 1.380649e-23; eps0 = 8.8541878e-12; e = 1.602177e-19; hc = 1.986446e-23
    lam = np.sqrt(eps0*kB*T/(2*ne_m3*e**2))
    return (z + 1)*e2_4pieps0/lam/hc#/100.0   # J -> cm^-1

if __name__ == "__main__":
    Ts = (300, 3000, 6000, 10000, 15000, 20000, 25000)
    for name in SPECIES:
        print(name)
        for d1 in (500.0, 1000.0, 2000.0):
            row = "  ".join(f"{Q_el(name, T, d1=d1):9.4g}" for T in Ts)
            print(f"  d1={d1:5.0f}  {row}")
    print("T:", Ts)
    q, r = Q_el("C I", 10000, parts=True)
    print("C I @10000 K: explicit %.3f  Rydberg %.3f" % (q, r))
