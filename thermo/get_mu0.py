# get_mu0.py

# Implemented review recommendations: corrected five-point derivatives, direct H/S formulas with fixed-pressure translation, mandatory total entropy anchors, O2+ routing and explicit reference selection, bounded Shomate evaluation, corrected O2 coefficients, and reviewed C2- anchors with proxy warning. Carbon Method B is primary; Method A remains selectable for comparison. Use electron_mu0 with Method B and match the standard pressure in the equilibrium solver. Inputs C++, C III denote atomic C2+; the string C2+ is not silently reinterpreted.

# The original transcribed level lists and quantum-defect model have not been independently revalidated against a complete spectrum. Default explicit-only results are a finite isolated-species baseline, not a converged infinite bound-state sum. Rydberg sensitivity is opt-in and not a self-consistent nonideal plasma model. O2+ S298 is model-derived. C+/C++ Method A values and C2- anchors retain the prior review's values; independent source reconfirmation of these anchors remains outstanding. Proxy thermal functions for anions/C2O remain approximate, and their temperature range is capped at 6000 K. Do not treat the 298 K anchor check as independent validation. Small raw Shomate boundary jumps are retained and reported rather than hidden.

# O2 Shomate source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1&Ta=&Type=JANAFG
# Ionization source: https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html (rounded IE values used).

# Run: python3 validate_mu0.py
# Requires Python, NumPy. Copy code blocks into the stated .py filenames in one directory.

"""Standard chemical potentials [J/mol], p0=1 bar.
C I=C, C II=C+, C III=C++; C2+ is a different molecular species.
Carbon defaults to a shared statistical energy reference (Method B).
method='anchored' uses the retained thermochemical H298/S298 anchors (Method A).
Cantera molar properties are J/kmol: divide by 1000 before combining.
Finite explicit level lists are a baseline, not a complete high-T spectrum.
Rydberg extension is opt-in; no Coulomb/nonideal correction is included.
"""

import numpy as np
import warnings

# --- import spectroscopic partition functions ---
from o2_partition      import Q_O2, Q_O2plus
from carbon_partition  import Q_el, SPECIES as C_SPECIES

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R      = 8.31446261815324       # J/mol/K
kB     = 1.380649e-23   # J/K
hPl    = 6.62607015e-34   # J·s
NA     = 6.02214076e23    # mol^-1
T0     = 298.15         # K  (JANAF reference temperature)

# eV → kJ/mol conversion
EV2KJ  = 96.48533       # kJ / (mol·eV)
EV2JM  = EV2KJ * 1e3    # J  / (mol·eV)

# ---------------------------------------------------------------------------
# JANAF reference values  (kJ/mol and J/mol/K)
# ---------------------------------------------------------------------------
# Hf298[sp]  : standard enthalpy of formation at 298.15 K  [kJ/mol]
# S298[sp]   : absolute entropy at 298.15 K                [J/mol/K]
# S298 is required by the anchored route; never omit translation for ions.
Hf298 = {
    "O2":   0.0,
    "O2+":  1165.221,
    "O-":   101.63,
    "C":    716.68,
    "C+":   1803.13,    # = 716.68 + 11.2603 eV * 96.485 kJ/(mol·eV)
    "C++":  4155.73,    # = 1803.13 + 24.383 eV * 96.485
    "C2":   837.74,
    "C3":   820.064,
    "C-":   589.621,
    "C2-":  443.504,    # reviewed anchor; thermal proxy still approximate
    "C2O":  290.0,      # estimated; limited thermochemical data
}

# Standard molar entropy at 298.15 K, 1 bar
# Units: J/(mol*K)

S298 = {
    "O2":   205.152,
    "O-":   157.790,
    "C":    158.100,  # C I
    "C+":   154.660,  # C II - NIST/JANAF
    "C++":  139.846,  # C III - calculated estimate
    "C2":   199.382,
    "C3":   237.246,
    "C2-":  196.597,
}

# ===========================================================================
# Section 1: Partition-function route (O2, O2+, C, C+, C++)
# ===========================================================================

P0 = 1e5
ME = 9.1093837139e-31
M_C = 12.0e-3 / NA  # 12C, consistent with the level model
# NIST ASD isolated-atom ionization energies, eV (rounded).
IE_EV = (11.260288, 24.383143)
CARBON_NAMES = {"C": "C I", "C+": "C II", "C++": "C III"}


def _check_T(T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")


def _lnQ(Q_func, T):
    _check_T(T)
    q = Q_func(T)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("Partition function must be finite and positive")
    return np.log(q)


def _dlnQ(Q_func, T, dT=None):
    _check_T(T)
    step = min(dT if dT is not None else max(0.01, T*1e-4), T/4)
    if step <= 0:
        raise ValueError("Derivative step must be positive")
    f = lambda t: _lnQ(Q_func, t)
    return (-f(T+2*step)+8*f(T+step)-8*f(T-step)+f(T-2*step))/(12*step)


def _S_from_Q(Q_func, T, dT=None):
    """Internal entropy only [J/mol/K]; translation is added separately."""
    return R*(_lnQ(Q_func, T)+T*_dlnQ(Q_func, T, dT))


def _dH_from_Q(Q_func, T, n_pts=None):
    """Internal energy increment; avoids numerical quadrature of Cv."""
    return R*(T*T*_dlnQ(Q_func, T)-T0*T0*_dlnQ(Q_func, T0))


def _Cv_from_Q(Q_func, T, dT=None):
    _check_T(T)
    step = min(dT if dT is not None else max(.01,T*1e-4), T/4)
    f = lambda t: _lnQ(Q_func,t)
    d2=(-f(T+2*step)+16*f(T+step)-30*f(T)+16*f(T-step)-f(T-2*step))/(12*step**2)
    return R*(2*T*_dlnQ(Q_func,T,step)+T*T*d2)


def _mu0_from_Q(Q_func, Hf298_kJ, S298_total_JmolK, T):
    """Method A: direct anchored H-T S, at fixed standard pressure."""
    if S298_total_JmolK is None or not np.isfinite(S298_total_JmolK):
        raise ValueError("A total S298 anchor including translation is required")
    dH=2.5*R*(T-T0)+_dH_from_Q(Q_func,T)
    S=S298_total_JmolK+2.5*R*np.log(T/T0)+_S_from_Q(Q_func,T)-_S_from_Q(Q_func,T0)
    return Hf298_kJ*1000+dH-T*S


def _log_activity_partition(Q_func, mass, T):
    """ln[(2 pi m kT/h^2)^(3/2) kT/p0 Qint], dimensionless."""
    _check_T(T)
    return 1.5*np.log(2*np.pi*mass*kB*T/hPl**2)+np.log(kB*T/P0)+_lnQ(Q_func,T)


def _entropy_stat(Q_func, mass, T):
    return R*(_log_activity_partition(Q_func,mass,T)+2.5+T*_dlnQ(Q_func,T))


def carbon_mu0(species,T,method="statistical",include_rydberg=False,d1=1000.0):
    """Method B uses one neutral H298 anchor and successive IE offsets.
    Electron energy zero is a stationary free electron; Qe=2.
    Method A retains independent ionic H298/S298 anchors for comparison.
    d1 is held constant, and is used only when include_rydberg=True.
    """
    Q=lambda t: Q_el(CARBON_NAMES[species],t,include_rydberg=include_rydberg,d1=d1)
    if method == "anchored":
        return _mu0_from_Q(Q,Hf298[species],S298[species],T)
    if method != "statistical":
        raise ValueError("method must be statistical or anchored")
    neutralQ=lambda t: Q_el("C I",t,include_rydberg=include_rydberg,d1=d1)
    epsilon0=Hf298["C"]*1000-2.5*R*T0-R*T0*T0*_dlnQ(neutralQ,T0)
    charge={"C":0,"C+":1,"C++":2}[species]
    mass=M_C-charge*ME
    energy=epsilon0+sum(IE_EV[:charge])*EV2JM
    return energy-R*T*_log_activity_partition(Q,mass,T)


def electron_mu0(T):
    """Electron chemical potential in the Method B convention [J/mol]."""
    return -R*T*_log_activity_partition(lambda t:2.0,ME,T)


def _mu0_O2(T):
    return _mu0_from_Q(lambda t:Q_O2(t,ref="v0"),Hf298["O2"],S298["O2"],T)


def _mu0_O2plus(T):
    Q=lambda t:Q_O2plus(t,ref="v0")
    # Model-derived total entropy, explicitly labeled rather than fabricated JANAF data.
    S0=_entropy_stat(Q,31.989829239e-3/NA-ME,T0)
    return _mu0_from_Q(Q,Hf298["O2+"],S0,T)


def _mu0_C(T): return carbon_mu0("C",T)
def _mu0_Cplus(T): return carbon_mu0("C+",T)
def _mu0_C2plus(T): return carbon_mu0("C++",T)  # compatibility name: atomic C++


# ===========================================================================
# Section 2: Shomate-polynomial route  (C2, C3, C-, C2-, C2O, O-, O2)
# ===========================================================================
# Coefficients: [A, B, C, D, E, F, G, H]  where t = T/1000
# mu0(T) = (H° - H°298.15) + Hf298 - T*S°
# H° - H°298.15 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H   [kJ/mol]
# S°             = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2t^2) + G  [J/mol/K]

# Each entry is a list of (T_max, [A,B,C,D,E,F,G,H]) intervals,
# in ascending T_max order; no extrapolation beyond 6000 K.

_SHOMATE = {

    # NIST WebBook / Chase 1998
    # O2: reference element Hf298=0;  three intervals 100-700, 700-2000, 2000-6000
    "O2": (0.0, [
        (700.0,   [31.32234,  -20.23531,  57.86644, -36.50624, -0.007374, -8.903471,  246.7945,   0.0]),
        (2000.0,  [30.03235,   8.772972, -3.988133,  0.788313, -0.741599, -11.32468,  236.1663,   0.0]),
        (6000.0,  [20.91111, 10.72071, -2.020498, 0.146449, 9.245722, 5.337651, 237.6185, 0.0]),
    ]),

    # C2 (dicarbon): Hf298=837.74;  298-700, 700-6000
    "C2": (837.74, [
        (700.0,   [123.9092,  -348.0067,  485.0971, -232.7994, -1.240298,  808.2777,  426.6132, 837.7372]),
        (6000.0,  [ 30.50408,    5.445811, -0.853373,   0.065641,  0.814750,  831.5784,  240.4039, 837.7372]),
    ]),

    # C3 (tricarbon): Hf298=820.064;  298-1000, 1000-6000
    "C3": (820.064, [
        (1000.0,  [26.33364,   20.26830,   2.788842,  -5.048698,  0.466985,  812.8717,  265.6380, 820.0640]),
        (6000.0,  [43.89853,    5.491207,  -0.819001,   0.0,      -3.879841,  798.2737,  278.3469, 820.0640]),
    ]),

    # O- (oxygen anion): Hf298=101.63 kJ/mol; S298=172.2 J/mol/K
    # Approximated from monatomic O (JANAF) shifted by electron affinity offset.
    # Using O Shomate (Chase 1998) as proxy: A=21.11...; differences absorbed in anchor.
    "O-": (101.63, [
        (6000.0,  [21.11562,  -0.516942,   0.081376,  -0.040538,  -0.010024,  244.7490,  172.2000, 249.1700]),
    ]),

    # C- (carbon anion): Hf298=589.621 kJ/mol (ATcT v1.122)
    # No Shomate data in JANAF; use monatomic C Shomate (Chase 1998) as proxy.
    # The thermal function is nearly identical to C I; the anchor shifts H level.
    "C-": (589.621, [
        (6000.0,  [21.17510,  -0.812428,   0.448537,  -0.043256,  -0.013103,  710.3470,  183.8734, 716.6690]),
    ]),

    # C2-: reviewed H298/S298 anchors; neutral C2 thermal proxy remains approximate.
    # Use C2 Shomate (high-T branch) as proxy thermal function.
    "C2-": (443.504, [
        (700.0,   [123.9092,  -348.0067,  485.0971, -232.7994, -1.240298,  808.2777,  426.6132, 837.7372]),
        (6000.0,  [ 30.50408,    5.445811, -0.853373,   0.065641,  0.814750,  831.5784,  240.4039, 837.7372]),
    ]),

    # C2O (ethynyl oxide / dicarbon monoxide): Hf298=290.0 kJ/mol (estimated)
    # Linear triatomic, use CO2 Shomate as a rough thermal proxy.
    "C2O": (290.0, [
        (1200.0,  [24.99735,  55.18696, -33.69137,   7.948387, -0.136638,  -403.6075,  228.2431, -393.5224]),
        (6000.0,  [58.16639,   2.720074,  -0.492289,   0.038844, -6.447293, -425.9186,  263.6125, -393.5224]),
    ]),
}


def _shomate_values(coefs,T):
    A,B,C,D,E,F,G,H=coefs
    t=T/1000.0
    return (A*t+B*t*t/2+C*t**3/3+D*t**4/4-E/t+F-H,
            A*np.log(t)+B*t+C*t*t/2+D*t**3/3-E/(2*t*t)+G)


def _shomate_mu0(sp,T):
    _check_T(T)
    lo=100.0 if sp == "O2" else T0
    if not lo <= T <= 6000:
        raise ValueError("Shomate/proxy route is limited to %g..6000 K" % lo)
    Hf,intervals=_SHOMATE[sp]
    coefs=next(c for upper,c in intervals if T <= upper)
    H,S=_shomate_values(coefs,T)
    # Evaluate the reference in its own interval, not the high-T branch.
    H0,S0=_shomate_values(intervals[0][1],T0)
    if sp in {"C-","O-","C2-","C2O"}:
        warnings.warn(sp+": approximate proxy thermal function",RuntimeWarning,stacklevel=2)
        if sp in S298:
            S += S298[sp]-S0
    return (Hf+H-H0)*1000-T*S


# ===========================================================================
# Section 3: Public dispatcher
# ===========================================================================

# Species routed via partition functions
_PARTITION_SPECIES = {"O2", "O2+", "C", "C+", "C++"}

# Species routed via Shomate polynomials
_SHOMATE_SPECIES   = set(_SHOMATE.keys())

# Cantera-handled species (returned as None — caller queries Cantera)
_CANTERA_SPECIES   = {"CO2", "CO", "O", "O+", "e-"}


def get_mu0_stat(species, T, method="statistical", include_rydberg=False, d1=1000.0):
    """Return mu0 in J/mol at 1 bar, or None for external species.
    For Method B use electron_mu0(T) rather than an unmatched electron anchor.
    O++ is atomic oxygen: no oxygen double-ion spectrum is supplied here.
    """
    _check_T(T)
    species={"C I":"C","CI":"C","C II":"C+","CII":"C+", "C III":"C++","CIII":"C++"}.get(species,species)
    if species in CARBON_NAMES:
        return carbon_mu0(species,T,method,include_rydberg,d1)
    if species == "O++":
        raise NotImplementedError("O++ needs its own atomic double-ion thermochemistry")
    if species in _CANTERA_SPECIES:
        return None

    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T}")

    if species == "O2":
        return _mu0_O2(T)
    elif species == "O2+":
        return _mu0_O2plus(T)
    elif species == "C":
        return _mu0_C(T)
    elif species == "C+":
        return _mu0_Cplus(T)
    elif species == "C++":
        return _mu0_C2plus(T)
    elif species in _SHOMATE_SPECIES:
        return _shomate_mu0(species, T)
    else:
        raise KeyError(f"Unknown species '{species}' in mu0_stat.get_mu0_stat()")


def get_mu0_all(species_list, T):
    """
    Vectorised wrapper: returns a numpy array of mu0 [J/mol] for every species
    in species_list.  Cantera species return NaN — replace them after querying
    Cantera.
    """
    out = np.empty(len(species_list))
    for i, sp in enumerate(species_list):
        val = get_mu0_stat(sp, T)
        out[i] = val if val is not None else np.nan
    return out


# ===========================================================================
# Section 4: Sanity check
# ===========================================================================

if __name__ == "__main__":
    for T in (298.15,1000,6000,10000,25000):
        print(T,{sp:round(carbon_mu0(sp,T)/1000,6) for sp in CARBON_NAMES})

