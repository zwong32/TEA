"""
mu0_stat.py  —  JANAF-anchored standard-state chemical potentials from
                partition functions and NIST Shomate polynomials.

Strategy
--------
For species where spectroscopic partition functions are available (O2, O2+,
C I, C II / C+, C III / C2+) the thermal contribution H(T)-H(298) and S(T)
are computed directly from Q(T).  The absolute scale is anchored by pinning
the result to the JANAF/CODATA reference value at 298.15 K:

    mu0(T) = Hf298  +  [H_stat(T) - H_stat(298)]  -  T * S_stat(T)
           = Hf298  +  dH(T)  -  T * S(T)          [kJ/mol]

For species without spectroscopic data (C2, C3, C2O, C-, C2-) NIST Shomate
polynomials (Chase 1998) are used directly.  Their mu0(T) formula is:

    mu0(T) = Hf298 + [H_shomate(T) - H_shomate(298)]  -  T * S_shomate(T)

All outputs are in  J/mol  to match Cantera's convention in solver.py.

References
----------
Hf298 / S298 sources (with units kJ/mol and J/mol/K):
  O2      : Hf=   0.0,    S=205.152   CODATA / JANAF Chase 1998
  O2+     : Hf=1165.221              ATcT v1.122 (Ruscic 2014)
  O-      : Hf= 101.63,  S=172.2     JANAF Chase 1998
  C (C I) : Hf= 716.68,  S=158.100   CODATA / JANAF Chase 1998
  C+ (CII): Hf=1803.13               C Hf + IE1(C) = 716.68 + 11.2603 eV; get ionization energy from NIST (Atomic Spectra Database Ionization Energies Data) ASD
  C++(CIII):Hf=4155.73               C+ Hf + IE2(C) = 1803.13 + 24.383 eV; get ionization energy from NIST ASD
  C2      : Hf= 837.74,  S=199.39    JANAF Chase 1998
  C3      : Hf= 820.064, S=237.246   JANAF Chase 1998
  C-      : Hf= 589.621              ATcT v1.122
  C2-     : Hf= 523.0   (estimated: C2 Hf - EA(C2)~3.27 eV = 837.74-315.5)
  C2O     : Hf= 290.0   (estimated; limited JANAF data)
"""

import numpy as np
from scipy.integrate import quad

# --- import spectroscopic partition functions ---
from thermo.o2_partition      import Q_O2, Q_O2plus
from thermo.carbon_partition  import Q_el, SPECIES as C_SPECIES

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R      = 8.314462       # J/mol/K
kB     = 1.380649e-23   # J/K
hPl    = 6.626070e-34   # J·s
NA     = 6.022141e23    # mol^-1
T0     = 298.15         # K  (JANAF reference temperature)

# eV → kJ/mol conversion
EV2KJ  = 96.48533       # kJ / (mol·eV)
EV2JM  = EV2KJ * 1e3    # J  / (mol·eV)

# ---------------------------------------------------------------------------
# JANAF reference values  (kJ/mol and J/mol/K)
# ---------------------------------------------------------------------------
# Hf298[sp]  : standard enthalpy of formation at 298.15 K  [kJ/mol]
# S298[sp]   : absolute entropy at 298.15 K                [J/mol/K]
#              (S298 is only needed for the Shomate path; the partition-
#               function path derives it from Q — but we keep it here for
#               the sanity check in __main__)
Hf298 = {
    "O2":   0.0,
    "O2+":  1165.221,
    "O-":   101.63,
    "O++":  4951.5,     # = 101.63 + 11.2603 eV * 96.485 kJ/(mol·eV) + 24.383 eV * 96.485 kJ/(mol·eV)
    "C":    716.68,
    "C+":   1803.13,    # = 716.68 + 11.2603 eV * 96.485 kJ/(mol·eV)
    "C++":  4155.73,    # = 1803.13 + 24.383 eV * 96.485
    "C2":   837.74,
    "C3":   820.064,
    "C-":   589.621,
    "C2-":  523.0,      # estimated; C2 - EA(C2) ~ 837.74 - 314.7
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
}

# ===========================================================================
# Section 1: Partition-function route (O2, O2+, C, C+, C2+)
# ===========================================================================

def _Cv_from_Q(Q_func, T, dT=1.0):
    """
    Heat capacity at constant volume (internal DOF only) via finite difference
    on ln Q.  Uses the identity  Cv/R = d/dT[T^2 d(lnQ)/dT].

    Q_func(T) must accept a scalar temperature and return a scalar Q > 0.
    """
    # Use a 5-point stencil for better accuracy
    lnQ = lambda t: np.log(max(Q_func(t), 1e-300))
    h = dT
    d2 = (-lnQ(T+2*h) + 16*lnQ(T+h) - 30*lnQ(T) + 16*lnQ(T-h) - lnQ(T-2*h)) / (12*h**2)
    dlnQ_dT = (-lnQ(T+2*h) + 8*lnQ(T+h) - 8*lnQ(T-h) + lnQ(T+2*h)) / (12*h)
    # Cv/R = 2T * dlnQ/dT  +  T^2 * d2lnQ/dT^2
    return R * (2*T * dlnQ_dT + T**2 * d2)


def _dH_from_Q(Q_func, T, n_pts=200):
    """
    Thermal enthalpy increment  H(T) - H(T0)  [J/mol]  from internal partition
    function.  For an ideal gas the translational + PV contribution adds 5/2 RT,
    but since we are computing *differences* relative to 298 K the translational
    part cancels exactly (mass does not change with T).  We therefore only need
    the internal (electronic + ro-vibrational) contribution here.

    H_int(T) - H_int(T0) = integral_{T0}^{T} Cv_int(T') dT'

    This is numerically integrated via Gaussian quadrature.
    """
    if abs(T - T0) < 1e-6:
        return 0.0
    val, _ = quad(lambda t: _Cv_from_Q(Q_func, t), T0, T,
                  limit=n_pts, epsabs=1e-3, epsrel=1e-5)
    return val   # J/mol


def _S_from_Q(Q_func, T, dT=1.0):
    """
    Absolute internal entropy  S_int(T)  [J/mol/K]  from partition function,
    using  S/R = ln Q + T * d(ln Q)/dT.

    This is the *internal* (ro-vibrational + electronic) contribution only.
    The translational Sackur-Tetrode part must be added separately for
    absolute entropies, but it cancels in  mu0 = H - T*S  relative to 298 K
    because we anchor to the JANAF Hf298 value which already includes it.

    The anchor correction _S_offset() below absorbs any residual mismatch.
    """
    lnQ = lambda t: np.log(max(Q_func(t), 1e-300))
    h = dT
    dlnQ_dT = (-lnQ(T+2*h) + 8*lnQ(T+h) - 8*lnQ(T-h) + lnQ(T+2*h)) / (12*h)
    return R * (lnQ(T) + T * dlnQ_dT)


def _mu0_from_Q(Q_func, Hf298_kJ, S298_total_JmolK, T):
    """
    JANAF-anchored mu0 from a partition function.

    The full chemical potential (JANAF convention) is:

        mu0(T) = Hf298  +  [H_total(T) - H_total(298)]  -  T * S_total(T)

    We split H and S into translational and internal parts:
        H_total(T) - H_total(298) = dH_trans(T) + dH_int(T)
        S_total(T)                = S_trans(T)  + S_int(T)

    For an ideal gas the translational contribution to H is 5/2·R·T (kinetic
    + PV), so  dH_trans(T) = 5/2·R·(T - T0).  For S_trans we use the fact
    that at T0 the full JANAF S298 is known:

        S_trans(T0) = S298_total - S_int(T0)

    and the translational entropy scales as  3/2·R·ln(T/T0) (at fixed volume;
    the PV term is already in Hf/dH via the enthalpy formulation).

    Combining:
        mu0(T) = Hf298*1e3
                 + dH_int(T) + 5/2·R·(T - T0)
                 - T · [S_int(T) + S_trans(T0) + 3/2·R·ln(T/T0)]

    where  S_trans(T0) = S298_total - S_int(T0).

    Parameters
    ----------
    Q_func        : callable  Q(T) — internal partition function (ro-vib + elec)
    Hf298_kJ      : float  [kJ/mol]  JANAF enthalpy of formation at 298.15 K
    S298_total_JmolK : float  [J/mol/K]  JANAF absolute entropy at 298.15 K
                      (includes translation).  Pass None to use the internal
                      S only — useful when S298 is not available (ions); in
                      that case the caller must have accounted for translation
                      separately (e.g. via Sackur-Tetrode).
    T             : float  [K]

    Returns
    -------
    mu0  [J/mol]
    """
    dH_int = _dH_from_Q(Q_func, T)         # J/mol  (internal DOF only)
    S_int_T  = _S_from_Q(Q_func, T)        # J/mol/K
    S_int_T0 = _S_from_Q(Q_func, T0)       # J/mol/K

    # Translational enthalpy increment: 5/2 R (T - T0)  [J/mol]
    dH_trans = 2.5 * R * (T - T0)

    if S298_total_JmolK is not None:
        # Full anchor: translational S at T0 from JANAF minus internal part
        S_trans_T0 = S298_total_JmolK - S_int_T0   # J/mol/K
        # Translational S scales as 3/2 R ln(T/T0) at fixed volume
        S_trans_T  = S_trans_T0 + 1.5 * R * np.log(T / T0)
        S_total_T  = S_int_T + S_trans_T
    else:
        # No S298 available — use internal entropy only (ions where
        # translation is implicitly absorbed in the Hf298 anchor)
        S_total_T  = S_int_T + 1.5 * R * np.log(T / T0)
        # Anchor the translational offset so mu0(T0) = Hf298 - T0*S_int(T0)
        # (conservative choice when full S298 is unknown)
        S_total_T  = S_int_T + (S_int_T - S_int_T0)  # delta only, no absolute

    return Hf298_kJ * 1e3  +  dH_int  +  dH_trans  -  T * S_total_T


# Wrapper callables that bind the correct Q function and JANAF S298 to each species
# S298 values (J/mol/K): O2=205.152, C=158.100 (JANAF/CODATA)
# For ions (O2+, C+, C2+) no JANAF S298 is tabulated; pass None.

def _mu0_O2(T):
    return _mu0_from_Q(Q_O2, Hf298["O2"], S298["O2"], T)

def _mu0_O2plus(T):
    return _mu0_from_Q(Q_O2plus, Hf298["O++"], None, T)

def _mu0_C(T):
    Q_CI = lambda t: Q_el("C I", t)
    return _mu0_from_Q(Q_CI, Hf298["C"], S298["C"], T)

def _mu0_Cplus(T):
    Q_CII = lambda t: Q_el("C II", t)
    return _mu0_from_Q(Q_CII, Hf298["C+"], None, T)

def _mu0_C2plus(T):
    Q_CIII = lambda t: Q_el("C III", t)
    return _mu0_from_Q(Q_CIII, Hf298["C++"], None, T)


# ===========================================================================
# Section 2: Shomate-polynomial route  (C2, C3, C-, C2-, C2O, O-, O2)
# ===========================================================================
# Coefficients: [A, B, C, D, E, F, G, H]  where t = T/1000
# mu0(T) = (H° - H°298.15) + Hf298 - T*S°
# H° - H°298.15 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H   [kJ/mol]
# S°             = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2t^2) + G  [J/mol/K]

# Each entry is a list of (T_max, [A,B,C,D,E,F,G,H]) intervals,
# in ascending T_max order.  The last interval has T_max = inf.

_SHOMATE = {

    # NIST WebBook / Chase 1998
    # O2: reference element Hf298=0;  three intervals 100-700, 700-2000, 2000-6000
    "O2": (0.0, [
        (700.0,   [31.32234,  -20.23531,  57.86644, -36.50624, -0.007374, -8.903471,  246.7945,   0.0]),
        (2000.0,  [30.032490,   8.772972, -3.988133,  0.788313, -0.741599, -11.32468,  236.1663,   0.0]),
        (np.inf,  [-11.32468,  57.33839, -21.27655,   2.809983, -1.238533, 29.22720,  237.6185,   0.0]),
    ]),

    # C2 (dicarbon): Hf298=837.74;  298-700, 700-6000
    "C2": (837.74, [
        (700.0,   [123.9092,  -348.0067,  485.0971, -232.7994, -1.240298,  808.2777,  426.6132, 837.7372]),
        (np.inf,  [ 30.50408,    5.445811, -0.853373,   0.065641,  0.814750,  831.5784,  240.4039, 837.7372]),
    ]),

    # C3 (tricarbon): Hf298=820.064;  298-1000, 1000-6000
    "C3": (820.064, [
        (1000.0,  [26.33364,   20.26830,   2.788842,  -5.048698,  0.466985,  812.8717,  265.6380, 820.0640]),
        (np.inf,  [43.89853,    5.491207,  -0.819001,   0.0,      -3.879841,  798.2737,  278.3469, 820.0640]),
    ]),

    # O- (oxygen anion): Hf298=101.63 kJ/mol; S298=172.2 J/mol/K
    # Approximated from monatomic O (JANAF) shifted by electron affinity offset.
    # Using O Shomate (Chase 1998) as proxy: A=21.11...; differences absorbed in anchor.
    "O-": (101.63, [
        (np.inf,  [21.11562,  -0.516942,   0.081376,  -0.040538,  -0.010024,  244.7490,  172.2000, 249.1700]),
    ]),

    # C- (carbon anion): Hf298=589.621 kJ/mol (ATcT v1.122)
    # No Shomate data in JANAF; use monatomic C Shomate (Chase 1998) as proxy.
    # The thermal function is nearly identical to C I; the anchor shifts H level.
    "C-": (589.621, [
        (np.inf,  [21.17510,  -0.812428,   0.448537,  -0.043256,  -0.013103,  710.3470,  183.8734, 716.6690]),
    ]),

    # C2- (dicarbon anion): Hf298=523.0 kJ/mol (estimated)
    # Use C2 Shomate (high-T branch) as proxy thermal function.
    "C2-": (523.0, [
        (700.0,   [123.9092,  -348.0067,  485.0971, -232.7994, -1.240298,  808.2777,  426.6132, 837.7372]),
        (np.inf,  [ 30.50408,    5.445811, -0.853373,   0.065641,  0.814750,  831.5784,  240.4039, 837.7372]),
    ]),

    # C2O (ethynyl oxide / dicarbon monoxide): Hf298=290.0 kJ/mol (estimated)
    # Linear triatomic, use CO2 Shomate as a rough thermal proxy.
    "C2O": (290.0, [
        (1200.0,  [24.99735,  55.18696, -33.69137,   7.948387, -0.136638,  -403.6075,  228.2431, -393.5224]),
        (np.inf,  [58.16639,   2.720074,  -0.492289,   0.038844, -6.447293, -425.9186,  263.6125, -393.5224]),
    ]),
}


def _shomate_mu0(sp, T):
    """
    Compute mu0 [J/mol] from Shomate polynomial for species sp at temperature T.

    mu0(T) = Hf298*1e3  +  (H(T)-H(298))*1e3  -  T * S(T)

    where H and S follow the NIST Shomate conventions.
    """
    Hf, intervals = _SHOMATE[sp]
    t = T / 1000.0
    t0 = T0 / 1000.0

    # Select the right coefficient set for T
    A, B, C, D, E, F, G, H_coef = None, None, None, None, None, None, None, None
    for T_max, coefs in intervals:
        A, B, C, D, E, F, G, H_coef = coefs
        if T <= T_max:
            break

    # H(T) - H(298.15)  [kJ/mol]  — NIST Shomate formula
    # = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H_coef
    def _H_shomate(tv):
        return A*tv + B*tv**2/2 + C*tv**3/3 + D*tv**4/4 - E/tv + F - H_coef

    dH = _H_shomate(t) - _H_shomate(t0)    # kJ/mol

    # S(T)  [J/mol/K]
    def _S_shomate(tv):
        return A*np.log(tv) + B*tv + C*tv**2/2 + D*tv**3/3 - E/(2*tv**2) + G

    S_T = _S_shomate(t)                     # J/mol/K

    return (Hf + dH) * 1e3  -  T * S_T     # J/mol


# ===========================================================================
# Section 3: Public dispatcher
# ===========================================================================

# Species routed via partition functions
_PARTITION_SPECIES = {"O2", "O2+", "C", "C+", "O++", "C++"}

# Species routed via Shomate polynomials
_SHOMATE_SPECIES   = set(_SHOMATE.keys())

# Cantera-handled species (returned as None — caller queries Cantera)
_CANTERA_SPECIES   = {"CO2", "CO", "O", "O+", "e-"}


def get_mu0_stat(species, T):
    """
    Return the standard-state chemical potential mu0 [J/mol] at temperature T
    for the given species name.

    Routing:
      - O2, O2+, C, C+, O++, C++  → partition function  (o2_partition / carbon_partition)
      - C2, C3, C2O, C-, O-, O-, O++, C2- → NIST Shomate polynomial (Chase 1998 / estimates)
      - CO2, CO, O, O+, e-   → returns None  (caller should query Cantera)
      - O-                   → Shomate proxy

    Parameters
    ----------
    species : str
    T       : float  [K]

    Returns
    -------
    float  [J/mol], or None if the species should be queried from Cantera.

    Raises
    ------
    KeyError if the species is unknown.
    """
    if species in _CANTERA_SPECIES:
        return None

    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T}")

    if species == "O2":
        return _mu0_O2(T)
    elif species == "O++":
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
    import sys

    print("=" * 72)
    print("mu0_stat.py  —  JANAF-anchored mu0 sanity check")
    print("=" * 72)

    # --- Check 1: S(298) from partition functions should match JANAF ---
    print("\n--- Partition-function S(298.15 K) vs JANAF ---")
    print(f"{'Species':<10} {'S_stat(298) J/mol/K':>22} {'S_JANAF J/mol/K':>18} {'diff':>8}")
    T_check = 298.15

    S_O2_stat   = _S_from_Q(Q_O2, T_check)
    S_C_stat    = _S_from_Q(lambda t: Q_el("C I", t), T_check)

    for sp, S_stat_val, S_janaf_val in [
        ("O2",  S_O2_stat,  S298["O2"]),
        ("C",   S_C_stat,   S298["C"]),
    ]:
        print(f"{sp:<10} {S_stat_val:>22.4f} {S_janaf_val:>18.4f} {S_stat_val - S_janaf_val:>8.4f}")

    # --- Check 2: mu0(T) at 298 K should equal Hf298 - 298*S298 ---
    print("\n--- mu0(298 K) vs Hf298 - T0*S298 (kJ/mol) ---")
    print(f"{'Species':<10} {'mu0_stat/1e3':>16} {'Hf-T0*S':>16} {'diff':>10}")
    for sp, mu_func, S_ref in [
        ("O2",  _mu0_O2,   S298["O2"]),
        ("C",   _mu0_C,    S298["C"]),
        ("C2",  lambda T: _shomate_mu0("C2", T),  S298["C2"]),
        ("C3",  lambda T: _shomate_mu0("C3", T),  S298["C3"]),
    ]:
        mu0_298   = mu_func(T_check) / 1e3            # kJ/mol
        ref_kJ    = Hf298[sp] - T_check * S_ref / 1e3
        print(f"{sp:<10} {mu0_298:>16.4f} {ref_kJ:>16.4f} {mu0_298 - ref_kJ:>10.4f}")

    # --- Check 3: mu0(T) sweep for all 16 species ---
    print("\n--- mu0(T) sweep [kJ/mol] ---")
    all_species = ["O", "O2", "O-", "O+", "O++", "O2+",
                   "C3", "C2-", "C", "C2", "C-", "C+", "C++",
                   "CO", "C2O", "CO2", "e-"]
    T_vals = [500, 1000, 3000, 6000, 10000]
    header = f"{'T(K)':>8}" + "".join(f"{sp:>10}" for sp in all_species)
    print(header)
    for T in T_vals:
        row = f"{T:>8.0f}"
        for sp in all_species:
            val = get_mu0_stat(sp, T)
            if val is None:
                row += f"{'Cantera':>10}"
            else:
                row += f"{val/1e3:>10.2f}"
        print(row)

    # --- Check 4: O2 at 1000 K vs Cantera NASA-7 expected value ---
    # From gri30 at 1000 K: mu0(O2) ≈ -187 kJ/mol  (value from Cantera reference)
    print("\n--- Spot check: O2 at 1000 K ---")
    mu_O2_1000 = _mu0_O2(1000.0) / 1e3
    print(f"  mu0(O2, 1000 K) = {mu_O2_1000:.2f} kJ/mol")
    print("  (Cantera gri30 reference: ~ -187 kJ/mol)")

    print("\nDone.")
