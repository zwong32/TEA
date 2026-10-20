"""
fit_nasa9.py  —  Fit NASA9 polynomial coefficients from mu0_stat.py partition
                 functions / Shomate data, then print yaml-ready species blocks
                 for pasting into gri30_plasma.yaml.

NASA9 Cp/R per interval:
    Cp/R = a1/T^2 + a2/T + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4

Integration constants (from H and S boundary conditions):
    H/RT  = -a1/T^2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3
            + a6*T^3/4 + a7*T^4/5 + a8/T
    S/R   = -a1/(2T^2) - a2/T + a3*ln(T) + a4*T + a5*T^2/2
            + a6*T^3/3 + a7*T^4/4 + a9

a8 is fixed so that H(T_ref) matches the reference enthalpy at the interval
boundary.  a9 is fixed so that S(T_ref) matches reference entropy at the
interval boundary.

Usage:
    python fit_nasa9.py
"""

import numpy as np
from scipy.optimize import curve_fit
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from mu0_stat import (
    get_mu0_stat,
    _Cv_from_Q, _dH_from_Q, _S_from_Q,
    _shomate_mu0,
    Q_O2, Q_O2plus,
    Hf298, S298, R, T0,
)
from carbon_partition import Q_el

# ---------------------------------------------------------------------------
# Helpers: extract Cp, H-H298, S from mu0_stat sources
# ---------------------------------------------------------------------------

def cp_s_dh_from_partition(Q_func, Hf298_kJ, S298_JmolK, T_array):
    """Return Cp [J/mol/K], S [J/mol/K], dH=H(T)-H(298) [J/mol] arrays."""
    Cp, S, dH = [], [], []
    for T in T_array:
        # Cp = Cv + R  (ideal gas)
        Cv = _Cv_from_Q(Q_func, T)
        Cp.append(Cv + R)
        S.append(_S_from_Q(Q_func, T))
        dH.append(_dH_from_Q(Q_func, T) + 2.5 * R * (T - T0))
    return np.array(Cp), np.array(S), np.array(dH)


def cp_s_dh_from_shomate(sp, T_array):
    """Extract Cp, S, dH from Shomate mu0 via finite differences."""
    from mu0_stat import _SHOMATE
    Hf, intervals = _SHOMATE[sp]
    Cp, S_arr, dH = [], [], []
    for T in T_array:
        t = T / 1000.0
        t0_ = T0 / 1000.0
        A, B, C, D, E, F, G, H_coef = next(
            coefs for T_max, coefs in intervals if T <= T_max
        )
        cp = (A + B*t + C*t**2 + D*t**3 + E/t**2)  # J/mol/K (Shomate Cp)
        s  = A*np.log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G
        def _H(tv):
            return A*tv + B*tv**2/2 + C*tv**3/3 + D*tv**4/4 - E/tv + F - H_coef
        dh = (_H(t) - _H(t0_)) * 1e3  # kJ/mol -> J/mol
        Cp.append(cp)
        S_arr.append(s)
        dH.append(dh)
    return np.array(Cp), np.array(S_arr), np.array(dH)


# ---------------------------------------------------------------------------
# NASA9 coefficient fitter
# ---------------------------------------------------------------------------

def fit_nasa9_interval(T_arr, Cp_arr, H_ref, S_ref, T_ref):
    """
    Fit NASA9 coefficients [a1..a9] for a single temperature interval.

    Cp_arr  : [J/mol/K]  at T_arr points
    H_ref   : H(T_ref)   [J/mol]   — fixes integration constant a8
    S_ref   : S(T_ref)   [J/mol/K] — fixes integration constant a9
    T_ref   : reference temperature for H and S anchoring (interval lower bound)
    """
    # Fit a1..a7 to Cp/R
    CpR = Cp_arr / R

    def nasa9_CpR(T, a1, a2, a3, a4, a5, a6, a7):
        return a1/T**2 + a2/T + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4

    p0 = [0, 0, Cp_arr.mean()/R, 0, 0, 0, 0]
    try:
        popt, _ = curve_fit(nasa9_CpR, T_arr, CpR, p0=p0, maxfev=50000)
    except RuntimeError:
        popt = p0
    a1, a2, a3, a4, a5, a6, a7 = popt

    # Fix a8 from H(T_ref) = H_ref
    # H/RT = -a1/T^2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3
    #        + a6*T^3/4 + a7*T^4/5 + a8/T
    # => a8 = H_ref/R - T_ref*(-a1/T_ref^2 + a2*ln(T_ref)/T_ref + a3
    #                           + a4*T_ref/2 + a5*T_ref^2/3
    #                           + a6*T_ref^3/4 + a7*T_ref^4/5)
    Tr = T_ref
    H_poly = Tr * (-a1/Tr**2 + a2*np.log(Tr)/Tr + a3
                   + a4*Tr/2 + a5*Tr**2/3 + a6*Tr**3/4 + a7*Tr**4/5)
    a8 = H_ref/R - H_poly

    # Fix a9 from S(T_ref) = S_ref
    # S/R = -a1/(2T^2) - a2/T + a3*ln(T) + a4*T + a5*T^2/2
    #       + a6*T^3/3 + a7*T^4/4 + a9
    S_poly = (-a1/(2*Tr**2) - a2/Tr + a3*np.log(Tr)
              + a4*Tr + a5*Tr**2/2 + a6*Tr**3/3 + a7*Tr**4/4)
    a9 = S_ref/R - S_poly

    return [a1, a2, a3, a4, a5, a6, a7, a8, a9]


# ---------------------------------------------------------------------------
# YAML block formatter
# ---------------------------------------------------------------------------

def format_nasa9_yaml(name, composition, intervals_data, note="fit_nasa9.py"):
    """
    intervals_data : list of (T_low, T_high, [a1..a9])
    """
    T_ranges = [intervals_data[0][0]] + [iv[1] for iv in intervals_data]
    lines = [f"- name: {name}"]
    lines.append(f"  composition: {{{composition}}}")
    lines.append(f"  thermo:")
    lines.append(f"    model: NASA9")
    t_range_str = ", ".join(f"{t}" for t in T_ranges)
    lines.append(f"    temperature-ranges: [{t_range_str}]")
    lines.append(f"    data:")
    for _, _, coeffs in intervals_data:
        c = coeffs
        lines.append(f"    - [{c[0]:.8e}, {c[1]:.8e}, {c[2]:.8e}, {c[3]:.8e},")
        lines.append(f"       {c[4]:.8e}, {c[5]:.8e}, {c[6]:.8e}, {c[7]:.8e}, {c[8]:.8e}]")
    lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-species fitting
# ---------------------------------------------------------------------------

def fit_species(name, Q_func, Hf298_kJ, S298_JmolK, source="partition",
                intervals=((298.15, 1000.0), (1000.0, 6000.0), (6000.0, 20000.0)),
                n_pts=80):
    """
    Fit NASA9 for a species over the given temperature intervals.
    Returns yaml block string.
    """
    # Compute full S(T) = S_trans(T) + S_int(T) for partition-function species.
    # S_trans scales as 3/2*R*ln(T/T0) from the T0 anchor.
    # S298_JmolK (from Gordon/NIST) gives the correct total S at 298.15 K.
    def full_S(T):
        if source == "partition":
            S_int_T  = _S_from_Q(Q_func, T)
            S_int_T0 = _S_from_Q(Q_func, T0)
            # Total S at T0 is S298_JmolK; translational part at T0:
            S_trans_T0 = S298_JmolK - S_int_T0
            # Translational S scales as 3/2*R*ln(T/T0)
            S_trans_T  = S_trans_T0 + 1.5 * R * np.log(T / T0)
            return S_int_T + S_trans_T
        else:
            _, S_arr, _ = cp_s_dh_from_shomate(name, [T])
            return S_arr[0]

    def full_H(T):
        if source == "partition":
            dH = _dH_from_Q(Q_func, T) + 2.5 * R * (T - T0)
            return Hf298_kJ * 1e3 + dH
        else:
            _, _, dH_arr = cp_s_dh_from_shomate(name, [T])
            return Hf298_kJ * 1e3 + dH_arr[0]

    results = []
    for i, (T_lo, T_hi) in enumerate(intervals):
        T_arr = np.linspace(T_lo + 1, T_hi - 1, n_pts)

        if source == "partition":
            Cp, _, _ = cp_s_dh_from_partition(Q_func, Hf298_kJ, S298_JmolK, T_arr)
        else:
            Cp, _, _ = cp_s_dh_from_shomate(name, T_arr)

        # Anchor at lower bound of interval using consistent H and S
        T_ref = T_lo
        H_ref = full_H(T_ref)
        S_ref = full_S(T_ref)

        coeffs = fit_nasa9_interval(T_arr, Cp, H_ref, S_ref, T_ref)
        results.append((T_lo, T_hi, coeffs))

    return results


# ---------------------------------------------------------------------------
# Main: fit and print all species
# ---------------------------------------------------------------------------

SPECIES_TO_FIT = [
    # (name,  composition_yaml,  Q_func_or_None,  Hf298_kJ,  S298_JmolK,  source)
    # S298 values from Gordon 1999 (NASA TP-1999-208523) — includes translation + electronic
    ("C2+",  "C: 1, E: -2",  lambda t: Q_el("C III", t),  Hf298["C2+"],  148.85,       "partition"),
    ("O2+",  "O: 2, E: -1",  Q_O2plus,                     Hf298["O2+"],  172.58,       "partition"),
    ("C2",   "C: 2",         None,                          Hf298["C2"],   S298.get("C2", 199.39), "shomate"),
    ("C3",   "C: 3",         None,                          Hf298["C3"],   S298.get("C3", 237.246), "shomate"),
    ("C2O",  "C: 2, O: 1",   None,                          Hf298["C2O"],  None,         "shomate"),
    ("C2-",  "C: 2, E: 1",   None,                          Hf298["C2-"],  None,         "shomate"),
]

if __name__ == "__main__":
    print("# NASA9 coefficients generated by fit_nasa9.py")
    print("# Paste these species blocks into gri30_plasma.yaml under 'species:'\n")

    for name, composition, Q_func, Hf298_kJ, S298_JmolK, source in SPECIES_TO_FIT:
        print(f"# Fitting {name} ({source})...")
        try:
            intervals_data = fit_species(
                name, Q_func, Hf298_kJ, S298_JmolK, source=source,
                intervals=((298.15, 1000.0), (1000.0, 6000.0), (6000.0, 20000.0)),
            )
            yaml_block = format_nasa9_yaml(name, composition, intervals_data)
            print(yaml_block)
            print()
        except Exception as e:
            print(f"# ERROR fitting {name}: {e}\n")
