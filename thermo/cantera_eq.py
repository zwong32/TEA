from utils.figure_style import *
from utils.imports import *
from utils.co2_density_plot import *
from thermo.mu0_stat import *
### python conventois
# All caps == constant
# leading underscore = internal module variable

# load inputs from config file
# P = config.PRESSURE  # Pa (1 bar)
P0 = config.REFPRESSURE        # Pa  (standard state reference)
R = ct.gas_constant / 1000.0  # J/mol/K  (Cantera uses J/kmol/K)

# need to improve this one
_HOME     = "/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/"
_GRI30PLASMA_YAML = Path(_HOME) / "cantera" / "data" / "gri30_plasma.yaml"  # charged species: O+, e-
_PLASMA_YAML = Path(_HOME) / "cantera" / "data" / "plasma.yaml"  # charged species: O+, e-
_AIR_YAML   =  Path(_HOME) / "cantera" / "data" / "gri30.yaml"

# --- Cantera solution objects ---
#gas = ct.Solution("/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/cantera/data/gri30_plasma.yaml")  # create 1 thermo state w/ neutral species: CO2, CO, O
#gas_plasma = ct.Solution(_yaml_path)  # charged species:gri30v1.yaml O+, e-

# cantera objects
_gas = ct.Solution(config.CANTERA_YAML)  # neutral species: CO2, CO, O
_gas_gri30plasma = ct.Solution(_GRI30PLASMA_YAML)  # neutral species: CO2, CO, O
_gas_plasma = ct.Solution(_PLASMA_YAML)  # charged species: O+, e-
_gas_ct = ct.Solution(_AIR_YAML)  # neutral species: CO2, CO, O # same as _gas_gri30plasma

def get_mole_fractions(gas, species):
    """Return mole fractions in the requested species order."""
    mole_fractions = dict(zip(gas.species_names, gas.X))
    return np.array([mole_fractions.get(sp, 0.0) for sp in species])

def cantera_equilibrium(T, P, species):
    """
    Use Cantera's built-in VCS equilibrium solver.
    Returns mole-fraction array in SPECIES order.

    gri30.yaml only contains neutral species (CO2, CO, O, O2, C, C2, C3, ...).
    Ions and clusters not in gri30 are returned as 0.

    T is temperature; an array from config file but fed in as a scalar
    P is pressure, a scalar from config file
    species is an array of species names from SPECIES_DENSITY_MAP
    """

    # _gas_gri30plasma was _gas_ct which was based air_yaml then gri30_plasma.yaml; but saw a duplication of thermo states 
    init_comp = config.VCS_INITIAL_GUESS  # initial guess for VCS solver; must be charge-neutral

    # _gas_ct.TPX = T, P, init_comp  # seed charge-neutral ion pair so VCS can explore ion DOF
    # _gas_ct.equilibrate("TP", solver="vcs")

    # _gas_gri30plasma.TPX = T, P, config.VCS_INITIAL_GUESS  # seed charge-neutral ion pair so VCS can explore ion DOF
    # _gas_gri30plasma.equilibrate("TP", solver="vcs")

    # x = np.zeros(len(species))
    # for i, sp in enumerate(species):
    #     try:
    #         idx = _gas_gri30plasma.species_index(sp)
    #         x[i] = _gas_gri30plasma.X[idx]
    #     except ValueError:
    #         x[i] = 0.0   # species not in gri30_plasma.yaml - higher order ion_yaml_path = os.path.join(_HOME, "cantera", "data", "plasma.yaml")  # charged species: O+, e-

    #_YAML_PATH = _HOME + "cantera/data/gri30_plasma.yaml"  # charged species: O+, e-s

    _gas.TPX = T, P, init_comp  # seed charge-neutral ion pair so VCS can explore ion DOF
    _gas.equilibrate("TP", solver="vcs")
    x = get_mole_fractions(_gas, species)
    
    rho = _gas.density 
    Z, phi = rho_pr(T) 
    rho = rho/Z  # correct for real-gas behavior using Peng-Robinson EOS; matches w/ nasa-nist DB within 3% :(

    return x, rho

def coolprop_density(T, P): # for debugging bc cantera can only handle ideal gas behavior
    """
    Use CoolProp to get the density of a real gas.
    Returns density in kg/m^3.
    """
    rho = cp.PropsSI('D', 'T', T, 'P', P, config.SPECIES_NAME)  
    return rho

def get_mu0_air(T, P, species):
    """
    Return standard-state chemical potentials mu^0 [J/mol] at temperature T.
    Neutral species from gri30.yaml (NASA-7); O+ and e- from plasma.yaml.
    Pressure-independent: mu(T,P,x) = mu0(T) + R*T*ln(x*P/P0).
    """
    mu0 = []

    for sp in species:
            _gas.TPX = T, P0, f"{sp}:1"
            mu0.append(_gas.chemical_potentials[_gas.species_index(sp)] / 1000.0) # cantera default units are J/kmol. convert from J/kmol to J/mol by dividing by 1000.0

    return np.array(mu0)  


# for a pure species, x_i = 1, P = P0 so mu_i = mu_0
def get_mu0(T, P, species): # relies on many .yaml and partition functions
    """
    Return standard-state chemical potentials mu^0 [J/mol] at temperature T.

    Routing (per species):
      CO2, CO, O, O2      → gri30.yaml  (Cantera NASA-7)
      O+, e-              → plasma.yaml (Cantera NASA-7)
      O2+, C, C+, C2+     → partition functions  (mu0_stat.py)
      C2, C3, C2O, O-,
      C-, C2-             → NIST Shomate polynomials (mu0_stat.py)

    Pressure-independent: mu(T,P,x) = mu0(T) + R*T*ln(x*P/P0).
    Units: J/mol  (Cantera returns J/kmol — divided by 1000 here;
                   mu0_stat returns J/mol directly).
    """
    from thermo.mu0_stat import get_mu0_stat  # local import avoids circular dependency at module level

    # Species handled by Cantera
    _GRI_SPECIES    = {"CO2", "CO", "O", "O2"}  # gri30.yaml  (NASA-7 polynomials)
    _PLASMA_SPECIES = {"O+", "e-"}              # plasma.yaml (NASA-7 polynomials)

    mu0 = []

    for sp in species:
        if sp in _GRI_SPECIES:
            _gas_gri30plasma.TPX = T, P0, f"{sp}:1" # related to gri30_plasma.yaml
            Z, phi = rho_pr(T) 
            # Cantera: chemical_potentials in J/kmol → divide by 1000 for J/mol

            # mu0 = mu0 + RTnp.log9phi*x*P/Po)
            mu = _gas_gri30plasma.chemical_potentials[_gas_gri30plasma.species_index(sp)] / 1000.0 + R*T*np.log(phi*P/P0)
            mu0.append(mu)  
        # mu0.append(_gas_gri30plasma.chemical_potentials[_gas_gri30plasma.species_index(sp)] / 1000.0)

        elif sp in _PLASMA_SPECIES:
            # plasma.yaml needs a non-zero composition to avoid singular matrix
            _gas_plasma.TPX = T, P0, f"{sp}:1" # related to plasma.yaml
            # _gas_plasma.TPX = T, P0, f"{sp}:1" #"O+:0.5, e-:0.5"
            mu0.append(_gas_plasma.chemical_potentials[_gas_plasma.species_index(sp)] / 1000.0)

        else:
            # Partition-function / Shomate route — mu0_stat returns J/mol directly
            val = get_mu0_stat(sp, T)
            if val is None:
                raise KeyError(
                    f"get_mu0: species '{sp}' not found in any source. "
                    "Add it to mu0_stat.py or check config.SPECIES_DENSITY_MAP."
                )

        mu = val + R*T*np.log(phi*P/P0)
        mu0.append(mu)  # already in J/mol
                
   
    return np.array(mu0)

def q_saha(nO, ntot, mu0, T):
    """
    Solve analytically for q = n[O+] = n[e-] from ionization equilibrium:
        mu_O+ + mu_e- = mu_O
    => 2*R*T*ln(q/ntot * P/P0) = (mu0_O - mu0_O+ - mu0_e-) + R*T*ln(nO/ntot * P/P0)
    This is exact in the dilute-ion limit (q << n_neutral).
    """
    dmu = mu0[2] - mu0[3] - mu0[4]          # mu0_O - mu0_O+ - mu0_e-  [J/mol]
    xO  = np.clip(nO / ntot, 1e-300, None)
    lnq = dmu / (2.0 * R * T) + 0.5 * np.log(xO * P / P0) + 0.5 * np.log(ntot / (P / P0))
    return np.exp(np.clip(lnq, -700, 0))    # q always in (0, 1)


