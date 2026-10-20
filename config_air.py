import numpy as np

PRESSURE = 0.1 * 1.0e6  # Pa (matching pressure in the of paper, Fig. 4)
REFPRESSURE = 1.0e5        # Pa  (standard state reference)

TEMPERATURE = np.arange(310, 5000, 100)  # K  (Cantera requires T > 0; start at 200 K)

# CANTERA INPUTS
SPECIES_NAME = "air"


# input whether we want co2 vs air .yaml files
_HOME     = "/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/"
# _GRI30PLASMA_YAML = Path(_HOME) / "cantera" / "data" / "gri30_plasma.yaml"  # charged species: O+, e-
# _PLASMA_YAML = Path(_HOME) / "cantera" / "data" / "plasma.yaml"  # charged species: O+, e-
_AIR_YAML   =  _HOME + "/cantera/" + "data" + "/gri30.yaml"

CANTERA_YAML = _AIR_YAML  # neutral species: CO2, CO, O

VCS_INITIAL_GUESS = "N2:0.7809, O2:0.2094, Ar:0.0093, CO2:0.0004, CO:1e-10, NO:1e-10, NO2:1e-10, N2O:1e-10, O:1e-10, N:1e-10"      


# USER INPUTS FOR SYSTEM OF EQUATIONS FOR SOLVING FOR THE MOLE FRACTIONS OF EACH SPECIES
# SOLVE A*n_initial_guess = b
ATOM_BALANCE_MATRIX = A = np.array([
#    0      1      2       3       4       5       6       7       8       9
#    N2     O2     Ar      CO2     CO      NO      NO2     N2O     O       N
    [2,     0,     0,      0,      0,      1,      1,      2,      0,      1],  # N
    [0,     2,     0,      2,      1,      1,      2,      1,      1,      0],  # O
    [0,     0,     0,      1,      1,      0,      0,      0,      0,      0],  # C
    [0,     0,     1,      0,      0,      0,      0,      0,      0,      0],  # Ar
], dtype=float)


# add a small amount in CO... N to avoid log(0) in the objective funtion, Gibbs free energy
INITAL_MOLE_FRACTION_GUESS = np.array([
    0.7809,  # N2
    0.2094,  # O2
    0.0093,  # Ar
    0.0004,  # CO2
    1.0e-10,     # CO
    1.0e-10,     # NO
    1.0e-10,     # NO2
    1.0e-10,     # N2O
    1.0e-10,     # O
    1.0e-10      # N
])

# elemental conservation vector (RHS of the atom-balance equations)
n_initial_physical = np.array([0.7809, 0.2094, 0.0093, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
b =  A @ n_initial_physical

# LXCAT NAMES FOR ELECTRON IMPACT CROSS SECTIONS

# BOLOS CROSS-SECTION DATA FILES
LXCAT_DATA_DIR = "Download-lxhat"

CROSS_SECTION_FILES = [
    "cross_section.txt",    # Main Morgan/LXCat database (N2, O2, Ar, CO2, CO, NO, O, N)
    "N2O_hayashi.txt"  ,    # N2O — Hayashi database
    "sakai_NO2.txt",         # NO2 — Sakai database
]

# BOLTZMANN SOLVER DENSITY SETTINGS
MIN_DENSITY = 1e-10  # floor to prevent division-by-zero in BOLOS

# Maps species name -> column index in Xeq
# (NO2 and N2O come from non-Morgan/LXCat databases, so listed separately)
SPECIES_DENSITY_MAP = {
    'N2':  0,
    'O2':  1,
    'Ar':  2,
    'CO2': 3,
    'CO':  4,
    'NO':  5,
    'O':   8,
    'N':   9,
    'NO2': 6,
    'N2O': 7,
}

# Ordered list of species names (index matches Xeq columns), derived from SPECIES_DENSITY_MAP
SPECIES = [sp for sp, _ in sorted(SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

MOLAR_MASS_SPECIES = np.array([
    0.0280134, 
    0.0319988, 
    0.039948, 
    0.04401, 
    0.02801, 
    0.0300061, 
    0.0460055, 
    0.0440128, 
    0.0159994, 
    0.0140067
]) # units: kg/mol

### SOLVERS
INHOUSE_MINIMIZER = True # True = yes; False = no
GET_EEDF = False # True = yes; False = no
GET_EN_CRIT = False # True = yes; False = no
GET_DENSITY = False # True = yes; False = no

### POST PROCESSING
# PLOT RESULTS?
PLOT_RESULTS = True # true = yes; false = no
FIGNAME_EQUILB_COMP = "air_equilibrium_comparison.png" # 
FIGNAME_DENSITY = "air_equilibrium_density_1bar_withZ=1.png" #
FIGNAME_EN_BALANCE = "air_equilibrium_EN_balance.png" #
# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no
TXTNAME_DENSITY = "air_equilibrium_density_1bar.txt" #
TXTNAME_COMP = "air_equilibrium_mole_fractions.txt" #