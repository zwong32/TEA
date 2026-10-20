import numpy as np

PRESSURE = 0.1e6  # match jia wei pressure; crossectional data should be independent of pressure
REFPRESSURE = 1.0e5        # Pa  (standard state reference)

TEMPERATURE = np.arange(310, 10000, 1000)  # stay within critical temperature of CO2 (304.13 K) to avoid phase change; units: K

# CANTERA INPUTS
SPECIES_NAME = "co2"
SPECIS_WEIGHT = 0.04401 # kg/mol

_HOME     = "/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/"
_GRI30PLASMA_YAML = _HOME + "/cantera/" + "data" + "/gri30_plasma.yaml"  # charged species: O+, e-
# _PLASMA_YAML = Path(_HOME) / "cantera" / "data" / "plasma.yaml"  # charged species: O+, e-
# _AIR_YAML   =  _HOME + "/cantera/" + "data" + "/gri30.yaml"

CANTERA_YAML = _GRI30PLASMA_YAML  # neutral species: CO2, CO, O


#VCS_INITIAL_GUESS = {"CO2": 1.0, "O+": 1e-10, "C+": 1e-10, "e-": 2e-10, "O++": 1e-10, "C++": 1e-10}   # "equilibrium" or "initial" (initial guess from config.py)
VCS_INITIAL_GUESS = {
    
    "CO2": 1.0,
    "CO" : 1e-10,
    "O"  : 1e-10,
    "O2" : 1e-10,
    "O+" : 1e-10,
    "C+" : 1e-10,
    "e-" : 2e-10,
    "C"  : 1e-10,
    "C-" : 1e-10,
    "O-" : 1e-10,
    "O++": 1e-10,
    "C++": 1e-10,

}
# USER INPUTS FOR SYSTEM OF EQUATIONS FOR SOLVING FOR THE MOLE FRACTIONS OF EACH SPECIES
# SOLVE A*n_initial_guess = b
# ATOM_BALANCE_MATRIX = A = np.array([
# #    0      1      2       3       4       5       6       7       8       9
# #    N2     O2     Ar      CO2     CO      NO      NO2     N2O     O       N
#     [2,     0,     0,      0,      0,      1,      1,      2,      0,      1],  # N
#     [0,     2,     0,      2,      1,      1,      2,      1,      1,      0],  # O
#     [0,     0,     0,      1,      1,      0,      0,      0,      0,      0],  # C
#     [0,     0,     1,      0,      0,      0,      0,      0,      0,      0],  # Ar
# ], dtype=float)

ATOM_BALANCE_MATRIX = A = np.array([
#   1     2     3      4    5     6     7     8     9     10.   11.   12
#   CO2   CO    O     O2    O+    C+    e-    C     C-    O-    O++   C++
    [1.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  1.0,  0.0,  0.0,  0.0],  # C atoms
    [2.0,  1.0,  1.0,  2.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  2.0,  1.0],  # O atoms
    [0.0,  0.0,  0.0,  0.0,  1.0,  1.0, -1.0,  0.0, -1.0, -1.0,  2.0,  2.0],  # charge: O+ + C+ - e- - C- - O- = 0
    ], dtype=float)

# Initial guess: start close to pure CO2, tiny amounts of products
# Must satisfy A @ n_initial_physical = b (pure CO2 = 1C, 2O, 0 charge)
# Charge row: O+ + C+ - e- - C- - O- + O2+ + 2*C2+ = 0
# => e- = O+ + C+ - C- - O- + O2+ + 2*C2+
#       = 1e-10 + 1e-10 - 1e-10 - 1e-10 + 1e-10 + 2*1e-10 = 3e-10
INITAL_MOLE_FRACTION_GUESS = np.array([
    1.0,    # CO2
    1e-10,  # CO
    1e-10,  # O
    1e-10,  # O2
    1e-10,  # O+
    1e-10,  # C+
    3e-10,  # e-  = O+ + C+ - C- - O- + O2+ + 2*C2+ (charge neutrality)
    1e-10,  # C
    1e-10,  # C-
    1e-10,  # O-
    1e-10,  # O++
    1e-10,  # C++
])

# Atom balance RHS: pure CO2 → 1 C atom, 2 O atoms, 0 net charge
n_initial_physical = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
b = A @ n_initial_physical   # = [1.0, 2.0, 0.0]

# LXCAT NAMES FOR ELECTRON IMPACT CROSS SECTIONS

# BOLOS CROSS-SECTION DATA FILES
LXCAT_DATA_DIR = "lxcat"

CROSS_SECTION_FILES = [
    "phelps.txt",       # Main Morgan/LXCat database (N2, O2, Ar, CO2, CO, NO, O, N)
    "itikawa.txt"  ,    # co2 — itikawa database
    "hayashi.txt",      # NO2 — hayashi database
    #"N2O_hayashi.txt",    # Main Morgan/LXCat database (N2O)
    #"sakai_NO2.txt"  ,    # N2O — Hayashi database
    #"cross_section.txt",         # NO2 — Sakai database
]

# CROSS_SECTION_FILES = [
#     "N2O_hayashi.txt",    # Main Morgan/LXCat database (N2, O2, Ar, CO2, CO, NO, O, N)
#     "sakai_NO2.txt"  ,    # N2O — Hayashi database
#     "cross_section.txt",   # NO2 — Sakai database
# ]


# BOLTZMANN SOLVER DENSITY SETTINGS
MIN_DENSITY = 1e-10  # floor to prevent division-by-zero in BOLOS

# Maps species name -> column index in Xeq
# (NO2 and N2O come from non-Morgan/LXCat databases, so listed separately)
# SPECIES_DENSITY_MAP = {
#     'N2':  0,
#     'O2':  1,
#     'Ar':  2,
#     'CO2': 3,
#     'CO':  4,
#     'NO':  5,
#     'O':   8,
#     'N':   9,
#     'NO2': 6,
#     'N2O': 7,
# }

SPECIES_DENSITY_MAP = {
    'CO2':  0,
    'CO':   1,
    'O':    2,
    'O2':   3,
    'O+':   4,
    'C+':   5,
    'e-':   6,
    'C':    7,
    'C-':   8,
    'O-':   9,
    'O++':  10,
    'C++':  11,
}

# Ordered list of species names (index matches Xeq columns), derived from SPECIES_DENSITY_MAP
SPECIES = [sp for sp, _ in sorted(SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

MOLAR_MASS_SPECIES = np.array([
    44.0095e-3,       # CO2
    28.0101e-3,       # CO
    15.9994e-3,       # O
    31.9988e-3,       # O2
    15.9989e-3,       # O+  (O minus electron mass, negligible difference)
    12.0102e-3,       # C+  (C minus electron mass, negligible difference)
    9.10938e-31,      # e-
    12.0107e-3,       # C
    12.0112e-3,       # C-  (C plus electron mass, negligible difference)
    15.9999e-3,       # O-  (O plus electron mass, negligible difference)
    15.9999e-3,       # O++ (O plus two electron masses, negligible difference)
    24.0204e-3,       # C2+
]) # units: kg/mol


### SOLVERS
INHOUSE_MINIMIZER = True # True = yes; False = no
GET_EEDF = True # True = yes; False = no
GET_EN_CRIT = True # True = yes; False = no
GET_DENSITY = True # True = yes; False = no

### POST PROCESSING
# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no
FIGNAME_EQUILB_COMP = "equilibrium_comparison.png" # 
FIGNAME_DENSITY = "co2_equilibrium_density_1bar_withZ=1.png" #
FIGNAME_EN_BALANCE = "co2_equilibrium_EN_balance.png" #
# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no
TXTNAME_DENSITY = "co2_equilibrium_density_1bar.txt" #
TXTNAME_COMP = "co2_equilibrium_mole_fractions.txt" #