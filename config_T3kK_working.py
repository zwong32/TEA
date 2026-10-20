import numpy as np

PRESSURE = 10.00e6  # match jia wei pressure; crossectional data should be independent of pressure
REFPRESSURE = 1.0e5        # Pa  (standard state reference)

TEMPERATURE = np.arange(310, 000, 10)  # stay within critical temperature of CO2 (304.13 K) to avoid phase change; units: K

# CANTERA INPUTS
SPECIES_NAME = "CO2:1"
SPECIS_WEIGHT = 44.0095e-3

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
    [1.0, 1.0,  0.0, 0.0 ],
    [2.0, 1.0,  1.0, 2.0 ],
    ],dtype=float)

# add a small amount in CO... N to avoid log(0) in the objective funtion, Gibbs free energy
# INITIAL_MOLE_FRACTION_GUESS = np.array([
#     0.7809,  # N2
#     0.2094,  # O2
#     0.0093,  # Ar
#     0.0004,  # CO2
#     1.0e-10,     # CO
#     1.0e-10,     # NO
#     1.0e-10,     # NO2
#     1.0e-10,     # N2O
#     1.0e-10,     # O
#     1.0e-10      # N
# ])

INITAL_MOLE_FRACTION_GUESS = np.array([0.7, 0.3, 0.1, 0.1
                                     ])

#CANTERA_INITIAL_MOLE_FRACTION_GUESS = "CO2:1"

# elemental conservation vector (RHS of the atom-balance equations)
n_initial_physical = np.array([0.7, 0.3, 0.1, 0.1
                                     ])
b =  A @ n_initial_physical

# LXCAT NAMES FOR ELECTRON IMPACT CROSS SECTIONS

# BOLOS CROSS-SECTION DATA FILES
LXCAT_DATA_DIR = "Download-lxhat"

CROSS_SECTION_FILES = [
    "phelps.txt",    # Main Morgan/LXCat database (N2, O2, Ar, CO2, CO, NO, O, N)
    "itikawa.txt"  ,    # N2O — Hayashi database
    "hayashi.txt",         # NO2 — Sakai database
]

# BOLTZMANN SOLVER DENSITY SETTINGS
MIN_DENSITY = 1e-10  # floor to prevent division-by-zero in BOLOS


SPECIES_DENSITY_MAP = {
    'CO2':  0,
    'CO':   1,
    'O':    2,
    'O2':   3,

}

# Ordered list of species names (index matches Xeq columns), derived from SPECIES_DENSITY_MAP
SPECIES = [sp for sp, _ in sorted(SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

MOLAR_MASS_SPECIES = np.array([ # kg/mol
    44.0095e-3,  # 14 CO2 
    28.0101e-3,  # 12 CO
    15.9994e-3,  # 0  O
    31.9988e-3,  # 1  O2
]) # units: kg/mol

### SOLVERS
GET_EEDF = False # True = yes; False = no
GET_EN_CRIT = False # True = yes; False = no

### POST PROCESSING
# PLOT RESULTS?
PLOT_RESULTS = True # true = yes; false = no

# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no