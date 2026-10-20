import numpy as np

PRESSURE = 10e6  # match jia wei pressure; crossectional data should be independent of pressure
REFPRESSURE = 1.0e5        # Pa  (standard state reference)

TEMPERATURE = np.arange(310, 320, 10)  # stay within critical temperature of CO2 (304.13 K) to avoid phase change; units: K

# CANTERA INPUTS
SPECIES_NAME = "CO2:1"
SPECIS_WEIGHT = "0.028965"

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
    [1, 2,  1, 1, 2, 0,  0, 0, 0,  0, 0, 0, 1, 1, 2, 0],
    [0, 0,  0, 0, 0, 3,  2, 1, 2,  1, 1, 1, 1, 2, 1, 0],
    [0, 0, -1, 1, 1, 0, -1, 0, 0, -1, 1, 2, 0 ,0, 0, -1]
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

INITAL_MOLE_FRACTION_GUESS = np.array([1.0,   # 0  O
                                       2.0,   # 1  O2
                                       1e-10, # 2  O-
                                       1e-10, # 3  O+
                                       1e-10, # 4  O2+
                                       1e-10, # 5  C3
                                       1e-10, # 6  C2-
                                       1e-10, # 7  C
                                       1e-10, # 8  C2
                                       1e-10, # 9  C-
                                       1e-10, # 10 C+
                                       1e-10, # 11 C2+
                                       1e-10, # 12 CO
                                       1e-10, # 13 C2O
                                       1e-10, # 14 CO2
                                       1e-10, # 15 e-
                                     ])

#CANTERA_INITIAL_MOLE_FRACTION_GUESS = "CO2:1"

# elemental conservation vector (RHS of the atom-balance equations)
n_initial_physical = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
#n_initial_physical = np.array([1.0, 2.0, 0.0])
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
    'O':    0,
    'O2':   1,
    'O-':   2,
    'O+':   3,
    'O2+':  4,
    'C3':   5,
    'C2-':  6,
    'C':    7,
    'C2':   8,
    'C-':   9,
    'C+':  10,
    'C2+': 11,
    'CO':  12,
    'C2O': 13,
    'CO2': 14,
    'e-':  15,
}

# Ordered list of species names (index matches Xeq columns), derived from SPECIES_DENSITY_MAP
SPECIES = [sp for sp, _ in sorted(SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

MOLAR_MASS_SPECIES = np.array([
    15.9994e-3,  # 0  O
    31.9988e-3,  # 1  O2
    15.9994e-3,  # 2  O-   (electron mass negligible vs O)
    15.9994e-3,  # 3  O+
    31.9988e-3,  # 4  O2+
    36.0321e-3,  # 5  C3
    24.0214e-3,  # 6  C2-
    12.0107e-3,  # 7  C
    24.0214e-3,  # 8  C2
    12.0107e-3,  # 9  C-
    12.0107e-3,  # 10 C+
    24.0214e-3,  # 11 C2+
    28.0101e-3,  # 12 CO
    40.0214e-3,  # 13 C2O  (C2 + O = 24.02 + 15.99)
    44.0095e-3,  # 14 CO2
     0.5486e-6,  # 15 e-   (electron: 9.109e-31 kg * 6.022e23 ≈ 5.486e-7 kg/mol)
]) # units: kg/mol

### SOLVERS
GET_EEDF = False # True = yes; False = no
GET_EN_CRIT = False # True = yes; False = no

### POST PROCESSING
# PLOT RESULTS?
PLOT_RESULTS = True # true = yes; false = no

# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no