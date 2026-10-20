import numpy as np

PRESSURE = 10e6  # match jia wei pressure; crossectional data should be independent of pressure
REFPRESSURE = 1.0e5        # Pa  (standard state reference)

TEMPERATURE = np.arange(310, 5000, 10)  # stay within critical temperature of CO2 (304.13 K) to avoid phase change; units: K

# CANTERA INPUTS or COOLPROP INPTUS
SPECIES_NAME = "CO2:1" # CO2:1 for cantera; CO2 for CoolProp
SPECIS_WEIGHT = 44.01e-3 # kg/mol, molar mass of CO2

# COOL PROP INPUTS (if real gas)
# USER INPUTS FOR SYSTEM OF EQUATIONS FOR SOLVING FOR THE MOLE FRACTIONS OF EACH SPECIES
# SOLVE A*n_initial_guess = b

ATOM_BALANCE_MATRIX = A = np.array([
    [1]
    ], dtype=float)


INITAL_MOLE_FRACTION_GUESS = np.array([1.0,     #CO2
                                     ])

#CANTERA_INITIAL_MOLE_FRACTION_GUESS = "CO2:1"

# elemental conservation vector (RHS of the atom-balance equations)
# ideally should be the same as the intial guess
n_initial_physical = np.array([1.0,])
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
    'CO2':    0
    
}

# Ordered list of species names (index matches Xeq columns), derived from SPECIES_DENSITY_MAP
SPECIES = [sp for sp, _ in sorted(SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

MOLAR_MASS_SPECIES = np.array([
    44.0095e-3,  # 14 CO2
    
]) # units: kg/mol

### SOLVERS
GET_EEDF = False # True = yes; False = no
GET_EN_CRIT = False # True = yes; False = no

### POST PROCESSING
# PLOT RESULTS?
PLOT_RESULTS = True # true = yes; false = no

# SAVE RESULTS?
SAVE_RESULTS = True # true = yes; false = no