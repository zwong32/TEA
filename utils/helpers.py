from config         import FIGNAME_EQUILB_COMP
#from scipy.optimize     import minimize_scalar, minimize, LinearConstraint, Bounds
#from test5_validate2 import Xeq, Xeq_ct
from dataclasses    import dataclass
from utils.figure_style import make_single_column_figure, set_figure_defaults
from utils.figure_style   import *
from utils.imports        import *

@dataclass
class Data:
    markers: list
    colors: list
    species: list
    A: list
    n_initial: list
    b: list
    constraint: object
    bounds: object
    n_guess: list

# @dataclass 
# class InitArrays:
#     Xeq: list
#     Xeq_ct: list
#     rhoarray: list
#     EN_balance_values: list
#     rho_ct: list

def set_fig_aesthetics():
    markers = ["o", "+", "x", "s", "^", "v", "D", "p", "h", "*", ">", "<"]
    colors = np.array([
        [45, 85, 150],      # CO2
        [119, 119, 35],     # CO
        [19, 29, 245],      # O
        [234, 52, 37],      # O2
        [87, 191, 71],      # O+
        [53, 89, 140],      # C+
        [0, 0, 0],          # e-
        [52, 120, 31],      # C
        [0, 255, 255],      # C-
        [184, 86, 144],     # O++
        [13, 13, 13],       # C++
    ], dtype=float) / 255.0
    return markers, colors


def _resolve_color(color):
    if isinstance(color, np.ndarray):
        color = color.astype(float)
        if np.max(color) > 1.0:
            color = color / 255.0
        return tuple(color)
    return color

def set_constants(config):

    P = config.PRESSURE  # Pa (1 bar)
    P0 = config.REFPRESSURE        # Pa  (standard state reference)
    R = ct.gas_constant / 1000.0  # J/mol/K  (Cantera uses J/kmol/K)


    # Temperature sweep
    T_values      = config.TEMPERATURE
    species_names = config.SPECIES_NAME  # order used throughout

    return P, P0, T_values, species_names, R


def init_arrays(config, T_values):
    species = [sp for sp, _ in sorted(config.SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

    Xeq      = np.zeros((len(T_values), len(species)))
    Xeq_ct   = np.zeros((len(T_values), len(species)))  # Cantera reference
    rhoarray = np.zeros(len(T_values))
    EN_balance_values = np.full(len(T_values), np.nan)  # collect E/N_crit for each temperature
    rho_ct = np.zeros(len(T_values)) # rho from cantera for comparison

    return Xeq, Xeq_ct, rhoarray, EN_balance_values, rho_ct

def load_data(config, T_values):    

    species = [sp for sp, _ in sorted(config.SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

    A = config.ATOM_BALANCE_MATRIX

    # add a small amount in CO... N to avoid log(0) in the objective funtion, Gibbs free energy
    n_initial = config.INITAL_MOLE_FRACTION_GUESS

    # *** quick comment:
    # Atom-balance RHS must come from the physical composition only —
    # using n_initial (which has 1e-10 placeholders) was found to affect the composition calculation bc at ionizing temperatures,
    # it is possible to have molar values within this order of magntidue. so to avoid violating conservation b so was recomputed 
    # via b = A @ n_initial_physical in the config file 
    b = config.b

    constraint = LinearConstraint(A, b, b)
    bounds = Bounds(np.full(len(species), 1e-12), np.full(len(species), np.inf))

    #Xeq      = np.zeros((len(T_values), len(species)))
    #Xeq_ct   = np.zeros((len(T_values), len(species)))  # Cantera reference
    #rhoarray = np.zeros(len(T_values))
    n_guess  = n_initial.copy()
    #EN_balance_values = []  # collect E/N_crit for each temperature
    #rho_ct = np.zeros(len(T_values)) # rho from cantera for comparison

    return species, A, n_initial, b, constraint, bounds, n_guess



def make_plots(T_values, Xeq, Xeq_ct, rhoarray, rho_ct, EN_balance_values, species, markers, colors, OUTDIR):

    # read data:
    set_figure_defaults()
    fig1, ax1 = make_single_column_figure()
    fig3, ax3 = make_single_column_figure()

    df = pd.read_csv("output/co2_equilibrium_mole_fractions.txt", delim_whitespace=True)

    T = df["T"]

    # for sp, marker, c in zip(df.columns[1:], markers, colors):  # skip T
    #    print(f"Plotting species: {sp} with marker {marker} and color {c}")
    #    ax1.semilogy(T,  df[sp], label=sp, marker=marker, color = c)

    ax1.set_ylabel("Mole fraction")
    ax1.set_ylim(9e-5, 1.4)
    ax1.set_xlim(300, 30000)
    fig1.suptitle(
        f"CO$_2$ equilibrium Composition at 1 bar",
        #x=(0.12 + 0.92) / 2,
        x = 0.5,
        y = 0.95,
        ha="center",
        fontweight="bold",
        fontsize=10
    )
    ax1.set_xlabel("Temperature (K)")
    # Build a clean legend: one entry per species (marker shape), plus solver color key
    handles = []
    # for m, c, sp in zip(markers, colors, species):
    #     handles.append(mlines.Line2D([], [], color=_resolve_color(c), marker=m, linestyle="None", markersize=5, label=f"{sp} (VCS)"))

    # Adjust these temperatures to position labels without overlap.
    # label_T = {
    #     "CO2": 1000,
    #     "CO": 8000,
    #     "O": 9500,
    #     "O2": 4500,
    #     "C": 10000,
    #     "C+": 19500,
    #     "O+": 11500,
    #     "C++": 23500,
    #     "O++": 25500,
    #     "e-": 22000,
    # }

    # display_labels = {
    # "CO2": r"$\mathrm{CO_2}$",
    # "O2": r"$\mathrm{O_2}$",
    # "C+": r"$\mathrm{C^{+}}$",
    # "C++": r"$\mathrm{C^{2+}}$",
    # "O+": r"$\mathrm{O^{+}}$",
    # "O++": r"$\mathrm{O^{2+}}$",
    # "e-": r"$\mathrm{e^{-}}$",
    # }

    label_T = {
            "N2":  500,
            "O2":  500,
            "Ar":  500,
            "CO2": 500,
            "CO":  2000,
            "NO":  1100,
            "NO2": 1000,
            "N2O": 3000,
            "O":   4000,
            "N":   4000,
    }

    display_labels = {
    "N2": r"$\mathrm{N_2}$",
    "O2": r"$\mathrm{O_2}$",
    "Ar": r"$\mathrm{Ar}$",
    "CO2": r"$\mathrm{CO_2}$",
    "CO": r"$\mathrm{CO}$",
    "NO": r"$\mathrm{NO}$",
    "NO2": r"$\mathrm{NO_2}$",
    "N2O": r"$\mathrm{N_2O}$",
    "O": r"$\mathrm{O}$",
    "N": r"$\mathrm{N}$",
    }

    
    for m, c, sp in zip(markers, colors, species):
        y = df[sp]
        color = c

        ax1.semilogy(T, y, color=color)

        if sp in label_T:
            # Select the data point nearest the desired temperature.
            i = np.argmin(np.abs(np.asarray(T) - label_T[sp]))

            if np.isfinite(y[i]) and y[i] > 0:
                ax1.text(
                    # Use the display label if it exists, otherwise use the species name
                    
                    T[i], y[i], display_labels.get(sp, sp),
                    #T[i], y[i], #sp
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=0.5,
                    ),
                    zorder=10,

                    
                )

    # plot density vs temperature for air at 100 bar, using Peng-Robinson EOS
    fig2, ax2 = make_single_column_figure()

    #M = config.SPECIS_WEIGHT 
    #ax2.plot(T_values,P*M/(R*T_values),'--',color='tab:gray',label='Ideal gas (Z = 1)')
    ax2.plot(T_values, rho_ct,   markersize=1.0, marker='o', linestyle='-', color=_resolve_color(colors[0]), label='Real Air (Peng-Robinson, 310-10000 K)')
    ax2.plot(T_values, rhoarray, markersize=1.0, marker='s', linestyle='-', color=_resolve_color(colors[1]), label='Present Work (Peng-Robinson, 310-10000 K)')

    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('Temperature (K)',fontsize=12)
    ax2.set_ylabel('Density (kg/m$^3$)',fontsize=12)
    ax2.set_title('CO$_2$ density at 1 bar')
    ax2.set_xlim(300, 25000)
    ax2.grid(True,which='both',alpha=0.3)
    ax2.legend(fontsize=6)

    fig2.tight_layout()

    ax3.plot(T_values, EN_balance_values, markersize=1.0, marker='o', linestyle='-', color=_resolve_color(colors[2]), label='Present Work (Peng-Robinson, 310-10000 K)')
    ax3.set_xlabel('Temperature (K)',fontsize=12)
    ax3.set_ylabel('E/N$_{crit}$ (Td)',fontsize=12)
    ax3.set_title('Townsend breakdown E/N$_{crit}$ vs Temperature')
    ax3.set_xlim(300, 6000)
    ax3.grid(True,which='both',alpha=0.3)   
    fig3.tight_layout()
    
    
    fig1.savefig(OUTDIR / config.FIGNAME_EQUILB_COMP, dpi=300, bbox_inches="tight")
    print("Saved " + config.FIGNAME_EQUILB_COMP)
    
    fig2.savefig(OUTDIR / config.FIGNAME_DENSITY, dpi=300, bbox_inches="tight")
    print("Saved " + config.FIGNAME_DENSITY)

    fig3.savefig(OUTDIR / config.FIGNAME_EN_BALANCE, dpi=300, bbox_inches="tight")
    print("Saved " + config.FIGNAME_EN_BALANCE)
    

# colors from Yang et al. Thermodynamic Properties and Transport Coefficients of CO2–Cu Thermal Plasmas
#       CO2: 45, 85, 150
#       CO: 119, 119, 35
#       O:   19, 29, 245
#       O2:  234, 52, 37
#       O+: 87, 191, 71
#       C+: 53, 89, 140 
#       e-: 0,0,0
#       C: 52, 120, 31
#       C-: "cyan"
#       O++ 184, 86, 144
#       C++:13,13,13
