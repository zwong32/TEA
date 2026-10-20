# this is the first version to incorporate a config file
import warnings
import sys
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import cantera           as ct # good for ideal gas behavior
from CoolProp import CoolProp  as cp  # good for real gas behavior

import os
import config
from co2_density_plot import lnphi, rho_pr
from bolos import grid, solver, grid, target, parser
import scipy.constants as co
from scipy.optimize import minimize_scalar, minimize, LinearConstraint, Bounds
from pathlib import Path
from utils.figure_style import *

## inputs 
#_HERE     = os.path.dirname(os.path.abspath(__file__))
_HERE     = "/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/"
#_AIR_YAML = os.path.join(_HERE, "cantera", "data", "gri30_plasma.yaml")
_AIR_YAML = "/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/cantera/data/gri30_plasma.yaml"


# validate for air at 0.1MPa from 0 to 5000K from this paper: https://doi.org/10.1063/1.5145146
P = config.PRESSURE  # Pa (1 bar)
P0 = config.REFPRESSURE        # Pa  (standard state reference)

# Temperature sweep
T_values = config.TEMPERATURE
species_names = config.SPECIES_NAME  # order used throughout

# --- Cantera solution objects ---
gas = ct.Solution("/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/cantera/data/gri30_plasma.yaml")  # create 1 thermo state w/ neutral species: CO2, CO, O
_yaml_path = os.path.join(_HERE, "cantera", "data", "plasma.yaml")  # charged species: O+, e-
gas_plasma = ct.Solution(_yaml_path)  # related to plasma.yaml charged species:gri30v1.yaml O+, e-

R = ct.gas_constant / 1000.0  # J/mol/K  (Cantera uses J/kmol/K)

# ---------------------------------------------------------------------------
# Cantera reference: equilibrate('TP')
# ---------------------------------------------------------------------------
_gas_ct = ct.Solution(_AIR_YAML) # related to gri30_plasma.yaml  # neutral species: CO2, CO, O


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

    # doubly defined; _gas_ct == gas
    # _gas_ct.TPX = T,P, {"CO2": 1.0}
    # _gas_ct.equilibrate("TP", solver="vcs")

    _gas_ct.TPX = T, P, config.VCS_INITIAL_GUESS  # seed charge-neutral ion pair so VCS can explore ion DOF
    _gas_ct.equilibrate("TP", solver="vcs")

    x = np.zeros(len(species))
    for i, sp in enumerate(species):
        try:
            idx = _gas_ct.species_index(sp)
            x[i] = _gas_ct.X[idx]
        except ValueError:
            x[i] = 0.0   # species not in gri30_plasma.yaml - higher order ions

    rho = _gas_ct.density 
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

# for a pure species, x_i = 1, P = P0 so mu_i = mu_0
def get_mu0(T, P, species):
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
            gas.TPX = T, P0, f"{sp}:1" #related to gri30_plasma.yaml
            Z, phi = rho_pr(T) 
            # Cantera: chemical_potentials in J/kmol → divide by 1000 for J/mol

            # mu0 = mu0 + RTnp.log9phi*x*P/Po)
            mu = gas.chemical_potentials[gas.species_index(sp)] / 1000.0 + R*T*np.log(phi*P/P0)
            mu0.append(mu)  
           # mu0.append(gas.chemical_potentials[gas.species_index(sp)] / 1000.0)

        elif sp in _PLASMA_SPECIES:
            # plasma.yaml needs a non-zero composition to avoid singular matrix
            gas_plasma.TPX = T, P0, f"{sp}:1" # related plasma.yaml
            #gas_plasma.TPX = T, P0, f"{sp}:1" #"O+:0.5, e-:0.5"
            mu0.append(gas_plasma.chemical_potentials[gas_plasma.species_index(sp)] / 1000.0)

        else:
            # Partition-function / Shomate route — mu0_stat returns J/mol directly
            val = get_mu0_stat(sp, T)
            if val is None:
                raise KeyError(
                    f"get_mu0: species '{sp}' not found in any source. "
                    "Add it to mu0_stat.py or check config.SPECIES_DENSITY_MAP."
                )

            # zw: uncomment this if doesnt work 
            #mu0.append(val)  # already in J/mol

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
    return np.exp(np.clip(lnq, -700, 0))    # q always in (0, 1]


# gibbs for higher gradients?
def gibbs_grad(n, mu0, T, P, P0, phi):
      """
        Gibbs free energy of an ideal-gas mixture.
    
        n   : species mole numbers
        mu0 : standard chemical potentials [J/mol]
        T   : temperature [K]
        P   : mixture pressure [Pa]
        P0  : standard-state pressure [Pa]
        """
      
      ntot = np.sum(n)
      x = np.clip(n / ntot, 1e-300, None)
      #dGdn_i = mu0 + R*T*(np.log(x * P/P0) + 1.0)
      # correction for the ntot dependence via mole fractions
      #dGdntot = -R*T * np.sum(n/ntot)

      #return dGdn_i + dGdntot # full gradient via chain rule
      return mu0 + R * T * np.log(phi * x * P / P0) # could add fugacity coefficient phi for real-gas correction but matches witin 3% w/ nist database?

# need to check how to generalize this to air... 
def gibbs(n, mu0, T, P, P0, phi):
    """
    Gibbs free energy of an ideal-gas mixture.

    n   : species mole numbers
    mu0 : standard chemical potentials [J/mol]
    T   : temperature [K]
    P   : mixture pressure [Pa]
    P0  : standard-state pressure [Pa]
    """

    ntot = np.sum(n)

    # Avoid log(0)
    n_safe = np.maximum(n, 1e-300) # elements between two mole arrays element-wise.
    x = n_safe / ntot

    mu = mu0 + R*T*np.log(phi* x * P/P0)

    return np.sum(n * mu)

def townsend_balance(EN_Td, bsolver, f_guess):
    """
    Return alpha/N - eta/N at a given E/N [Td].
    """

    bsolver.EN = EN_Td * solver.TOWNSEND

    # Rebuild operators for the new E/N if required by your BOLOS version
    bsolver.init()

    # Solve the EEDF
    f = bsolver.converge(
        f_guess,
        maxn=200,
        rtol=1e-6
    )

    muN = bsolver.mobility(f)

    k_ion = 0.0
    k_attach = 0.0

    for target, process in bsolver.iter_growth():

        kind = process.kind.upper()

        rate = bsolver.rate(
            f,
            process,
            weighted=True
        )

        if kind == "IONIZATION":
            k_ion += rate

        elif kind == "ATTACHMENT":
            k_attach += rate

    alpha_N = k_ion / (muN * bsolver.EN)
    eta_N   = k_attach / (muN * bsolver.EN)

    return alpha_N - eta_N

# for co2 - instead of minimizing G(n)
def gibbs_1d(t, mu0, T):
    """
    Total Gibbs free energy as a function of a single variable t = n_CO.

    Atom-balance constraints reduce the 5-species system to 1 free variable:
      Carbon:  n_CO2 + n_CO = 1          =>  n_CO2 = 1 - t
      Oxygen:  2*n_CO2 + n_CO + n_O = 2  =>  n_O   = t   (when q << 1)
      Charge:  n_O+ = n_e- = q           (solved analytically via Saha)
    """
    nCO2 = 1.0 - t
    nCO  = t
    nO   = t
    ntot0 = nCO2 + nCO + nO                 # neutral total (ignoring tiny q)
    q     = q_saha(nO, ntot0, mu0, T)
    n     = np.array([nCO2, nCO, nO, q, q])
    ntot  = np.sum(n)
    x     = np.clip(n / ntot, 1e-300, None)
    return np.sum(n * (mu0 + R * T * np.log(x * P / P0)))

# --- Main temperature sweep ---
# Build header dynamically from species list (set after this block)
# Header printed after species list is built below.


set_figure_defaults()
fig1, ax1 = make_single_column_figure()

# for eqilibrium mole fractions and densities
#markers = ["o", "+", "x", "s", "^", "v", "D", "p", "h", "*"]
#colors = ["blue", "orange", "purple", "green", "red", "brown", "pink", "gray", "olive", "cyan"]
# see more colors here!
#https://matplotlib.org/stable/gallery/color/named_colors.html
markers = ["o", "+", "x", "s", "^", "v", "D", "p", "h", "*", ">", "<"]
colors = ["blue", "orange", "purple", "green", "red", "brown", "pink", "gray", "olive", "turquoise", "mediumslateblue", "lightgreen"]


species = [sp for sp, _ in sorted(config.SPECIES_DENSITY_MAP.items(), key=lambda x: x[1])]

# Print header using actual species names
_hdr = f"{'T (K)':>10}  " + "  ".join(f"{sp:>12}" for sp in species)
print(_hdr)
print("-" * len(_hdr))

A = config.ATOM_BALANCE_MATRIX

# add a small amount in CO... N to avoid log(0) in the objective funtion, Gibbs free energy
n_initial = config.INITAL_MOLE_FRACTION_GUESS

# Atom-balance RHS must come from the physical composition only —
# using n_initial (which has 1e-10 placeholders) would corrupt b. 
# n_initial_physical = np.array([0.7809, 0.2094, 0.0093, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# b = A @ n_initial_physical
b = config.b

constraint = LinearConstraint(A, b, b)

bounds = Bounds(np.full(len(species), 1e-12), np.full(len(species), np.inf))

Xeq      = np.zeros((len(T_values), len(species)))
Xeq_ct   = np.zeros((len(T_values), len(species)))  # Cantera reference
rhoarray = np.zeros(len(T_values))
n_guess  = n_initial.copy()
EN_balance_values = []  # collect E/N_crit for each temperature
rho_ct = np.zeros(len(T_values)) # rho from cantera for comparison

for k, T in enumerate(T_values): # two indices for temperature and species sweep

    # get fugacity coefficient from Peng-Robinson EOS for real-gas correction
    Z, phi = rho_pr(T)

    # Cantera reference equilibrium for this T
    x_c, rho_ct[k] = cantera_equilibrium(T, P, species)
    Xeq_ct[k, :] = x_c # molar fractions as function of temperautre
   
    # Your function/table interpolation:
    mu0 = get_mu0(T, P, species) # add Z 

    # inhouse minimizer
    if T < 1000: # mixture unchanged below dissociation onset
        n_eq = n_guess.copy()
        
        Xeq[k, :] = n_eq / np.sum(n_eq)

        row = f"{T:>10.1f}  " + "  ".join(f"{n_eq[i]:>12.4e}" for i in range(len(species)))
        #row2 = f"{T:>10.1f}  " + "  ".join(f"{Xeq[k, i]:>12.4e}" for i in range(len(species)))

        print(row)
        #print(row2)
        color = "blue"

    else:

        result = minimize(
            gibbs,
            x0=n_guess,
            args=(mu0, T, P, P0, phi),
            method="SLSQP",
            jac=gibbs_grad, # manually input the gradient to speed up convergence; otherwise optimizer defaults to finite-difference approximation
            constraints=[constraint],
            bounds=bounds,
            options={
                "ftol": 1e-12,
                "maxiter": 8000  # increased for 16-species system at high T
            }
        )

        if not result.success:
            print(f"T = {T:.0f} K: {result.message}")

        n_eq = result.x

        # Only warm-start from successful solves — don't propagate bad solutions
        if result.success:
            n_guess = n_eq.copy()

        # Mole fractions: moles of species / total moles
        Xeq[k, :] = n_eq / np.sum(n_eq)

        converged = result.success
        row = f"{T:>10.1f}  " + "  ".join(f"{n_eq[i]:>12.4e}" for i in range(len(species)))
        #row2 = f"{T:>10.1f}  " + "  ".join(f"{Xeq[k, i]:>12.4e}" for i in range(len(species)))
        print(row)
        #print(row2)

        color = "blue" if converged else "red"

    if config.GET_EEDF == True:

        # call BOLOS
        energy_grid = grid.LinearGrid(0, 60., 400)
        bsolver = solver.BoltzmannSolver(energy_grid)

        # cross sectional files
        for fname in config.CROSS_SECTION_FILES:
            with open(os.path.join(config.LXCAT_DATA_DIR, fname)) as fp:
                bsolver.load_collisions(parser.parse(fp))
        
        # Set the conditions.  And initialize the solver
        # Guard against zero densities: BOLOS g3 division-by-zero occurs when a
        # target has density 0 and its cross-section integral collapses to g->0.
        # Clamp each density to a small floor so the solver stays well-defined.
        for sp_name, idx in config.SPECIES_DENSITY_MAP.items():                                                                                                                                                                                                                                           
            # if sp_name not in bsolver.target:
            #     continue
            bsolver.target[sp_name].density = max(float(Xeq[k, idx]), config.MIN_DENSITY) 

        print("debug:", list(bsolver.target))
        bsolver.kT = 300 * co.k / co.eV
        bsolver.EN = 300.0 * solver.TOWNSEND
        bsolver.init()

        # Start with a Maxwellian EEDF at 2 eV as initial guess
        f0 = bsolver.maxwell(2.0)

        # Solve the Boltzmann equation
        f1 = bsolver.converge(f0, maxn=50, rtol=1e-5)

        # Calculate the mean energy and build a better-resolved quadratic grid
        mean_energy = bsolver.mean_energy(f1)

        newgrid = grid.QuadraticGrid(0, 15 * mean_energy, 400)

        # Keep the original linear grid (matching the paper) but re-init
        bsolver.grid = energy_grid
        bsolver.init()
        print("Boltzmann solver was re-initialized with the new grid.")

        # Interpolate EEDF onto the new grid and converge again
        finterp = bsolver.grid.interpolate(f1, newgrid)
        f2 = bsolver.converge(finterp, maxn=200, rtol=1e-5)

        # Find E/N_crit: the E/N where alpha/N - eta/N = 0 (Townsend breakdown).
        # Scan for a sign change, then bisect to get an accurate root.
        EN_scan = np.linspace(10, 350, 100)
        EN_crit = None

        prev_val = townsend_balance(EN_scan[0], bsolver, f2)

        plt.figure()
        plt.plot(bsolver.grid.c, f2, label=f"T = {T:.0f} K")
        plt.xlabel("Energy (eV)")
        plt.ylabel("EEDF")
        plt.title("Electron Energy Distribution Function")
        plt.legend()
        plt.savefig(f"co2_eedf_{T:.0f}K.png", dpi=300, bbox_inches="tight")
#        plt.show()

    if config.GET_EN_CRIT == True:

        for EN in EN_scan[1:]:
            curr_val = townsend_balance(EN, bsolver, f2)
            if prev_val * curr_val <= 0:
                # Sign change found — bisect between EN-step and EN
                lo, hi = EN - (EN_scan[1] - EN_scan[0]), EN
                for _ in range(50):  # 50 bisection steps → ~1e-15 relative accuracy
                    mid = 0.5 * (lo + hi)
                    mid_val = townsend_balance(mid, bsolver, f2)
                    if prev_val * mid_val <= 0:
                        hi = mid
                    else:
                        lo = mid
                EN_crit = 0.5 * (lo + hi)
                print(f"T = {T:.0f} K: Townsend breakdown E/N_crit = {EN_crit:.2f} Td")
                break
            prev_val = curr_val

        if EN_crit is None:
            print(f"T = {T:.0f} K: No sign change found in [10, 350] Td — breakdown may be outside scan range.")

        EN_balance_values.append(EN_crit)  # None if not found

        
    # get density
    M_species = config.MOLAR_MASS_SPECIES
    M_mix = np.dot(Xeq[k, :], M_species)   # kg/mol, mixture molar mass at this T (Gibbs minimizer)
    M_mix_ct = np.dot(Xeq_ct[k, :], M_species)  # kg/mol, mixture molar mass from Cantera

    # Peng-Robinson compressibility factor and fugacity coefficient at this T and P
    # rho_pr uses pure-CO2 PR EOS — valid approximation when mixture is CO2-dominated
    Z, phi = rho_pr(T)

    rhoarray[k] = M_mix    * P / (Z * R * T)   # kg/m^3, real-gas density (Gibbs minimizer composition)
    rho_ct[k]   = M_mix_ct * P / (Z * R * T)   # kg/m^3, real-gas density (Cantera composition, same PR EOS)

    print(f"T = {T:.0f} K: M_mix = {M_mix:.4f} kg/mol, Z = {Z:.6f}, phi = {phi:.6f}")

    
    # for i, (marker, c, sp) in enumerate(zip(markers, colors, species)):
    #     # in-house Gibbs solver
    #     #val_g = max(Xeq[k, i], 1e-300)
    #     #ax1.semilogy(T, val_g, marker, color=color, markersize=3, markerfacecolor="none")

    #     # Cantera reference
    #     val_ct = max(Xeq_ct[k, i], 1e-300)
    #     ax1.semilogy(T, val_ct, marker, color=c, markersize=1.0, alpha=0.5)



if config.GET_EN_CRIT == True:
    # plot E/N_critical vs T for air
    fig_EN, (axEN) = make_single_column_figure()
    # Filter out None values (temperatures where no root was found)
    T_plot  = [T_values[i] for i, v in enumerate(EN_balance_values) if v is not None]
    EN_plot = [v           for v      in EN_balance_values           if v is not None]
    axEN.plot(T_plot, EN_plot, "o-", label="Present Work")
    axEN.set_xlabel("Temperature (K)")
    #axEN.set_ylim(40, 130.0)
    axEN.set_ylabel("E/N$_{crit}$ (Td)")
    axEN.set_title("Townsend Breakdown Condition")
    axEN.legend()
    np.savetxt("0.1bar_EN_critical_co2.txt", np.column_stack([T_plot, EN_plot]), fmt="%d", delimiter="\t", header="T\tEN", comments="")
    print("Saved 0.1bar_EN_critical_co2.txt")

    fig_EN.savefig("co2_equilibrium_EN.png", dpi=300, bbox_inches="tight")
    print("Saved co2_equilibrium_EN.png")


########### saving plots


ax1.set_ylabel("Mole fraction")
ax1.set_ylim(9e-5, 1.4)
ax1.set_xlim(300, 30000)
fig1.suptitle(
    f"CO$_2$ equilibrium Composition at 1 bar",
   #x=(0.12 + 0.92) / 2,
    x = 0.4,
    y=0.7,
    ha="center",
    fontweight="bold",
    fontsize=10
)
ax1.set_xlabel("Temperature (K)")

# Build a clean legend: one entry per species (marker shape), plus solver color key
handles = []
#colors = ["blue", "orange", "purple", "green", "red", "brown", "pink", "gray", "olive", "cyan"]
for m, c, sp in zip(markers, colors, species):
    #handles.append(mlines.Line2D([], [], color=c,   marker=m, linestyle="None", markersize=5, label=f"{sp} (SLSQP)"))
    handles.append(mlines.Line2D([], [], color=c, marker=m, linestyle="None", markersize=5, label=f"{sp} (VCS)"))

fig1.subplots_adjust(right=0.75, top = .6) # adjust margins; small
leg = ax1.legend(
    handles=handles,
    fontsize=6,
    loc="upper right",
    bbox_to_anchor=(2.0, 0.8),
    ncol=2
)

# plot density vs temperature for air at 100 bar, using Peng-Robinson EOS
fig2, ax2 = make_single_column_figure()

#M = config.SPECIS_WEIGHT 
#ax2.plot(T_values,P*M/(R*T_values),'--',color='tab:gray',label='Ideal gas (Z = 1)')
ax2.plot(T_values, rho_ct,   markersize=1.0, marker='o', linestyle='-', color='tab:orange',label='Real Air (Peng-Robinson, 310-10000 K)')
ax2.plot(T_values, rhoarray, markersize=1.0, marker='s', linestyle='-', color='tab:blue',label='Present Work (Peng-Robinson, 310-10000 K)')

ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Temperature (K)',fontsize=12)
ax2.set_ylabel('Density (kg/m$^3$)',fontsize=12)
ax2.set_title('CO$_2$ density at 1 bar')
ax2.set_xlim(300, 25000)
ax2.grid(True,which='both',alpha=0.3)
ax2.legend(fontsize=6)

fig2.tight_layout()


# first make directories
OUTDIR = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

fig1.savefig(OUTDIR / config.FIGNAME_EQUILB_COMP, dpi=300, bbox_inches="tight")
print("Saved " + config.FIGNAME_EQUILB_COMP)


fig2.savefig(OUTDIR / config.FIGNAME_DENSITY, dpi=300, bbox_inches="tight")
print("Saved " + config.FIGNAME_DENSITY)

np.savetxt(OUTDIR / config.TXTNAME_DENSITY, np.column_stack([T_values, rhoarray, rho_ct]), fmt="%0.8f", delimiter="\t", header="T\trho\trho_ct", comments="")
print("Saved " + config.TXTNAME_DENSITY)