# imports
#from cantera.test.python.test_kinetics import species
from thermo.gibbs_minizer import get_equilibrium
from utils.figure_style import *
from utils.imports import *
from utils.helpers import *
from utils.co2_density_plot import *
from thermo.cantera_eq import *
from thermo.bolos_eq import *
# from thermo.gibbs_minimizer import *
from config import *

set_figure_defaults()

# ============================================================================================= 
# Set constants and parameters
# =============================================================================================
P, P0, T_values, species_names, R = set_constants(config)

# validate for air at 0.1MPa from 0 to 5000K from this paper: https://doi.org/10.1063/1.5145146

Xeq, Xeq_ct, rhoarray, EN_balance_values, rho_ct      = init_arrays(config, T_values)
markers, colors                                       = set_fig_aesthetics()
species, A, n_initial, b, constraint, bounds, n_guess = load_data(config, T_values)

# ============================================================================================= 
# Make output directory
# =============================================================================================
OUTDIR = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================================= 
# Get equilibrium composition as function of T
# =============================================================================================

# Print header using actual species names
_hdr = f"{'T (K)':>10}  " + "  ".join(f"{sp:>12}" for sp in species_names)
print(_hdr)
print("-" * len(_hdr))

# start T loop
for k, T in enumerate(T_values): # two indices for temperature and species sweep

    # inhouse minimizer
    if config.INHOUSE_MINIMIZER == True:

        # get fugacity coefficient from Peng-Robinson EOS for real-gas correction
        Z, phi = rho_pr(T)
    
        # Cantera reference equilibrium for this T
        x_c, rho_ct[k] = cantera_equilibrium(T, P, species)
        Xeq_ct[k, :] = x_c # molar fractions as function of temperature from Cantera's VCS solver
        
        # get chemical potential based on mixture
        if config.SPECIES_NAME == "air":
            mu0 = get_mu0_air(T, P, species) # add Z
        elif config.SPECIES_NAME == "co2":
            mu0 = get_mu0(T, P, species) # add Z 
        
        n_eq, Xeq = get_equilibrium(T, P, n_guess, mu0, P0, phi, species, Xeq, k, A, b)


    if config.GET_DENSITY == True:
        # ---- get density ------------------------------------------------
        M_species = config.MOLAR_MASS_SPECIES
        M_mix = np.dot(Xeq[k, :], M_species)   # kg/mol, mixture molar mass at this T (Gibbs minimizer)
        M_mix_ct = np.dot(Xeq_ct[k, :], M_species)  # kg/mol, mixture molar mass from Cantera

        # Peng-Robinson compressibility factor and fugacity coefficient at this T and P
        # rho_pr uses pure-CO2 PR EOS — valid approximation when mixture is CO2-dominated
        Z, phi = rho_pr(T)
        rhoarray[k] = M_mix    * P / (Z * R * T)   # kg/m^3, real-gas density (Gibbs minimizer composition)
        rho_ct[k]   = M_mix_ct * P / (Z * R * T)   # kg/m^3, real-gas density (Cantera composition, same PR EOS)
    
        print(f"T = {T:.0f} K: M_mix = {M_mix:.4f} kg/mol, Z = {Z:.6f}, phi = {phi:.6f}")

    # np.savetxt(OUTDIR / config.TXTNAME_COMP, np.column_stack([T_values, Xeq]), fmt="%0.8e", delimiter="\t", header="T\t" + "\t".join(species), comments="")
    # np.savetxt(OUTDIR / config.TXTNAME_DENSITY, np.column_stack([T_values, rhoarray, rho_ct]), fmt="%0.8f", delimiter="\t", header="T\trho\trho_ct", comments="")
    
    # print(f"Saved equilibrium mole fractions to {config.TXTNAME_COMP}")
    # print(f"Saved density data to {config.TXTNAME_DENSITY}")

    # #if config.GET_EEDF == True:
    # EN_crit, bsolver, f2, prev_val = get_EEDF(T, P, species, Xeq[k, :])
           
    # EN_balance_values[k] = get_EN_crit(bsolver, f2, prev_val, T)
    

    if config.GET_EEDF == True:

        # get EEDF and Townsend breakdown voltage for this T
        EN_crit, bsolver, f2, prev_val = get_EEDF(T, P, species, Xeq[k, :])
       
        if config.GET_EN_CRIT == True:
            EN_balance_values[k] = get_EN_crit(bsolver, f2, prev_val, T)


if config.SAVE_RESULTS == True:
    np.savetxt(OUTDIR / config.TXTNAME_COMP, np.column_stack([T_values, Xeq_ct]), fmt="%0.8e", delimiter="\t", header="T\t" + "\t".join(species), comments="")
    np.savetxt(OUTDIR / config.TXTNAME_DENSITY, np.column_stack([T_values, rhoarray, rho_ct]), fmt="%0.8f", delimiter="\t", header="T\trho\trho_ct", comments="")
        
    print(f"Saved equilibrium mole fractions to {config.TXTNAME_COMP}")
    print(f"Saved density data to {config.TXTNAME_DENSITY}")

    make_plots(T_values, Xeq, Xeq_ct, rhoarray, rho_ct, EN_balance_values, species, markers, colors, OUTDIR)
    print(f"Saved figures to OUTDIR: {config.FIGNAME_EQUILB_COMP}, {config.FIGNAME_DENSITY}")
    
# # if __name__ == "__main__":
# #     main()
