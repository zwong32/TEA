from utils.figure_style import *
from utils.imports import *
from utils.co2_density_plot import *

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

def get_EEDF(T, P, species, Xeq_row):
    """
    Use BOLOS to solve for the critical reduced electric field (E/N_crit) for
    a given temperature, pressure, and species composition.  The critical reduced
    electric field is defined as the E/N where alpha/N - eta/N = 0 (Townsend breakdown).
    """

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
        if sp_name not in bsolver.target: 
            continue

        bsolver.target[sp_name].density = max(float(Xeq_row[idx]), config.MIN_DENSITY) 
    print(bsolver.target)
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
    print("this is prev_val", prev_val)
    print("this is EN_crit", EN_crit)

    return EN_crit, bsolver, f2, prev_val


def get_EN_crit(bsolver, f2, prev_val, T):
    EN_scan = np.linspace(10, 1000, 100) ## DANGER: HARD-CODED SCAN RANGE
    EN_crit = np.nan
    
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

    if np.isnan(EN_crit):
        print(f"T = {T:.0f} K: No sign change found in [10, 1000] Td — breakdown may be outside scan range.")

    return EN_crit
    