from utils.figure_style import *
from utils.imports      import *


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
      R = ct.gas_constant / 1000.0  # J/mol/K  (Cantera uses J/kmol/K)

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
    P0  : standard-state pressure [Pa]
    R   : gas constant [J/mol/K]
    """

    R = ct.gas_constant / 1000.0  # J/mol/K  (Cantera uses J/kmol/K)

    ntot = np.sum(n)

    # Avoid log(0)
    n_safe = np.maximum(n, 1e-300) # elements between two mole arrays element-wise.
    x = n_safe / ntot

    mu = mu0 + R*T*np.log(phi* x * P/P0)

    return np.sum(n * mu)


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

def get_equilibrium(T, P, n_guess, mu0, P0, phi, species, Xeq, k, A, b):
    """
    Goal: Compute the equilibrium mole fractions for a given temperature and pressure. 

    Method: minize gibbs free energy equation using scipy.optimize.minimize with SLSQP method.
    """
    bounds = Bounds(np.full(len(species), 1e-12), np.full(len(species), np.inf))
    constraint = LinearConstraint(A, b, b)


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

        color = "blue" if converged else "red"

    return n_eq, Xeq
