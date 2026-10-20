
#from test7 import cantera_equilibrium, get_mu0, gibbs_grad, gibbs, townsend_balance
from utils.figure_style import *
from utils.imports import *
#import cantera           as ct
#from bolos import grid, solver, grid, target, parser

#matplotlib.use("Agg")

# for CO2 at 100 bar, using Peng-Robinson EOS
R=8.314462; M=0.04401; P=100e5 # units: K, kg/mol, Pa
Tc,Pc,w=304.13,73.77e5,0.2239
k=0.37464+1.54226*w-0.26992*w**2


# peng-robinson departure equation to get the natural log of the fugacity coefficient; ln(phi)
def lnphi(Z, A, B):
    s2 = np.sqrt(2)
    return (Z - 1 - np.log(Z - B) - A/(2*s2*B)*np.log((Z + (1 + s2)*B)/(Z + (1 - s2)*B)))


def rho_pr(T):
    a = 0.45724*R**2*Tc**2/Pc*(1 + k*(1 - np.sqrt(T/Tc)))**2
    b = 0.07780*R*Tc/Pc
    A = a*P/(R*T)**2
    B = b*P/(R*T)
    roots = np.roots([1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)])
    Zs = np.sort(roots[np.abs(roots.imag) < 1e-9].real)  # real roots only
    Zs = Zs[Zs > B]                                      # physical: V > b

    if len(Zs) == 1:
        Z = Zs[0]
        phi = np.exp(lnphi(Z, A, B))

    else:
        
        Zliq, Zvap = Zs[0], Zs[-1]        # skip the middle (unstable) root

        if lnphi(Zliq, A, B) < lnphi(Zvap, A, B):
            Z = Zliq
            phi = np.exp(lnphi(Zliq, A, B))
        else:
            Z = Zvap
            phi = np.exp(lnphi(Zvap, A, B))

    return Z, phi #,  P*M/(Z*R*T) - this is density 

# for debugging
# fig, ax = make_single_column_figure()

# Tall=np.logspace(np.log10(300),np.log10(25000),300)
# #Treal=np.linspace(300, 2000,100)
# #fig,ax=plt.subplots(figsize=(8,5.5),dpi=150)
# ax.plot(Tall,P*M/(R*Tall),'--',color='tab:gray',label='Ideal gas (Z = 1)')

# ax.plot(Treal,[rho_pr(T) for T in Treal],color='tab:blue',label='Real Air (Peng-Robinson, 300-1800 K)')
# ax.set_xscale('log'); ax.set_yscale('log')
# ax.set_xlabel('Temperature (K)',fontsize=12)
# ax.set_ylabel('Density (kg/m$^3$)',fontsize=12)
# 
# ax.set_title('Air density at 100 bar vs temperature')
# ax.set_xlim(300,25000)
# ax.grid(True,which='both',alpha=0.3)
# ax.legend()
# fig.tight_layout()
# fig.savefig('co2_density_100bar.png')
