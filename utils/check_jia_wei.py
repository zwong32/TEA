import warnings
import sys
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import cantera           as ct
import os
import config
import scipy.constants      as co
from co2_density_plot import lnphi, rho_pr
from bolos import grid, solver, grid, target, parser

from scipy.optimize import minimize_scalar, minimize, LinearConstraint, Bounds
from figure_style import *

R=8.314462; M=0.04401; # kg
Tc,Pc,w=304.13,73.77e5,0.2239
T = 308 # K from jia wei's thesis

file = "vB_vs_density_jiaWei.csv" 

# initialize arrays to store stuff
p_array,N_array,rho_array,VB_array = np.array([]),np.array([]),np.array([]),np.array([])

def get_p_from_pr(T, rho): # return p in Pa
    v = M/rho
    k=0.37464+1.54226*w-0.26992*w**2
    a = 0.45724*R**2*Tc**2/Pc*(1 + k*(1 - np.sqrt(T/Tc)))**2
    b = 0.07780*R*Tc/Pc

    return R*T/(v-b) - a/(v*(v+b)+b*(v-b))

# sort files numerically # data is density (kg/m^3) as x value, breakdown voltage (kV) as y value
data = np.loadtxt(file, delimiter=",", skiprows=1)

# calculate the pressure for isothermal T for each rho inside the file
for i in range(len(data)):
    rho = data[i,0]     # kg/m^3
    P = get_p_from_pr(T, rho)
    N_A = co.Avogadro    # units mol^-1

    # get number density in m^-3
    N = data[:,1]*N_A/M

    print("rho: %.4f kg/m^3, P: %.4f Pa, N: %.4e m^-3" % (rho, P, N[i]))

    p_array = np.append(p_array, P) # Pa
    N_array = np.append(N_array, N[i]) # m^-3
    rho_array = np.append(rho_array, rho) #kg m^-3
    VB_array = np.append(VB_array, data[i,1]) # kV

# save output to csv file
np.savetxt("vB_vs_density_nA_jiaWei.txt", 
           np.column_stack([p_array, N_array, rho_array, VB_array]), fmt="%8.8f", delimiter="\t", 
                            header="P(Pa)\tN(m^-3)\trho(kg/m^3)\tVB(kV)", comments="")

print("Saved vB_vs_density_nA_jiaWei.txt")

# plot to double check
set_figure_defaults()
fig, ax = make_single_column_figure()
ax.plot(p_array*1e-7, data[:, 0], 'o')       # plot pressure vs density 
ax.set_xlabel("Pressure (MPa)", fontsize=12)
ax.set_title('CO$_2$ Density vs Temperature',fontsize=12)
ax.set_ylabel("Density (kg/m$^3$)", fontsize=12)
fig.tight_layout()

fig.savefig("p_vs_density_jiaWei", dpi=300, bbox_inches="tight")
print("Saved p_vs_density_jiaWei.png")