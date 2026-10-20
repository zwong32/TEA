import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import os

from figure_style import *
from pathlib import Path


set_figure_defaults()
fig, ax = make_single_column_figure()
fig2, ax2 = make_single_column_figure()


directory_path = Path("/Users/zoellewong/Desktop/pyfiles/research/gibbsFreeEnergy/co2/")
handles = ["Present Work", "Cantera", "Tanaka"]

files = sorted(directory_path.glob("co2_equilibrium_density_1bar.txt")) # sort files numerically

# Loop through only .txt files in this directory
for labelName, file in zip(handles, files):

    print(f"Processing file: {file}, Label Name: {labelName}")
    data = np.loadtxt(file, delimiter="\t", skiprows=1)
    T_plot  = data[:, 0]
    rho1 = data[:, 1]
    rho2 = data[:, 2]
    delta = (rho1 - rho2) / rho2 * 100


    ax.plot(T_plot, rho1, "o", label=handles[0], markersize=1, alpha=0.7)
    ax.plot(T_plot, rho2, "o", label=handles[1], markersize=1, alpha=0.7)

    ax2.plot(T_plot, delta, "o", label=handles[2])

tanaka_data = np.loadtxt("validation_data/oneMPa.csv", delimiter=",", skiprows=1)
T_tanaka = tanaka_data[:, 0]
rho_tanaka = tanaka_data[:, 1]
ax.plot(T_tanaka, rho_tanaka, "o", label=handles[2], markersize=1, alpha=0.7, color="green")   

# fromtting
ax.set_xlabel("Temperature (K)")
ax.set_ylabel('Density (kg/m$^3$)',fontsize=12)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('CO$_2$ Density at 100 Bar')
ax.legend(fontsize=8)
fig.savefig("co2_equilibrium_density_100bar.png", dpi=300, bbox_inches="tight")
print("Saved co2_equilibrium_density_100bar.png")


# format delta plot

ax2.set_xlabel("Temperature (K)")
ax2.set_ylabel("$\\Delta$ $\\rho$ ($\%$)")
ax2.set_title("Difference in Density")
ax2.legend()
fig2.savefig("co2_equilibrium_density_100bar_diff.png", dpi=300, bbox_inches="tight")
print("Saved co2_equilibrium_density_100bar_diff.png")