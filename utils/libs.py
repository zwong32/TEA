# python file of all the imports 
import warnings
import sys
import pandas as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import cantera           as ct # good for ideal gas behavior
from CoolProp import CoolProp  as cp  # good for real gas behavior

import os
import config
# from co2_density_plot import lnphi, rho_pr
from bolos import grid, solver, grid, target, parser
import scipy.constants as co
from scipy.optimize import minimize_scalar, minimize, LinearConstraint, Bounds
from pathlib import Path
