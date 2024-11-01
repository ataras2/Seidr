import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp
from prettytable import PrettyTable


import lanternfiber

import matplotlib.pyplot as plt

# PL params
n_core = 1.44
n_cladding = 1.4345
# n_core = 1.44
# n_cladding = n_core - 0.04
wavelength = 1.63  # microns
core_radius = 15.9 / 2  # microns
max_r = 6


lf = lanternfiber.lanternfiber(
    n_core=n_core,
    n_cladding=n_cladding,
    core_radius=core_radius,
    wavelength=wavelength,
)

# look at some of the fiber modes
lf.find_fiber_modes()
lf.make_fiber_modes(npix=512 // 2, show_plots=True, max_r=max_r)
