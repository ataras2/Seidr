import numpy as onp

import lanternfiber

import matplotlib.pyplot as plt


wavels = onp.linspace(1.5, 1.8, 20)
diameters = onp.linspace(10, 40, 30)

n_modes = onp.zeros((len(wavels), len(diameters)))

for i, wavel in enumerate(wavels):
    for j, diameter in enumerate(diameters):
        lf = lanternfiber.lanternfiber(
            n_core=1.44,
            n_cladding=1.4345,
            core_radius=diameter / 2,
            wavelength=wavel,
        )

        lf.find_fiber_modes()
        n_modes[i, j] = lf.nmodes


plt.figure()
plt.imshow(n_modes, origin="lower", aspect="auto", extent=(diameters[0], diameters[-1], wavels[0], wavels[-1]))
plt.colorbar()
plt.show()
