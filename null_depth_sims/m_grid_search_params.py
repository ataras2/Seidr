"""
Grid search over fiber parameters to match the incoming beam
"""

import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp

import dLux as dl
import dLux.utils as dlu

import lanternfiber

import matplotlib.pyplot as plt


# Wavefront properties
diameter = 1.8
wf_npixels = 256

# simulation params
wavelength = 1.65e-6

# psf params
# input_f_number = 1.25
input_f_number = 3
focal_length = input_f_number * diameter
psf_npixels = 256
psf_pixel_scale = 2 * 2 * 10 / psf_npixels


# aberration params:
n_zernikes = 3  # including piston
zernike_rms = 100e-9


coords = dlu.pixel_coords(wf_npixels, diameter)
circle = dlu.circle(coords, diameter / 2)

# Zernike aberrations
zernike_indexes = np.arange(1, n_zernikes + 1)
coeffs = np.zeros(zernike_indexes.shape)
coords = dlu.pixel_coords(wf_npixels, diameter)
basis = dlu.zernike_basis(zernike_indexes, coords, diameter)

layers = [("aperture", dl.layers.BasisOptic(basis, circle, coeffs, normalise=True))]

source = dl.PointSource(flux=1e5, wavelengths=[wavelength])

optics = dl.CartesianOpticalSystem(
    wf_npixels, diameter, layers, focal_length, psf_npixels, psf_pixel_scale
)


def compute_values(core_radius, delta_n, optics):
    psf_pixel_scale = 2 * 2 * core_radius / psf_npixels
    # # Construct Optics
    optics = optics.set("psf_pixel_scale", psf_pixel_scale)

    def prop_fibre_input_field(optics):
        output = source.model(optics, return_wf=True)
        ouput_wf_complex = (
            (output.amplitude * np.exp(1j * output.phase))
            * source.spectrum.weights[:, None, None]
        ).sum(axis=0)

        return ouput_wf_complex

    ouput_wf_complex = prop_fibre_input_field(optics)

    # now look at making the fiber
    n_core = 1.44
    n_cladding = n_core - delta_n
    core_radius = core_radius
    max_r = 2

    lf = lanternfiber.lanternfiber(
        n_core=n_core,
        n_cladding=n_cladding,
        core_radius=core_radius,
        wavelength=wavelength * 1e6,
    )

    lf.find_fiber_modes(verbose=False)
    lf.make_fiber_modes(npix=psf_npixels // 2, show_plots=False, max_r=max_r)

    n_modes = len(lf.allmodefields_rsoftorder)

    total_injection, injections = lf.calc_injection_multi(
        input_field=ouput_wf_complex,
        mode_field_numbers=list(range(n_modes)),
        show_plots=False,
    )

    return n_modes, total_injection, injections[0]


core_radii = np.linspace(3, 20, 31)
del_n = np.linspace(1.44 - 1.4345, 1.44 - 1.40, 30)

rr, dndn = np.meshgrid(core_radii, del_n)

n_modes, total_injection, injections = [], [], []

# loop over the grid and store values
for i, (r, dn) in enumerate(zip(rr.flatten(), dndn.flatten())):
    print(f"Running {i+1}/{rr.size}", end="\r")
    n_modes_, total_injection_, injections_ = compute_values(onp.array(r), dn, optics)
    n_modes.append(n_modes_)
    total_injection.append(total_injection_)
    injections.append(injections_)

n_modes = np.array(n_modes).reshape(rr.shape)
total_injection = np.array(total_injection).reshape(rr.shape)
injections = np.array(injections).reshape(rr.shape)

plt.figure()
plt.subplot(1, 3, 1)
plt.pcolormesh(core_radii, del_n, n_modes)
plt.xlabel("Core radius")
plt.ylabel("Delta n")
plt.colorbar()
plt.title("n_modes")

plt.subplot(1, 3, 2)
plt.pcolormesh(core_radii, del_n, total_injection)
plt.xlabel("Core radius")
plt.ylabel("Delta n")
plt.colorbar()
plt.title("Total injection")

plt.subplot(1, 3, 3)
plt.pcolormesh(core_radii, del_n, injections)
plt.xlabel("Core radius")
plt.ylabel("Delta n")
plt.colorbar()
plt.title("Injection in 0th mode")

plt.show()
