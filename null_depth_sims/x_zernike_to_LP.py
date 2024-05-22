import jax
import jax.numpy as np
import jax.random as jr

import dLux as dl
import dLux.utils as dlu

# Plotting/visualisation
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 72

# Wavefront properties
diameter = 1.8
wf_npixels = 256

coords = dlu.pixel_coords(wf_npixels, diameter)
circle = dlu.circle(coords, diameter / 2)

# Zernike aberrations
zernike_indexes = np.arange(1, 4)
# coeffs = np.zeros(zernike_indexes.shape)
coeffs = 300e-9 * jr.normal(jr.PRNGKey(1), zernike_indexes.shape)
coeffs = 300e-9 * np.array([0.0, 1.0, 0.0])
coords = dlu.pixel_coords(wf_npixels, diameter)
basis = dlu.zernike_basis(zernike_indexes, coords, diameter)

layers = [("aperture", dl.layers.BasisOptic(basis, circle, coeffs, normalise=True))]

# psf params
input_f_number = 1.25
focal_length = input_f_number * diameter
psf_npixels = 256
psf_pixel_scale = 20 / psf_npixels  # in microns

# # Construct Optics
optics = dl.CartesianOpticalSystem(
    wf_npixels, diameter, layers, focal_length, psf_npixels, psf_pixel_scale
)

# psf_pixel_scale = 1e-2
# optics = dl.AngularOpticalSystem(
#     wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale
# )
# Create a point source
source = dl.PointSource(flux=1e5, wavelengths=[1.6e-6])

# source = dl.BinarySource(
#     wavelengths=[1.6e-6],
#     position=(0, 0),
#     mean_flux = 1e5,
#     separation=dlu.arcsec2rad(20e-3),
#     position_angle=0,
#     contrast=1,
# )


# Model the psf and add some photon noise
# psf = optics.model(source, return_wf=True).sum(axis=0)
output = source.model(optics, return_wf=True)
# data = jr.poisson(jr.PRNGKey(1), psf)

ouput_wf_complex = (
    (output.amplitude * np.exp(1j * output.phase))
    * source.spectrum.weights[:, None, None]
).sum(axis=0)

# Get aberrations
opd = optics.aperture.eval_basis()

support = optics.aperture.transmission
support_mask = support.at[support < 0.5].set(np.nan)

# Plot
cmap = colormaps["inferno"]
circular_cmap = colormaps["twilight"]
cmap.set_bad("k", 0.5)
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(support_mask * opd * 1e6, cmap=cmap)
plt.title("Aberrations")
plt.colorbar(label="um")

plt.subplot(1, 3, 2)
plt.title("psf amplitude")
plt.imshow(np.abs(ouput_wf_complex))
plt.colorbar(label="Photons")


plt.subplot(1, 3, 3)
plt.title("psf phase")
plt.imshow(np.angle(ouput_wf_complex), cmap=circular_cmap)
plt.colorbar(label="Photons")


import lanternfiber

n_core = 1.44
n_cladding = 1.4345
wavelength = 1.5  # microns
core_radius = 32.8 / 2  # microns
max_r = 2
n_modes = 19

max_l = 10
show_plots = True

lf = lanternfiber.lanternfiber(
    n_core=n_core,
    n_cladding=n_cladding,
    core_radius=core_radius,
    wavelength=wavelength,
    nmodes=n_modes,
)

lf.find_fiber_modes()
lf.make_fiber_modes(npix=psf_npixels // 2, show_plots=False, max_r=max_r)

lf.calc_injection_multi(
    ouput_wf_complex, mode_field_numbers=list(range(n_modes)), show_plots=True
)


def prop_fibre_input_field(optics, zernikies):
    optics = optics.set("aperture.coefficients", zernikies)
    output = source.model(optics, return_wf=True)
    ouput_wf_complex = (
        (output.amplitude * np.exp(1j * output.phase))
        * source.spectrum.weights[:, None, None]
    ).sum(axis=0)

    return ouput_wf_complex


def compute_LP_overlaps(optics, zernikies):
    ouput_wf_complex = prop_fibre_input_field(optics, zernikies)
    integral = lf.calc_injection_multi(
        input_field=ouput_wf_complex,
        mode_field_numbers=[0, 1, 2],
        show_plots=False,
    )
    return integral[1]


# print(
#     compute_LP_overlaps(optics, np.array([0.0, 0.0, 0.0])),
#     compute_LP_overlaps(optics, np.array([50e-9, 0.0, 0.0])),
#     compute_LP_overlaps(optics, np.array([0.0, 100e-9, 0.0])),
# )

tilts = np.linspace(-300e-9, 300e-9, 20)

overlaps = jax.vmap(
    lambda tilt: compute_LP_overlaps(optics, np.array([0.0, tilt, 0.0]))
)(tilts)

plt.figure()
plt.plot(tilts, overlaps)
plt.legend(["LP1", "LP2", "LP3"])
plt.show()
