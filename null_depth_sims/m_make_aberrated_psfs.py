"""
Make a series of PSFs from aberrations and save them to disk
"""

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

fname = "./aberrated_psfs.npz"


# Wavefront properties
diameter = 1.8
wf_npixels = 256

# simulation params
n_runs = 1_000
wavelength = 1.65e-6

# psf params
input_f_number = 1.25
# input_f_number = 3
focal_length = input_f_number * diameter
psf_npixels = 256
psf_pixel_scale = 2 * 2 * 5 / psf_npixels


# aberration params:
n_zernikes = 3  # including piston
zernike_rms = 50e-9


coords = dlu.pixel_coords(wf_npixels, diameter)
circle = dlu.circle(coords, diameter / 2)

# Zernike aberrations
zernike_indexes = np.arange(1, n_zernikes + 1)
coeffs = np.zeros(zernike_indexes.shape)
# coeffs = 300e-9 * jr.normal(jr.PRNGKey(1), zernike_indexes.shape)
# coeffs = 50e-9 * np.array([0.0, 1.0, 0.0])
coords = dlu.pixel_coords(wf_npixels, diameter)
basis = dlu.zernike_basis(zernike_indexes, coords, diameter)

layers = [("aperture", dl.layers.BasisOptic(basis, circle, coeffs, normalise=True))]


# # Construct Optics
optics = dl.CartesianOpticalSystem(
    wf_npixels, diameter, layers, focal_length, psf_npixels, psf_pixel_scale
)

source = dl.PointSource(flux=1e5, wavelengths=[wavelength])


def prop_fibre_input_field(optics, zernikies):
    optics = optics.set("aperture.coefficients", zernikies)
    output = source.model(optics, return_wf=True)
    ouput_wf_complex = (
        (output.amplitude * np.exp(1j * output.phase))
        * source.spectrum.weights[:, None, None]
    ).sum(axis=0)

    return ouput_wf_complex


ouput_wf_complex = prop_fibre_input_field(optics, coeffs)

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
plt.imshow(
    np.abs(ouput_wf_complex),
    extent=[
        -psf_npixels * psf_pixel_scale / 2,
        psf_npixels * psf_pixel_scale / 2,
        -psf_npixels * psf_pixel_scale / 2,
        psf_npixels * psf_pixel_scale / 2,
    ],
)
plt.colorbar(label="Photons")


plt.subplot(1, 3, 3)
plt.title("psf phase")
plt.imshow(np.angle(ouput_wf_complex), cmap=circular_cmap)
plt.colorbar(label="Photons")
plt.show()

zernike_coeffs = zernike_rms * jr.normal(jr.PRNGKey(1), (n_runs, n_zernikes))

print("Making PSFs...", end="")
outputs = jax.vmap(prop_fibre_input_field, in_axes=(None, 0))(optics, zernike_coeffs)
print("done")

np.savez(
    fname,
    outputs=outputs,
    zernike_coeffs=zernike_coeffs,
    wavelength=wavelength,
    psf_pixel_scale=psf_pixel_scale,
    psf_npixels=psf_npixels,
    input_f_number=input_f_number,
    zernike_rms=zernike_rms,
    n_zernikes=n_zernikes,
    non_aberrated=ouput_wf_complex,
)
