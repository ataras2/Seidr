"""
Check coupling into a SMF-28 fibre for a range of f numbers
"""

import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp

import lanternfiber


import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 72


import dLux as dl
import dLux.utils as dlu

wavel = 1.63  # microns

n_pix = 256  # full width

n_core = 1.44
n_cladding = 1.4345

max_r = 2

print(f"% difference in n_cladding: {100*(n_core - n_cladding)/n_core:.2f}")

core_diameter = 8.2  # microns

lf = lanternfiber.lanternfiber(
    n_core=n_core,
    n_cladding=n_cladding,
    core_radius=core_diameter / 2,
    wavelength=wavel,
)

lf.find_fiber_modes()
lf.make_fiber_modes(npix=n_pix // 2, show_plots=True, max_r=max_r)


# Wavefront properties
diameter = 1.8
wf_npixels = 256

# psf params
# input_f_number = 1.25
# input_f_number = 12.5
input_f_number = 3.0
focal_length = input_f_number * diameter
psf_npixels = n_pix
psf_pixel_scale = max_r * core_diameter / psf_npixels


coords = dlu.pixel_coords(wf_npixels, diameter)
circle = dlu.circle(coords, diameter / 2)

# Zernike aberrations
zernike_indexes = np.arange(1, 2)
coeffs = np.zeros(zernike_indexes.shape)
# coeffs = 300e-9 * jr.normal(jr.PRNGKey(1), zernike_indexes.shape)
# coeffs = tip_tilt_rms * np.array([0.0, 1.0, 0.0])
coords = dlu.pixel_coords(wf_npixels, diameter)
basis = dlu.zernike_basis(zernike_indexes, coords, diameter)

layers = [("aperture", dl.layers.BasisOptic(basis, circle, coeffs, normalise=True))]


# # Construct Optics
optics = dl.CartesianOpticalSystem(
    wf_npixels, diameter, layers, focal_length, psf_npixels, psf_pixel_scale
)

source = dl.PointSource(flux=1.0, wavelengths=[wavel * 1e-6])


def plot_output_wf(wf):
    plt.figure()
    plt.subplot(121)
    plt.imshow(
        np.abs(wf),
        extent=[
            -psf_npixels * psf_pixel_scale / 2,
            psf_npixels * psf_pixel_scale / 2,
            -psf_npixels * psf_pixel_scale / 2,
            psf_npixels * psf_pixel_scale / 2,
        ],
    )
    plt.colorbar()
    plt.title("Amplitude")

    plt.subplot(122)
    plt.imshow(
        np.angle(wf),
        extent=[
            -psf_npixels * psf_pixel_scale / 2,
            psf_npixels * psf_pixel_scale / 2,
            -psf_npixels * psf_pixel_scale / 2,
            psf_npixels * psf_pixel_scale / 2,
        ],
    )
    plt.colorbar()
    plt.title("Phase")


def prop_fibre_input_field(optics, f_number):
    optics = optics.set("focal_length", f_number * diameter)
    output = source.model(optics, return_wf=True)
    ouput_wf_complex = (
        (output.amplitude * np.exp(1j * output.phase))
        * source.spectrum.weights[:, None, None]
    ).sum(axis=0)

    return ouput_wf_complex


def compute_overlap_int(f_number):
    ouput_wf_complex = prop_fibre_input_field(optics, f_number)
    return lf.calc_injection_multi(
        input_field=ouput_wf_complex,
        mode_field_numbers=list(range(len(lf.allmodefields_rsoftorder))),
        show_plots=False,
        return_abspower=True,
    )[0]


wf = prop_fibre_input_field(optics, input_f_number)
plot_output_wf(wf)
plt.pause(0.1)

f_numbers = np.linspace(3.1, 5.2, 50)
overlaps = jax.vmap(compute_overlap_int)(f_numbers)

print(
    f"Max overlap: {overlaps.max()} occurs at f_number: {f_numbers[overlaps.argmax()]:.2f} +/- {(f_numbers[1] - f_numbers[0])/2:.2f}"
)

plt.figure()
plt.plot(f_numbers, overlaps)
plt.xlabel("f number")
plt.ylabel("Coupling efficiency")

plt.show()
