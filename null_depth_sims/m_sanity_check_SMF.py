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

# wavel = 1.63  # microns
wavel = 1.55  # microns

n_pix = 256  # full width

n_core = 1.44
# n_cladding = 1.4345
n_cladding = (1 - 0.0036) * n_core

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
# lf.make_fiber_modes(npix=n_pix // 2, show_plots=True, max_r=max_r)


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


# r_vals = [2, 2.5, 3, 4, 5]
r_vals = np.linspace(2, 7, 10)


for n_pix in 2 ** np.array([7, 9, 11]):
    f_maxes = []
    overlap_maxes = []
    has_plotted = False
    for max_r in r_vals:
        lf.make_fiber_modes(npix=n_pix // 2, show_plots=False, max_r=max_r)
        optics = optics.set("psf_npixels", n_pix)
        optics = optics.set("psf_pixel_scale", max_r * core_diameter / n_pix)

        wf = prop_fibre_input_field(optics, input_f_number)

        if not has_plotted and max_r == r_vals[-1]:
            lf.plot_fiber_modes(0, fignum=50)

            plot_output_wf(wf)
            plt.pause(0.1)

            has_plotted = True

        # f_numbers = np.linspace(4.0, 5.0, 100)
        # overlaps = jax.vmap(compute_overlap_int)(f_numbers)

        # f_maxes.append(f_numbers[np.argmax(overlaps)])
        # overlap_maxes.append(np.max(overlaps))

        import scipy.optimize

        res = scipy.optimize.minimize(
            lambda x: -compute_overlap_int(x), 4.0, method="COBYLA"
        )

        f_maxes.append(res.x)
        overlap_maxes.append(-res.fun)

        
        if not has_plotted:
            wf = prop_fibre_input_field(optics, res.x)
            lf.plot_fiber_modes(0, fignum=50)

            plot_output_wf(wf)
            plt.pause(0.1)

            has_plotted = True

    plt.figure(100)
    plt.subplot(121)
    plt.plot(r_vals, f_maxes, "x", label=f"n_pix = {n_pix}")
    plt.xlabel("max_r (units of core)")
    plt.ylabel("optimal f number")

    plt.subplot(122)
    plt.plot(r_vals, overlap_maxes)
    plt.xlabel("max_r (units of core)")
    plt.ylabel("optimal injection value")


# plt.figure()
# plt.plot(f_numbers, overlaps)
# plt.xlabel("f number")
# plt.ylabel("Coupling efficiency")
plt.subplot(121)
plt.legend()

plt.show()
