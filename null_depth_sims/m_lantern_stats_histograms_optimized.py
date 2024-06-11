"""
Given a fibre face, do the following:

1. compute the modes possible and print out how many modes there are
2. dlux sim to optimize the f number for coupling into the first n modes (given n)
3. run over N=1000 random aberrations and compute the coupling efficiency in the first n modes

"""

import dLux as dl
import dLux.utils as dlu

import jax
import jax.numpy as np
import jax.random as jr

import numpy as onp

import lanternfiber

import matplotlib.pyplot as plt

from prettytable import PrettyTable

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 72

wavel = 1.63  # microns

n_pix = 512  # full width

n_core = 1.44
n_cladding = 1.4345

# SMF-28
# core_diameter = 8.2  # microns

# 3 modes
# core_diameter = 13.0  # microns

# 5 modes
# core_diameter = 15.9  # microns

# 5 modes
# core_diameter = 24  # microns

max_r = 6

table = PrettyTable()
table.field_names = ["core diameter (um)", "n_modes", "no ab.", "5", "50", "95", "5-95"]

for i, core_diameter in enumerate([8.2, 13.0, 15.9]):
    print(f"Core diameter: {core_diameter} microns")
    # part 1
    lf = lanternfiber.lanternfiber(
        n_core=n_core,
        n_cladding=n_cladding,
        core_radius=core_diameter / 2,
        wavelength=wavel,
    )

    lf.find_fiber_modes()
    lf.make_fiber_modes(npix=n_pix // 2, show_plots=False, max_r=max_r)

    print(f"Number of modes: {lf.nmodes}")

    # part 2
    # Wavefront properties
    diameter = 1.8
    wf_npixels = 512

    # psf params
    input_f_number = 1.25
    focal_length = input_f_number * diameter
    psf_npixels = n_pix
    psf_pixel_scale = max_r * core_diameter / psf_npixels

    n_zernikes = 100  # including piston
    tip_tilt_rms = (
        200e-9 / 4 / np.sqrt(2)
    )  # /4 because of dlux things, /sqrt(2) to acocunt for both tip/tilt
    rest_rms = 20e-9 / 4

    coords = dlu.pixel_coords(wf_npixels, diameter)
    circle = dlu.circle(coords, diameter / 2)

    # Zernike aberrations
    zernike_indexes = np.arange(1, n_zernikes + 1)
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

    def prop_fibre_input_field(optics, zernikies):
        optics = optics.set("aperture.coefficients", zernikies)
        output = source.model(optics, return_wf=True)
        ouput_wf_complex = (
            (output.amplitude * np.exp(1j * output.phase))
            * source.spectrum.weights[:, None, None]
        ).sum(axis=0)

        return ouput_wf_complex

    def compute_overlap_int(optics, f_number):
        optics = optics.set("focal_length", f_number * diameter)
        ouput_wf_complex = prop_fibre_input_field(
            optics, np.zeros(zernike_indexes.shape)
        )
        return lf.calc_injection_multi(
            input_field=ouput_wf_complex,
            mode_field_numbers=list(range(len(lf.allmodefields_rsoftorder))),
            show_plots=False,
            return_abspower=True,
        )[0]

    import scipy.optimize

    res = scipy.optimize.minimize(
        lambda x: -compute_overlap_int(optics, x), 4.0, method="COBYLA"
    )

    optimal_fnumber = res.x
    optics = optics.set("focal_length", optimal_fnumber * diameter)

    ouput_wf_complex = prop_fibre_input_field(optics, coeffs)

    print(f"Optimal f-number: {optimal_fnumber} had overlap of {-res.fun}")

    # part 3
    N = 1000

    zernike_coeffs = np.concatenate(
        [
            np.zeros((N, 1)),
            tip_tilt_rms * jr.normal(jr.PRNGKey(1), (N, 2)),
            rest_rms * jr.normal(jr.PRNGKey(1), (N, n_zernikes - 3)),
        ],
        axis=1,
    )

    print("Making PSFs...", end="")
    outputs = jax.vmap(prop_fibre_input_field, in_axes=(None, 0))(
        optics, zernike_coeffs
    )
    print("done")

    def get_total_injection(wf, lf):
        return lf.calc_injection_multi(
            input_field=wf,
            mode_field_numbers=list(range(len(lf.allmodefields_rsoftorder))),
            show_plots=False,
            return_abspower=True,
        )[0]

    no_ab_inj = get_total_injection(ouput_wf_complex, lf)

    injection_vals = jax.vmap(get_total_injection, in_axes=(0, None))(outputs, lf)

    plt.figure(1)
    plt.hist(
        injection_vals,
        alpha=0.5,
        label=f"{core_diameter} $\mu$m",  # *1e9 to convert to nm, *sqrt(2)*4 to convert to RMS/dlux things
    )
    plt.axvline(
        no_ab_inj, c=f"C{i}", linestyle="--", label=f"{core_diameter} $\mu$m, no ab."
    )

    table.add_row(
        [
            core_diameter,
            lf.nmodes,
            no_ab_inj,
            np.percentile(injection_vals, 5),
            np.percentile(injection_vals, 50),
            np.percentile(injection_vals, 95),
            np.percentile(injection_vals, 95) - np.percentile(injection_vals, 5),
        ]
    )


plt.xlabel("Injection efficiency (fraction of aperture power)")
plt.ylabel("Number of simulations")
plt.legend()

print(table)

plt.show()
