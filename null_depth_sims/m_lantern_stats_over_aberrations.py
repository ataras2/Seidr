"""
Observe the average coupling over a large number of simulations
"""

import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp

import lanternfiber

import matplotlib.pyplot as plt


def read_data(path, validate_lf=None, max_r=None):
    """
    read data from a path. If validate_lf is a lanterfiber object, check that the data is consistent with the object
    """
    data = np.load(path)

    if isinstance(validate_lf, lanternfiber.lanternfiber):
        onp.testing.assert_allclose(data["wavelength"] * 1e6, validate_lf.wavelength)
        onp.testing.assert_allclose(
            data["psf_pixel_scale"] * data["psf_npixels"],
            2 * validate_lf.core_radius * max_r,
        )

    return data


pth = "./aberrated_psfs_higher_f.npz"


# PL params
# n_core = 1.44
# n_cladding = 1.4345
n_core = 1.44
n_cladding = n_core - 0.04
wavelength = 1.65  # microns
core_radius = 5.0  # microns
max_r = 2


lf = lanternfiber.lanternfiber(
    n_core=n_core,
    n_cladding=n_cladding,
    core_radius=core_radius,
    wavelength=wavelength,
    # nmodes=n_modes,
)

data = read_data(pth, lf, max_r)


lf.find_fiber_modes()
lf.make_fiber_modes(npix=data["psf_npixels"] // 2, show_plots=False, max_r=max_r)

wavefronts = data["outputs"]

wf = wavefronts[0]
wf = data["non_aberrated"]


def get_total_injection(wf, lf):
    return lf.calc_injection_multi(
        input_field=wf,
        mode_field_numbers=list(range(len(lf.allmodefields_rsoftorder))),
        show_plots=False,
    )[0]


no_ab_inj = get_total_injection(data["non_aberrated"], lf)

injection_vals = jax.vmap(get_total_injection, in_axes=(0, None))(wavefronts, lf)

plt.figure()
plt.hist(
    injection_vals,
    alpha=0.5,
    label=f"Aberrated injection (rms = {data['zernike_rms']})",
)
plt.axvline(no_ab_inj, c="k", linestyle="--", label="Non-aberrated injection")
plt.xlabel("Injection efficiency")
plt.ylabel("Number of simulations")
plt.legend()


def get_injections(wf, lf):
    return lf.calc_injection_multi(
        input_field=wf,
        mode_field_numbers=list(range(len(lf.allmodefields_rsoftorder))),
        show_plots=False,
    )[1]


no_ab_injs = get_injections(data["non_aberrated"], lf)
injections = jax.vmap(get_injections, in_axes=(0, None))(wavefronts, lf)


from prettytable import PrettyTable

table = PrettyTable()

table.field_names = ["Modes", "5th", "50th", "95th", "no ab"]


def summary_stats(injs):
    return np.round(np.percentile(injs, np.array([5, 50, 95])), 3)


plt.figure()
first_n_modes_vals = [1, 3, 5]
for i, n in enumerate(first_n_modes_vals):
    injs = injections[:, :n].sum(axis=1)
    plt.hist(
        injs,
        alpha=0.5,
        color=f"C{i}",
        label=f"First {n} modes",
    )
    plt.axvline(no_ab_injs[:n].sum(), c=f"C{i}", linestyle="--")

    table.add_row([f"{1}-{n}", *summary_stats(injs), no_ab_injs[:n].sum()])

i += 1
central_with_tip_tilt = np.array([0, 3, 4])
injs = injections[:, central_with_tip_tilt].sum(axis=1)
plt.hist(
    injs,
    alpha=0.5,
    color=f"C{i}",
    label=f"Modes {central_with_tip_tilt}",
)
plt.axvline(no_ab_injs[central_with_tip_tilt].sum(), c=f"C{i}", linestyle="--")

table.add_row(
    [
        f"{central_with_tip_tilt}",
        *summary_stats(injs),
        no_ab_injs[central_with_tip_tilt].sum(),
    ]
)

plt.legend()
plt.xlabel("Injection efficiency")
plt.ylabel("Number of simulations")

print(table)

plt.show()
