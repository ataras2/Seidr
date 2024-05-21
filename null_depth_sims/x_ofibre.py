import ofiber
import jax.numpy as np
import matplotlib.pyplot as plt
import polarTransform

n_core = 1.44
n_cladding = 1.4345
wavelength = 1.5  # microns
core_radius = 32.8 / 2  # microns

max_l = 10
show_plots = True

NA = ofiber.numerical_aperture(n_core, n_cladding)
V = ofiber.V_parameter(core_radius, NA, wavelength)

allmodes_b = []
allmodes_l = []
allmodes_m = []
for l in range(max_l):
    cur_b = ofiber.LP_mode_values(V, l)
    if len(cur_b) == 0:
        break
    else:
        allmodes_b.extend(cur_b)
        ls = (np.ones_like(cur_b)) * l
        allmodes_l.extend(ls.astype(int))
        ms = np.arange(len(cur_b)) + 1
        allmodes_m.extend(ms)

allmodes_b = np.asarray(allmodes_b)
nLPmodes = len(allmodes_b)
# print('Total number of LP modes found: %d' % nLPmodes)
l = np.asarray(allmodes_l)
total_unique_modes = len(np.where(l == 0)[0]) + len(np.where(l > 0)[0]) * 2

print("Total number of unique modes found: %d" % total_unique_modes)
allmodes_b = allmodes_b
allmodes_l = allmodes_l
allmodes_m = allmodes_m
nLPmodes = nLPmodes


max_r = 2
npix = 50

r = np.linspace(0, max_r, npix)  # Radial positions, normalised so core_radius = 1

allmodefields_cos_polar = []
allmodefields_cos_cart = []
allmodefields_sin_polar = []
allmodefields_sin_cart = []
allmodefields_rsoftorder = []

array_size_microns = max_r * core_radius * 2
microns_per_pixel = array_size_microns / (npix * 2)

for mode_to_calc in range(nLPmodes):
    field_1d = ofiber.LP_radial_field(
        V, allmodes_b[mode_to_calc], allmodes_l[mode_to_calc], r
    )
    # TODO - LP02 comes out with core being pi phase and ring being 0 phase... investigate this.

    phivals = np.linspace(0, 2 * np.pi, npix)
    phi_cos = np.cos(allmodes_l[mode_to_calc] * phivals)
    phi_sin = np.sin(allmodes_l[mode_to_calc] * phivals)

    rgrid, phigrid = np.meshgrid(r, phivals)
    field_r_cos, field_phi = np.meshgrid(phi_cos, field_1d)
    field_r_sin, field_phi = np.meshgrid(phi_sin, field_1d)
    field_cos = field_r_cos * field_phi
    field_sin = field_r_sin * field_phi

    # Normalise each field so its total intensity is 1
    field_cos = field_cos / np.sqrt(np.sum(field_cos**2))
    field_sin = field_sin / np.sqrt(np.sum(field_sin**2))
    field_cos = np.nan_to_num(field_cos)
    field_sin = np.nan_to_num(field_sin)

    field_cos_cart, d = polarTransform.convertToCartesianImage(field_cos.T)
    field_sin_cart, d = polarTransform.convertToCartesianImage(field_sin.T)

    allmodefields_cos_polar.append(field_cos)
    allmodefields_cos_cart.append(field_cos_cart)
    allmodefields_sin_polar.append(field_sin)
    allmodefields_sin_cart.append(field_sin_cart)
    allmodefields_rsoftorder.append(field_cos_cart)
    if allmodes_l[mode_to_calc] > 0:
        allmodefields_rsoftorder.append(field_sin_cart)

    if show_plots:
        zlim = 0.1
        plt.figure(1)
        plt.clf()
        plt.subplot(121)
        sz = max_r * core_radius
        plt.imshow(
            allmodefields_cos_cart[mode_to_calc],
            extent=(-sz, sz, -sz, sz),
            cmap="bwr",
            vmin=-zlim,
            vmax=zlim,
        )
        plt.xlabel("Position ($\mu$m)")
        plt.ylabel("Position ($\mu$m)")
        plt.title(
            "Mode l=%d, m=%d (cos)"
            % (allmodes_l[mode_to_calc], allmodes_m[mode_to_calc])
        )
        core_circle = plt.Circle(
            (0, 0), core_radius, color="k", fill=False, linestyle="--", alpha=0.2
        )
        plt.gca().add_patch(core_circle)
        plt.subplot(122)
        sz = max_r * core_radius
        plt.imshow(
            allmodefields_sin_cart[mode_to_calc],
            extent=(-sz, sz, -sz, sz),
            cmap="bwr",
            vmin=-zlim,
            vmax=zlim,
        )
        plt.xlabel("Position ($\mu$m)")
        plt.title(
            "Mode l=%d, m=%d (sin)"
            % (allmodes_l[mode_to_calc], allmodes_m[mode_to_calc])
        )
        core_circle = plt.Circle(
            (0, 0), core_radius, color="k", fill=False, linestyle="--", alpha=0.2
        )
        plt.gca().add_patch(core_circle)
        plt.pause(0.001)
        print("LP mode %d, %d" % (allmodes_l[mode_to_calc], allmodes_m[mode_to_calc]))
        plt.pause(0.5)
