# read phase screen fits and see what they look like
# %%
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import os
import numpy as np
import cmocean
from lanternfiber import lanternfiber

# %%


def apply_complex_map(field, cmap, power=1.0, normalise=True, shift=0.0):
    """
    Maps a complex-valued 2D array to an RGB image using a colormap.

    The hue is determined by the phase (angle) of each complex value, and the brightness
    is scaled by the magnitude (absolute value) raised to the given power. The result is
    normalized to the [0, 1] range for display.

    Parameters
    ----------
    field : np.ndarray
        2D array of complex values to visualize.
    cmap : callable
        A matplotlib colormap function (e.g., plt.cm.hsv).
    power : float, optional
        Exponent to apply to the magnitude for brightness scaling/gamma (default is 1.0).

    Returns
    -------
    np.ndarray
        3D array representing the RGB image (shape: field.shape + (3,)).
    """
    angles = np.angle(field)
    angles_norm = np.mod(angles + shift, 2 * np.pi) / (2 * np.pi)

    if normalise:
        amp = np.abs(field) ** power / np.max(np.abs(field) ** power)
    else:
        amp = np.abs(field) ** power

    img = cmap(angles_norm) * (amp)[:, :, None]

    if normalise:
        img = img[..., :3] / np.max(img[..., 0:3])
    else:
        img = img[..., :3]

    return img


# %%
pth = os.path.join(
    *[
        "..",
        "data",
    ]
)
fn = "baldr_phase_AT_faint-poorATM.fits"
hdul = fits.open(os.path.join(pth, fn))
hdul.info()

# Read the data from the FITS file
data = hdul[6].data
print(data.shape)

# %%

# %%
import dLux
import dLux.utils as dlu

diameter = 1.8
wf_npixels = data.shape[1]
oversample = 5

coords = dlu.pixel_coords(wf_npixels * oversample, diameter)

# Generate outer aperture
primary = dlu.circle(coords, diameter / 2)

# Generate secondary mirror occultation
m2_diam = 0.14
secondary = dlu.circle(coords, m2_diam / 2, invert=True)

# Generate spiders
spider_width = 0.018
angles = [0, 80, 180, 260]
spiders = dlu.spider(coords, spider_width, angles)

# Combine and downsample
aperture = dlu.combine([primary, secondary, spiders], oversample)

plt.imshow(aperture, cmap="gray")

idx = 70
img = data[idx, :, :] - np.mean(data[idx, :, :])
# display phase but with nans where the aperture is 0
disp_img = np.where(aperture > 0, img, np.nan)
plt.imshow(disp_img, cmap="viridis")
plt.colorbar()

# %%
# Define the optical layers
# Note here we can pass in a tuple of (key, layer) pairs to be able to
# access the layer from the optics object with the key!
layers = [
    (
        "phase_screen",
        dLux.layers.AberratedLayer(
            # opd=img*1e-9,
            phase=img,
        ),
    ),
    (
        "aperture",
        dLux.layers.TransmissiveLayer(
            transmission=aperture,
        ),
    ),
]

psf_npixels = 256
psf_pixel_scale = 0.3  # um per pixel

# Construct the optics object
# optics = dLux.AngularOpticalSystem(
#     wf_npixels=wf_npixels,
#     diameter=diameter,
#     layers=layers,
#     psf_npixels=psf_npixels,
#     psf_pixel_scale=psf_pixel_scale,
# )

final_beam_diam = 4.8e-3
final_f_number = 25.4e-3 / final_beam_diam
effective_focal_length = final_f_number * diameter

optics = dLux.CartesianOpticalSystem(
    wf_npixels=wf_npixels,
    diameter=diameter,
    layers=layers,
    psf_npixels=psf_npixels,
    psf_pixel_scale=psf_pixel_scale,
    focal_length=effective_focal_length,
)


# Let examine the optics object! The dLux framework has in-built
# pretty-printing, so we can just print the object to see what it contains.
print(optics)

# %%
n_wavels = 5
wavels = np.linspace(1.5e-6, 1.85e-6, n_wavels)
source = dLux.PointSource(flux=1e5, wavelengths=wavels)
wf = optics.model(source, return_wf=True)

wavel_idx = 5
E = wf.amplitude[wavel_idx, :, :] * np.exp(1j * wf.phase[wavel_idx, :, :])
plt.imshow(apply_complex_map(E, cmocean.cm.phase, power=1.0), origin="lower")

# %%

n_core = 1.44
n_cladding = 1.4345
# n_core = 1.44
# n_cladding = n_core - 0.04
wavelength = 1.55  # microns
core_radius = 15.9 / 2  # microns
# core_radius = 8.2 / 2  # microns
# the max_r parameter specifies the extent of the
# mode field. This must be matched to the psf_pixel_scale
max_r = psf_pixel_scale * psf_npixels / (2 * core_radius)
# max_r = 1
lfs = []

for wavel in wavels * 1e6:
    lf = lanternfiber(
        n_core=n_core,
        n_cladding=n_cladding,
        core_radius=core_radius,
        wavelength=wavel,
    )

    lf.find_fiber_modes()
    lf.make_fiber_modes(npix=psf_npixels // 2, show_plots=False, max_r=max_r)

    lfs.append(lf)

# look at some of the fiber modes
len(lf.allmodefields_rsoftorder), lf.allmodefields_rsoftorder[0].shape

img_extent = psf_pixel_scale * psf_npixels

plt.imshow(
    # apply_complex_map(E, cmocean.cm.phase, power=1.0),
    np.abs(E) ** 2,
    origin="lower",
    extent=[-img_extent / 2, img_extent / 2, -img_extent / 2, img_extent / 2],
)
core_circle = plt.Circle(
    (0, 0),
    core_radius,
    color="white",
    fill=False,
)
plt.gca().add_artist(core_circle)

# %%
# lf = lfs[0]
mode_field_nums = list(range(len(lf.allmodefields_rsoftorder)))
lf.calc_injection_multi(E, mode_field_numbers=mode_field_nums)

# %%
# thinking about the chip now...
# need to be able to incoherently sum the intensities from each object in the field
