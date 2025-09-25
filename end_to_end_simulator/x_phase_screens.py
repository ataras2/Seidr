# read phase screen fits and see what they look like
# %%
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import os
import numpy as onp
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
    angles = onp.angle(field)
    angles_norm = onp.mod(angles + shift, 2 * onp.pi) / (2 * onp.pi)

    if normalise:
        amp = onp.abs(field) ** power / onp.max(onp.abs(field) ** power)
    else:
        amp = onp.abs(field) ** power

    img = cmap(angles_norm) * (amp)[:, :, None]

    if normalise:
        img = img[..., :3] / onp.max(img[..., 0:3])
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
img = data[idx, :, :] - onp.mean(data[idx, :, :])
# display phase but with nans where the aperture is 0
disp_img = onp.where(aperture > 0, img, onp.nan)
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
wavels = onp.linspace(1.5e-6, 1.85e-6, n_wavels)
source = dLux.PointSource(flux=1e5, wavelengths=wavels)
wf = optics.model(source, return_wf=True)

wavel_idx = 5
E = wf.amplitude[wavel_idx, :, :] * onp.exp(1j * wf.phase[wavel_idx, :, :])
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
    onp.abs(E) ** 2,
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
lf.calc_injection_multi(E, mode_field_numbers=mode_field_nums, return_abspower=True)


# %%
import jax.numpy as np


class TscopeInjection:
    def __init__(
        self,
        core_radius,
        phase_screen_fname,
        tscope="AT",
        n_wavels=5,
        wavel_bnds=(1.5e-6, 1.85e-6),
    ):
        self.core_radius = core_radius
        self.phase_screen_fname = phase_screen_fname
        self.tscope = tscope

        self.phase_screens = self._load_phase_screen()
        self.wf_npixels = self.phase_screens.shape[1]

        if tscope == "AT":
            self.diameter = 1.8  # m
        elif tscope == "UT":
            raise NotImplementedError("UT not implemented yet")
        else:
            raise ValueError("Invalid telescope type")

        self.aperture = self._create_aperture(self.diameter, self.wf_npixels)

        self.psf_npixels = 256
        self.psf_pixel_scale = 0.3  # um per pixel
        self.wavels = onp.linspace(wavel_bnds[0], wavel_bnds[1], n_wavels)
        self.lfs = self._create_lfs(self.psf_pixel_scale, self.psf_npixels, self.wavels)

        # find how many modes are supported at most
        self.max_n_modes = max([len(lf.allmodefields_rsoftorder) for lf in self.lfs])

        # create dlux object
        final_beam_diam = 4.8e-3
        final_f_number = 25.4e-3 / final_beam_diam
        effective_focal_length = final_f_number * diameter

        layers = [
            (
                "phase_screen",
                dLux.layers.AberratedLayer(
                    phase=self.phase_screens[
                        0, :, :
                    ],  # TODO: check units of phase screen (maybe should be OPD)
                ),
            ),
            (
                "aperture",
                dLux.layers.TransmissiveLayer(
                    transmission=self.aperture,
                ),
            ),
        ]

        self.optics = dLux.CartesianOpticalSystem(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            focal_length=effective_focal_length,
        )

    @staticmethod
    def _create_aperture(diameter, wf_npixels, oversample=5):
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
        return aperture

    @staticmethod
    def _create_lfs(
        psf_pixel_scale, psf_npixels, wavels, n_core=1.44, n_cladding=1.4345
    ):
        lfs = []
        max_r = psf_pixel_scale * psf_npixels / (2 * core_radius)

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
        return lfs

    def _load_phase_screen(self):
        pth = os.path.join(
            *[
                "..",
                "data",
            ]
        )
        hdul = fits.open(os.path.join(pth, self.phase_screen_fname))
        # Read the data from the FITS file
        data = hdul[1].data
        print(f"Phase screen data shape: {data.shape}")
        return np.array(data)

    def run_to_injection(self, screen_idx=None):
        source = dLux.PointSource(flux=1e5, wavelengths=self.wavels)

        if screen_idx is None:
            # use all screens in sequence
            screen_idx = list(range(self.phase_screens.shape[0]))
        else:
            # use a single screen index
            if not isinstance(screen_idx, list):
                screen_idx = [screen_idx]

        injections = []
        for i, si in enumerate(screen_idx):
            self.optics = self.optics.set(
                "phase_screen.phase", self.phase_screens[si, :, :]
            )
            wf = self.optics.model(source, return_wf=True)

            injections.append([])
            for wavel_idx, wavel in enumerate(self.wavels):
                E = wf.phasor[wavel_idx, :, :]
                lf = self.lfs[wavel_idx]

                mode_field_nums = list(range(len(lf.allmodefields_rsoftorder)))
                inj = lf.calc_injection_multi(E, mode_field_numbers=mode_field_nums)

                inj_breakdown = inj[1]
                # zero pad to max_n_modes
                if len(inj_breakdown) < self.max_n_modes:
                    inj_breakdown = np.pad(
                        inj_breakdown,
                        (0, self.max_n_modes - len(inj_breakdown)),
                        mode="constant",
                        constant_values=0,
                    )
                injections[i].append(inj_breakdown)

        injections = np.array(injections)
        return injections


t = TscopeInjection(
    core_radius=core_radius,
    phase_screen_fname="baldr_phase_AT_faint-poorATM.fits",
    tscope="AT",
    n_wavels=5,
    wavel_bnds=(1.5e-6, 1.85e-6),
)
# %%
injs = t.run_to_injection(screen_idx=list(range(100)))

# %%
injs.shape # (n_screens, n_wavels, n_modes)

# plot injection into mode 0 as a function of screen with wavelength as the color of the curve (viridis scale)
plt.figure(figsize=(10, 6))
for wavel_idx in range(injs.shape[1]):
    color = plt.cm.viridis(wavel_idx / injs.shape[1])
    plt.plot(injs[:, wavel_idx, 0], color=color, label=f"Wavel {wavel_idx+1}")
plt.xlabel("Phase Screen Index")
plt.ylabel("Injection into Mode 0")
plt.title("Injection into Mode 0 as a Function of Phase Screen Index")
# plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), label="Wavelength")
plt.legend()
plt.show()

# %%
# thinking about the chip now...
# need to be able to incoherently sum the intensities from each object in the field

# %%
