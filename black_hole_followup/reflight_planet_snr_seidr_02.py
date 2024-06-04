import numpy as np
import matplotlib.pyplot as plt

plt.ion()


null_depth = 1e-3
star_distance_pc = 10
albedo = 0.34  # Jupiter has a Bond albedo of 0.34
# albedo = 0.1
phase_angle_deg = 90
# titletext = 'SNR for 1 hr integration, 2*UTs, 1$R_{Jup}$ planet'
titletext = (
    "SNR in 1 hr, 1$R_{Jup}$ planet, %.0e null depth, %.2f albedo, %d$^\circ$ phase angle"
    % (null_depth, albedo, phase_angle_deg)
)


def get_contr_sepn(albedo=0.34):
    prad = 7.149e7  # 1 R_J in metres
    au = 1.496e11  # 1 au in metres
    # albedo = 0.34  # Jupiter has a Bond albedo of 0.34
    startsepn = 1e-3
    endsepn = 1.5  # in au
    npoints = 1000

    r_au = np.linspace(startsepn, endsepn, npoints)
    r = r_au * au

    planetArea = np.pi * prad**2
    lightShellArea = 4 * np.pi * r**2
    lightReflected = planetArea / lightShellArea * albedo
    # np.savez('contr_dist10_seidr.npz', r_au=r_au, contrast=lightReflected)
    return r_au, lightReflected


def get_photnoise_contrast(magnitude, contrast, verbose=True, null_depth=1):
    # wavelength = 1.6  # microns
    bandwidth = 0.35  # microns
    pup_diam = 8  # m
    n_tels = 2

    framerate = 1000  # Hz
    read_noise = 0.4  # e-
    dark_noise = 0.03 * 3500  # e-/px/s
    QE = 0.8
    int_time = 3600  # s
    num_outfibs = 50
    num_specchans = 30  # R~100 over H band
    num_pix = num_outfibs * num_specchans

    # Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
    # e.g. An H=5 object gives 1 ph/cm^2/s/A
    mag0flux = 1e10  # ph/um/s/m^2
    star_flux = mag0flux * 2.5**-magnitude

    # Collecting area:
    pupil_fraction = 0.8
    pupil_area = np.pi * (pup_diam / 2) ** 2 * pupil_fraction * n_tels
    injection_efficiency = 0.8
    instr_throughput = 0.5

    refl_frac = refl_frac_phase(phase_angle_deg)

    n_ints = int_time * framerate
    dark_noise_photons = (dark_noise * int_time) / QE
    read_noise_tot = read_noise * np.sqrt(num_pix) * np.sqrt(n_ints)
    read_noise_photons = read_noise_tot / QE
    star_photons = (
        star_flux
        * injection_efficiency
        * instr_throughput
        * pupil_area
        * bandwidth
        * int_time
        * refl_frac
    )
    star_snr = star_photons / np.sqrt(read_noise_photons**2 + dark_noise_photons**2)

    planet_flux = star_photons * contrast
    planet_snr = planet_flux / np.sqrt(
        star_photons + read_noise_photons**2 + dark_noise_photons**2
    )
    nulled_planet_snr = planet_flux / np.sqrt(
        star_photons * null_depth + read_noise_photons**2 + dark_noise_photons**2
    )

    if verbose:
        print("Stellar photons: %.3g" % star_photons)
        print("S/N ratio for star measurement: %f" % star_snr)
        print("No-nulling S/N ratio for planet: %f" % planet_snr)
        print("Nulled S/N ratio for planet: %f" % nulled_planet_snr)

    return nulled_planet_snr


def refl_frac_phase(phase_angle_deg):
    phase_rad = phase_angle_deg / 180 * np.pi
    frac = 1 / np.pi * (np.sin(phase_rad) + (np.pi - phase_rad) * np.cos(phase_rad))
    return frac


r_au, lightReflected = get_contr_sepn(albedo)
r_au = np.flip(r_au)
contr_dist = np.flip(np.log10(lightReflected))

nsamps_mags = 6
nsamps_contrs = 100
mag_contr_snr = np.zeros((nsamps_mags, nsamps_contrs))
maglims = (14, 4)
log_contrlims = (-6, -2)
mags = np.linspace(maglims[0], maglims[1], nsamps_mags)
log_contrs = np.linspace(log_contrlims[0], log_contrlims[1], nsamps_contrs)
r_au_sameinds = np.interp(log_contrs, contr_dist, r_au)
sep_mas = r_au_sameinds / star_distance_pc * 1000

mag_contr_snr_null = np.zeros((nsamps_mags, nsamps_contrs))
for k in range(nsamps_mags):
    for l in range(nsamps_contrs):
        magnitude = mags[k]
        contrast = 10 ** log_contrs[l]
        mag_contr_snr_null[k, l] = get_photnoise_contrast(
            magnitude, contrast, verbose=False, null_depth=null_depth
        )
snrdata = np.copy(mag_contr_snr.T)
snrdata_null = np.copy(mag_contr_snr_null.T)

plt.figure(1)
plt.clf()
plt.gca().set_prop_cycle(None)
plt.plot(log_contrs, snrdata_null, "-")
plt.gca().set_prop_cycle(None)
plt.plot(log_contrs, snrdata, ":")
plt.xlabel("Log planet contrast")
plt.ylabel("SNR")
plt.legend(mags, title="Star H mag", loc="upper left")
plt.ylim([0, 10])
plt.xlim([np.min(log_contrs), np.max(log_contrs)])

ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.plot(sep_mas, log_contrs, "k")
ax2.set_xscale("log")
ax2.invert_xaxis()
plt.xlim([np.max(sep_mas), np.min(sep_mas)])
plt.xlabel("Star-planet separation (mas at %.0f pc)" % star_distance_pc)
plt.title(titletext)
plt.tight_layout()
# plt.show()

plt.savefig("figout.png", dpi=150)
