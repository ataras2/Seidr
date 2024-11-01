import numpy as np
import knuller_sim as kn
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# -----------------------------------------------------
#              simulation demo parameters
# -----------------------------------------------------

npoints = 17  # number of hour angles
tindex = npoints // 2 + 1  # h.a. index @ transit
mycmap = cm.inferno  # color map
burst_mode = True  # script rhythm control

prms = 50  # rms of piston residuals
wavel = 1.6e-6  # wavelength


# Tau Boötis b:
# dec0 = kn.dec_deg(17, 27, 24.810)  # target declination
# dra, ddec, con = 3 / np.sqrt(2), 3 / np.sqrt(2), 10 ** (-3.632)  # test companion!
# magnitude = 3.55  # 22
# magnitude = 12 #Pa beta

# TOI-431 c
dec0 = kn.dec_deg(-26, 43, 26)  # target declination
dra, ddec, con = 1.59 / np.sqrt(2), 1.59 / np.sqrt(2), 10 ** (-5.1)  # test companion!
magnitude = 6.85  # 22

# Gaia18ajz:
# dec0 = kn.dec_deg(-8, 13, 12.756)  # target declination
# dra, ddec, con = 3 / np.sqrt(2), 3 / np.sqrt(2), 10 ** (-2.63)  # test companion!
# magnitude = 12.81


# suppose UTs
telescope_diameter = 8.2  # meters
secondary_diameter = 1.2  # meters
telescope_area = np.pi * (
    (telescope_diameter / 2) ** 2 - (secondary_diameter / 2) ** 2
)  # m^2
n_telescopes = 4

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
mag0flux = 1e10  # ph/um/s/m^2
star_flux = mag0flux * 2.5**-magnitude  # ph/um/s/m^2

wavelength = 1.6  # microns
bandwidth = 0.3  # 0.05 # microns #TODO - Treat chromaticity properly (photonic)


system_throughput = 0.039


read_noise = 0.5  # e-
spectral_R = 50
del_lambda = wavelength / spectral_R
n_pixels_per_null = np.ceil(bandwidth / del_lambda)
frame_int_time = 0.1  # seconds
print(f"n_pixels_per_null: {n_pixels_per_null}")
QE = 0.8
int_time = 3600 * 1  # 3600 # seconds

n_frames = int(int_time / frame_int_time)

# default companion parameters:
# dec0 = kn.dec_deg(-64, 42, 45)  # target declination
# dra, ddec, con = 5, 5, 1e-3  # test companion!
# prms = 50  # rms of piston residuals
# wavel = 3.6e-6  # wavelength


myk = kn.Nuller(wavel=wavel)  # default nuller is a 4T -> 3 output nuller
myk.update_observation(hawidth=4, npoints=npoints, combiner="kernel")

test_binary = myk.theoretical_signal_companion(dra=dra, ddec=ddec, con=con)
print(test_binary.shape)

myk.plot_projected_array_tracks()
# plt.show()

total_signal, on_axis, off_axis = myk.mc_perturbed_signal_companion(
    dra=dra, ddec=ddec, con=con, rms=prms, nmc=10000, return_split=True
)
print(total_signal.shape, on_axis.shape, off_axis.shape)

# SNR calculation
hr_index = 5
nuller_output_index = 1
kernel_index = 0

# the average null depth is just to do with how much of the star we suppress
# this should be done per telescope pair? per hour angle too
average_null_depth = np.mean(on_axis[nuller_output_index, hr_index, :])
print(f"Average null depth: {average_null_depth:.3e}")

read_noise_tot = read_noise * np.sqrt(n_pixels_per_null) * np.sqrt(n_frames)
print(f"Read noise total: {read_noise_tot:.3e}")

star_photons_per_scope = (
    star_flux * system_throughput * telescope_area * bandwidth * int_time * QE
)

# test_binary is the signal from the companion, in units of "fraction of telescope flux"
# TODO check this is right
companion_photons = star_photons_per_scope * test_binary
raw_comp_snr = companion_photons / np.sqrt(
    star_photons_per_scope * 2 + read_noise_tot**2
)


nulled_comp_snr = companion_photons / np.sqrt(
    star_photons_per_scope * 2 * average_null_depth + read_noise_tot**2
)

plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(raw_comp_snr)
# plt.colorbar()
# plt.title("Raw SNR")
# plt.ylabel("Nulled output index")
# plt.xlabel("Hour angle index")

# plt.subplot(1, 2, 2)
plt.imshow(nulled_comp_snr)
plt.colorbar()
plt.title("Nulled SNR")
plt.ylabel("Nulled output index")
plt.xlabel("Hour angle index")

print(f"Average SNR: {np.mean(raw_comp_snr):.3f} vs {np.mean(nulled_comp_snr):.3f}")

plt.figure()
fig, ax = plt.subplots()
for ii in range(test_binary.shape[0]):
    ax.plot(myk.har * 12 / np.pi, test_binary[ii], label="output #%d" % (ii + 1,))
ax.legend()
ax.set_xlabel("Hour angle")
ax.set_ylabel("Nuller output (fraction of telescope flux)")
mytitle = myk.combiner
mytitle += r" (%.1e companion @ (%+.1f, %+.1f))" % (con, dra, ddec)
ax.set_title(mytitle)
fig.set_tight_layout(True)


kernel = myk.kernel_signal(total_signal)
true_kernel = myk.kernel_signal(test_binary)

print(kernel.shape, true_kernel.shape)

plt.figure(figsize=(5, 2.5))
# plot histograms
plt.hist(
    total_signal[nuller_output_index, hr_index, :],
    bins=50,
    alpha=0.5,
    label="total signal",
)
plt.xlim(-0.01, 0.08)
plt.xlabel("Null depth (fraction of telescope flux)")
plt.ylabel("Number of simulations")
plt.title(f"Null depth distribution at piston rms {prms} nm")
# plt.axvline(
#     test_binary[nuller_output_index, hr_index], color="red", label="true binary signal"
# )
# plt.legend()


plt.figure(figsize=(12, 6))
plt.hist(kernel[kernel_index, hr_index, :], bins=50, alpha=0.5, label="kernel")
plt.axvline(true_kernel[kernel_index, hr_index], color="red", label="true kernel")

print(
    f"Median kernel: {np.median(kernel[kernel_index, hr_index, :])}, True kernel: {true_kernel[kernel_index, hr_index]}"
)
plt.legend()

plt.show()
