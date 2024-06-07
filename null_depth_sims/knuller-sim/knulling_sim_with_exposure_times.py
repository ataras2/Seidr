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

prms = 100  # rms of piston residuals
wavel = 1.6e-6  # wavelength


# suppose UTs
telescope_diameter = 8.2  # meters
telescope_area = np.pi * (telescope_diameter / 2) ** 2  # m^2
n_telescopes = 4
magnitude = 3.55  # 22
# magnitude = 12 #Pa beta

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
mag0flux = 1e10  # ph/um/s/m^2
star_flux = mag0flux * 2.5**-magnitude  # ph/um/s/m^2

wavelength = 1.6  # microns
bandwidth = 0.3  # 0.05 # microns #TODO - Treat chromaticity properly (photonic)


system_throughput = 0.1


read_noise = 0.5  # e-
QE = 0.8
int_time = 3600 * 10  # 3600 # seconds

# default companion parameters:
# dec0 = kn.dec_deg(-64, 42, 45)  # target declination
# dra, ddec, con = 5, 5, 1e-3  # test companion!
# prms = 50  # rms of piston residuals
# wavel = 3.6e-6  # wavelength


# Tau Boötis b:
dec0 = kn.dec_deg(17, 27, 24.810)  # target declination
dra, ddec, con = 3 / np.sqrt(2), 3 / np.sqrt(2), 10 ** (-3.632)  # test companion!


myk = kn.Nuller(wavel=wavel)  # default nuller is a 4T -> 3 output nuller
myk.update_observation(hawidth=4, npoints=npoints, combiner="kernel")

test_binary = myk.theoretical_signal_companion(dra=dra, ddec=ddec, con=con)
print(test_binary.shape)

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

star_photons = (
    star_flux * system_throughput * telescope_area * bandwidth * int_time * n_telescopes
)

# test_binary is the signal from the companion, in units of "fraction of telescope flux"
# TODO check this is right
companion_photons = star_photons * test_binary[nuller_output_index, hr_index]
raw_comp_snr = companion_photons / np.sqrt(star_photons)
print("No-nulling S/N ratio for companion: %f" % raw_comp_snr)

nulled_comp_snr = companion_photons / np.sqrt(star_photons * average_null_depth)
print("Nulled S/N ratio for companion: %f" % nulled_comp_snr)

exit()

kernel = myk.kernel_signal(total_signal)
true_kernel = myk.kernel_signal(test_binary)

print(kernel.shape, true_kernel.shape)

plt.figure(figsize=(12, 6))
# plot histograms
plt.hist(on_axis[nuller_output_index, hr_index, :], bins=50, alpha=0.5, label="on-axis")
plt.hist(
    off_axis[nuller_output_index, hr_index, :], bins=50, alpha=0.5, label="off-axis"
)
plt.hist(
    total_signal[nuller_output_index, hr_index, :],
    bins=50,
    alpha=0.5,
    label="total signal",
)
plt.axvline(
    test_binary[nuller_output_index, hr_index], color="red", label="true binary signal"
)
plt.legend()


plt.figure(figsize=(12, 6))
plt.hist(kernel[kernel_index, hr_index, :], bins=50, alpha=0.5, label="kernel")
plt.axvline(true_kernel[kernel_index, hr_index], color="red", label="true kernel")

print(
    f"Median kernel: {np.median(kernel[kernel_index, hr_index, :])}, True kernel: {true_kernel[kernel_index, hr_index]}"
)
plt.legend()

plt.show()
