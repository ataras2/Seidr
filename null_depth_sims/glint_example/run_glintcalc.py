"""
Calculate signal to noise ratio, etc. for upgraded GLINT instrument.
Uses glintcalc.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from glintcalc import *
plt.ion()


##### Set a bunch of values #####
magnitude = 6
# magnitude = 12 #Pa beta

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
mag0flux = 1e10 #ph/um/s/m^2
star_flux = mag0flux * 2.5**-magnitude

wavelength = 1.6 # microns
bandwidth = 0.001 #0.05 # microns #TODO - Treat chromaticity properly (photonic)

# # Pa beta:
# wavelength = 1.2 # microns
# bandwidth = 0.001

# Collecting area:
pupil_fraction = 0.7
pupil_area = np.pi*4**2 * pupil_fraction
n_apertures = 4 # Number of apertures
pupil_diam = 2*(pupil_area/n_apertures / np.pi)**0.5

# Wavefront properties
r0 = 0.795 # Fried's parameter at ``wavelength'' before AO correction
order_zernike = 8 # Order of Zernike  up to which wavefront correction is done
n_actuator = 15 # Number of actuators across the pupil
wfe = 0.05 # RMS of the wavefront error in microns. Stands for a Strehl of 90% at 1.6 microns.
r0_corr = (1.03 * (wavelength/(2*np.pi*wfe))**2)**(3/5) * pupil_diam # r0 of the corrected wavefront given wfe

deltaphi_sig = get_diff_piston(wfe, n_actuator) # RMS wavefront across baseline in microns.
# deltaphi_sig = wfe

# Throughputs:
scexao_throughput = 0.2

injection_efficiency, delta_inj = get_injection(pupil_diam, r0, order_zernike, wl=wavelength, wfe=wfe) # if wl and wfe
                                                                # are define, they are used indtead of Noll's residuals
deltaI_sig = delta_inj * 2**0.5 # Injections in two apertures are assumed to be independent and with the same statistics

# Companion contrast
contrast = 1e-6
# contrast = 1e-4 #Pa beta

# Detector properties
read_noise = 0.5 # e-
QE = 0.8
int_time = 3600#3600 # seconds
# int_time = 3600 # Pa beta



print('Using deltaphi_sig = %f and deltaI_sig = %f' % (deltaphi_sig, deltaI_sig))

##### Do some calculations #####

g = glintcalc(wavelength = wavelength)
av_null = g.get_null_vals_MC(deltaphi_sig, deltaI_sig, num_samps=100000, show_plot=True)
print('Average null depth: %f' % av_null)

g.plot_null_dphi(deltaI_sig=deltaI_sig, max_dphi=0.05)

chrom_null = g.get_chromatic_null(deltaphi_sig, deltaI_sig, bandwidth=bandwidth, npoints = 50, show_plot=True)
print('Chromatic average null depth: %f' % chrom_null)

null_depth = chrom_null
null_depth = av_null # Simulate achromatic nuller
nulled_comp_snr = g.get_snr(photon_flux=star_flux, bandwidth=bandwidth, contrast=contrast, null_depth=null_depth,
                            throughput=scexao_throughput*injection_efficiency, pupil_area=pupil_area,
                            int_time=int_time, read_noise=read_noise, QE=QE, num_pix=1)
