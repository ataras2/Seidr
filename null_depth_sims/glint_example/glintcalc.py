"""
Methods to use for the GLINT signal calculator
Run from run_glintcalc.py
"""
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

# =============================================================================
# Class section
# =============================================================================
class glintcalc:
    def __init__(self, wavelength=1.6):
        self.nullsamps = None
        self.wavelength = wavelength


    def get_null_vals_MC(self, deltaphi_sig, deltaI_sig, deltaphi_mu=0, deltaI_mu = 0,
                         num_samps=100000, show_plot=False, hist_bins=100):
        # Let total flux I1+I2 = 1
        # Assume N+ >> N-, deltaphi <<1 and deltaI << 1, so can approximate:
        # (See Hanot+ 2011, Norris+ 2020)
        wavelength = self.wavelength
        deltaphi_sig_rad = deltaphi_sig/wavelength * 2 * np.pi
        deltaphi_mu_rad = deltaphi_mu/wavelength * 2 * np.pi
        dIsamps = randn(num_samps) * deltaI_sig + deltaI_mu
        dphisamps = randn(num_samps) * deltaphi_sig_rad + deltaphi_mu_rad
        self.nullsamps = 0.25 * (dIsamps**2 + dphisamps**2)
        self.av_null = np.mean(self.nullsamps)

        if show_plot:
            plt.figure(1)
            plt.clf()
            plt.hist(self.nullsamps, hist_bins, density=True)
            plt.xlabel('Null depth')
            plt.ylabel('Frequency')

        return self.av_null


    def plot_null_dphi(self, deltaI_sig, max_dphi=None, npoints=100):
        if max_dphi is None:
            max_dphi = self.wavelength/10
        dphis = np.linspace(0, max_dphi, npoints)
        all_nulls = np.zeros(npoints)
        for k in range(npoints):
            all_nulls[k] = self.get_null_vals_MC(dphis[k], deltaI_sig)
        plt.figure(2)
        plt.clf()
        plt.plot(dphis,all_nulls)
        plt.xlabel('dphi sigma (microns)')
        plt.ylabel('Average null')


    def get_chromatic_null(self, deltaphi_sig, deltaI_sig, bandwidth, npoints = 50, show_plot=False):
        # Make the assumption that null is purely chromatic, i.e. behaves like free space optics
        all_wl_offsets = np.linspace(-bandwidth/2, bandwidth/2, npoints)

        all_nulls = np.zeros(npoints)
        for k in range(npoints):
            all_nulls[k] = self.get_null_vals_MC(deltaphi_sig, deltaI_sig, deltaphi_mu=all_wl_offsets[k],
                                                 num_samps=1000000)
        chromatic_null = np.mean(all_nulls)

        if show_plot:
            plt.figure(3)
            plt.clf()
            plt.plot(all_wl_offsets+self.wavelength, all_nulls)
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Average null depth')
            plt.tight_layout()

        return chromatic_null


    def get_snr(self, photon_flux, bandwidth, contrast, null_depth, throughput=1, pupil_area=50,
                             int_time=1, read_noise=1, n_ints=1, QE=1, num_pix=1):
        # photon_flux is in ph/um/s/m^2
        # pupil_area in m^2
        # bandwidth in microns
        # int_time in seconds

        read_noise_tot = read_noise * np.sqrt(num_pix) * np.sqrt(n_ints)
        #TODO is this read-noise scaling right? No, read_noise should be outside the sqrt unless it stands for the variance

        star_photons = photon_flux * throughput * pupil_area * bandwidth * int_time
        print("Stellar photons: %.3g" % star_photons)
        star_snr = (star_photons*QE) / read_noise_tot
        print('S/N ratio for star measurement: %f' % star_snr)

        companion_flux = star_photons * contrast
        raw_comp_snr = companion_flux / np.sqrt(star_photons + read_noise_tot**2)
        print('No-nulling S/N ratio for companion: %f' % raw_comp_snr)

        nulled_comp_snr = companion_flux / np.sqrt(star_photons*null_depth + read_noise_tot**2)
        print('Nulled S/N ratio for companion: %f' % nulled_comp_snr)

        return nulled_comp_snr
   
# =============================================================================
# Functions section
# =============================================================================
def get_noll_residuals(diam, r0, order=1):
    """
    Get the mean square residual phase error of the wavefront after correction of the first N orders of the Zernike.
    Source: https://ui.adsabs.harvard.edu/abs/1976JOSA...66..207N/abstract
    
    :Parameters:
    
        **diam**: diameter of the aperture
        
        **r0** : Fried's parameter at the considered wavelgnth
    
        **order** : order of the Zernike polynom up to which the correction of the wavefront is done.
    
    
    :Returns:
    
        mean square residual error sigma**2 of the wavefront after correction
    """
    
    # Array of coefficients of residuals for the first 21 order of Zernike
    coeffs = np.array([1.0299, 0.582, 0.134, 0.111, 0.0880, 0.0648, 0.0587, 
                       0.0525, 0.0463, 0.0401, 0.0377, 0.0352, 0.0328, 0.0304, 
                       0.0279, 0.0267, 0.0255, 0.0243, 0.0232, 0.022, 0.0208])
    
    if order <= 21:
        return coeffs[order-1] * (diam/r0)**(5/3.) # In rad^2
    else:
        return 0.2944 * order**(-3**0.5/2) * (diam/r0)**(5/3.) # In rad^2
    
def get_injection(diam, r0, order=1, geo_inj=0.8, wl=None, wfe=None):
    """
    Compute the average injection into a single-mode fiber/photonics.
    It uses the Marechal approximation to convert the standard deviation of the wavefront residuals into
    a Strehl ratio which is used to calculate the injection efficiency
    (https://ui.adsabs.harvard.edu/abs/2000A%26AS..145..305C/abstract)

    :Parameters:
    
        **diam** : float
            Diameter of the aperture.
        **r0** : float
            Friend's parameter, wavelength-dependent.
        **order** : int, optional
            Maximum order of correction of the Zernike polynom. Unused if both **wl** and **wfe** are not ``None''.
        **geo_inj** : float, optional
            Maximum injection allowed by the geometry of the pupil. The default is 0.8 for a plain pupil without spider
             (https://ui.adsabs.harvard.edu/abs/1988ApOpt..27.2334S/abstract)
        **wl** : float, optional
            Wavelength in microns. If not ``None'' (with ``wfe''), bypass ``get_noll_residuals'' and compute the mean
            square of the phase residuals using **wfe**
        **wfe** : float, optional
            RMS of the wavefront error in micron. If not ``None'' (with ``wl''), bypass ``get_noll_residuals'' and
            compute the mean square of the phase residuals using **wfe**
    
    :Returns:
    
        Injection efficiency.

    """
    if not wl is None and not wfe is  None:
        residuals = (2 * np.pi / wl * wfe)**2
    else:
        residuals = get_noll_residuals(diam, r0, order) # Get variance of the phase residuals after correction of the first N order of Zernike.
    strehl = np.exp(-residuals) # Marechal's approximation
    
    low_sr = 1 - 2 * (1 - strehl) # Lower bound of strehl. Source: Olivier Guyon's mail 2020-09-23 4:17 am
    low_sr = max(low_sr, 0) # Previous relationship can give negative strehl, set a threshold at 0
    total_range = 1 - low_sr
    std_sr = total_range / 6 # We assume a Normal distribution of the injection which is what we observe
    
    inj =  geo_inj * strehl
    std_inj = geo_inj * std_sr
    
    return inj, std_inj
    
    
def get_diff_piston2(diam, r0, wl, fromTT=False, baseline=5.55):
    """
    Get the RMS of the differential piston between 2 apertures.
    Source: https://ui.adsabs.harvard.edu/abs/1996ApOpt..35.3002K/abstract
    It is simply Delta_1 (1st order of Noll's residuals ie the piston) summed with itself, converted into OPD and take the square root.
    However, it seems it is quite over-estimated.
    And in the case of NRM-nulling, the shift of the fringes is mainly due to the tip-tilt.
    
    Another strategy is to estimate the TT and deduce the corresponding shift of the fringes so that piston = baseline * angle of TT
    WARNING: this method does not take into account dedicated TT correction device.
    
    NOTE: the two returns have different units! If fromTT is True, the return is in the same unit as **baseline**.
    Else, it is in the same unit as **wl**.
    
    :Parameters:
    
        **diam**: diameter of the aperture, in meter
        
        **r0** : Fried's parameter at the considered wavelength, in meter
    
        **wl** : wavelength of observation.
                
        **fromTT**: bool, optional
            If ``True'', get the differential piston given the variance of the TT and the baseline
    
        **baseline**: float, optional
            Lenght of the baseline, in meter. Used if **fromTT** is ``True''.
    
    :Returns:
    
        Root mean square of the differential piston.
    """
    
    if fromTT:
        varTT = 6.88/(2*np.pi**2) * (wl*1e-6)**2 * diam**(-1/3.) * r0**(-5/3.) # Variance of the TT in one direction. Source: Tyson, R.K. Principle of adaptive optics. Edn. 3rd, CRCpress, Boca Raton, 2011
        return baseline * varTT**0.5
    else:
        piston = 0.228 * wl * (diam / r0)**(5/6)
        return piston

def get_diff_piston(wfe, n_actuator):
    """
    Get the RMS of the differential piston between 2 apertures.
    
    :Parameters:
    
        **wfe**: RMS of the wavefront
        
        **n_actuator** : number of actuator across the pupil
    
    :Returns:
    
        RMS of the differential piston in the same unit as **wfe**.
    """
    
    ntot_actuators = np.pi * n_actuator**2 / 4
    piston_rms = wfe / ntot_actuators**0.5
    print('Piston rms: %f' % piston_rms)
    return piston_rms * 2**0.5 # Std diff piston = sqrt(std piston 1 + std piston 2), with 1 and 2 the pupils
