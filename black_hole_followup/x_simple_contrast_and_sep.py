import numpy as np
import astropy.units as u
import astropy.constants as const

event_year = 2018 * u.year
observing_year = 2027 * u.year

motion = 5 * u.mas / u.yr
star_mass = 1.0 * u.M_sun
star_temperature = 10_000 * u.K
star_distance = 8 * u.kpc

background_star_H_mag = 12.81 * u.mag

density = 1e9 * u.kg / u.m**3
radius = (3 * star_mass / (4 * np.pi * density)) ** (1 / 3)

luminosity = 4 * np.pi * radius**2 * const.sigma_sb * star_temperature**4
print(f"Luminosity: {luminosity.to(u.W)}")

flux = luminosity / (4 * np.pi * star_distance**2)
print(f"Flux: {flux.to(u.W / u.m**2)}")

# background_star_H_flux = 10 ** (-0.4 * background_star_H_mag) * u.W / u.m**2

# print(f"Background Star H Flux: {background_star_H_flux}")


angular_seperation = (motion * (observing_year - event_year)).to(u.mas)
print(f"Angular Seperation: {angular_seperation}")

wavel = 1.6 * u.micron
longest_baseline = 130 * u.m

resolution_limit = np.arctan(wavel / (2 * longest_baseline))
print(
    f"Bound on mu >  {(resolution_limit/(observing_year-event_year)).to(u.mas/u.yr) }"
)
