"""
updated version to deal with uncertainty too
"""

import astropy.units as u
import astropy.constants as c
import numpy as np


def spectral_energy_density(wavelength, temperature):
    return (
        2
        * c.h
        * c.c**2
        / wavelength**5
        / (np.exp(c.h * c.c / (wavelength * c.k_B * temperature)) - 1)
    )


def get_database_value_with_errors(row, key, units=None):
    value = row[key]
    value_min = row[key] - row[key + "_error_min"]
    value_max = row[key] + row[key + "_error_max"]

    if np.isnan(value_min):
        value_min = value
    if np.isnan(value_max):
        value_max = value

    if units is not None:
        value = value * units
        value_min = value_min * units
        value_max = value_max * units
    return [value_min, value, value_max]


def compute_angular_seperation(database_row):
    star_distance_min, star_distance, star_distance_max = (
        get_database_value_with_errors(database_row, "star_distance", u.pc)
    )

    planet_semi_major_axis_min, planet_semi_major_axis, planet_semi_major_axis_max = (
        get_database_value_with_errors(database_row, "semi_major_axis", u.au)
    )

    angular_seperation = np.arctan(planet_semi_major_axis / star_distance).to(u.mas)
    angular_seperation_min = np.arctan(
        planet_semi_major_axis_min / star_distance_max
    ).to(u.mas)
    angular_seperation_max = np.arctan(
        planet_semi_major_axis_max / star_distance_min
    ).to(u.mas)

    return [angular_seperation_min, angular_seperation, angular_seperation_max]


def compute_contrast(database_row, wavel):
    """
    Compute contrast at a given wavelength, with uncertainty

    returns [contrast_min, contrast_mean, contrast_max]
    """
    pass
