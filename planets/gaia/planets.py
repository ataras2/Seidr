"""
Collection of functions for working with planets
"""

import astropy.units as u
import astropy.constants as const
import numpy as np


def has_units(x):
    """
    Check if a variable has units

    Parameters
    ----------
    x : Any
        The variable to check

    Returns
    -------
    bool
        True if the variable has astropy units, False otherwise
    """
    return isinstance(x, u.quantity.Quantity)


def reflected_contrast(
    planet_radius,
    planet_semimajor_axis,
    albedo,
    phase_contrast_factor=0.5,
    planet_rad_units=u.Rjup,
    planet_semimajor_axis_units=u.AU,
):
    """
    Computes the contrast of a planet based on its radius, semi-major axis, and albedo using reflected light

    Parameters
    ----------
    planet_radius : float or Quantity
        The radius of the planet
    planet_semimajor_axis : float or Quantity
        The semi-major axis of the planet
    albedo : float
        The albedo of the planet, 0 <= albedo <= 1
    phase_contrast_factor : float
        The phase contrast factor of the planet, 0 <= phase_contrast_factor <= 1. e.g. phase angle of 90 degrees
        is 0.5
    planet_rad_units : Quantity
        The units of the planet radius, default is Jupiter radii
    planet_semimajor_axis_units : Quantity
        The units of the planet semi-major axis, default is AU

    Returns
    -------
    float
        The contrast of the planet based on reflected light
    """
    if not has_units(planet_radius):
        planet_radius = planet_radius * planet_rad_units
    if not has_units(planet_semimajor_axis):
        planet_semimajor_axis = planet_semimajor_axis * planet_semimajor_axis_units

    planet_area = np.pi * planet_radius**2
    light_shell_area = 4 * np.pi * planet_semimajor_axis**2
    contrast_reflect = planet_area / light_shell_area * albedo * phase_contrast_factor
    return contrast_reflect


def compute_angular_sep(
    semi_major_axis,
    distance_to_star,
    semi_major_axis_units=u.AU,
    distance_units=u.pc,
):
    """
    Compute the angular separation of a planet from its star

    Parameters
    ----------
    semi_major_axis : float or Quantity
        The semi-major axis of the planet
    distance_to_star : float or Quantity
        The distance to the star
    semi_major_axis_units : Quantity
        The units of the semi-major axis, default is AU
    distance_units : Quantity
        The units of the distance to the star, default is parsecs

    Returns
    -------
    float
        The angular separation of the planet from the star as a Quantity
    """
    if not has_units(semi_major_axis):
        semi_major_axis = semi_major_axis * semi_major_axis_units
    if not has_units(distance_to_star):
        distance_to_star = distance_to_star * distance_units

    return np.arctan((semi_major_axis) / (distance_to_star))


def period(
    semi_major_axis,
    star_mass,
    semi_major_axis_units=u.AU,
    star_mass_units=u.M_sun,
):
    """
    Find the period of a planet in days given its semi-major axis and the mass of the star
    """
    if not has_units(semi_major_axis):
        semi_major_axis = semi_major_axis * semi_major_axis_units
    if not has_units(star_mass):
        star_mass = star_mass * star_mass_units
    return 2 * np.pi * np.sqrt((semi_major_axis) ** 3 / (const.G * star_mass))
