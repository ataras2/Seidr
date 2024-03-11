# trying to follow last paragraph of section3 in "Kernel-nulling for a robust direct interferometric
# detection of extrasolar planets"


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


class Star:
    def __init__(
        self, name: str, distance, effective_temp, radius, mass, planets
    ) -> None:
        self.name = name
        self.distance = distance
        self.effective_temp = effective_temp
        self.radius = radius
        self.mass = mass

        self.planets = planets

    def __str__(self) -> str:
        return f"{self.name} at {self.distance} with T_eff={self.effective_temp} and R={self.radius}, M={self.mass}"

    def angular_diameter(self):
        return np.arctan(self.radius / self.distance)

    @staticmethod
    def create_from_exoplanet_database_row(row, planet_density):
        """assumes a row from the exoplanet database
        http://voparis-tap-planeto.obspm.fr/__system__/dc_tables/show/tableinfo/exoplanet.epn_core

        assumes input planet density has astropy units
        """
        # check input has astropy units
        assert isinstance(planet_density, u.Quantity)

        # M_sin_i, semi_major_axis, density
        planet_args_database_list = [
            "target_name",
            "mass_sin_i",
            "semi_major_axis",
        ]
        # the args for the constructor are not the same as the database
        planet_args_ctor_list = [
            "name",
            "M_sin_i",
            "semi_major_axis",
        ]

        planet_args_units = [None, u.M_jup, u.au]

        # validate
        for arg in planet_args_database_list:
            assert arg in row.keys(), f"missing {arg} in row {row}"

        planet_init_dict = {}
        for d_arg, p_arg, unit in zip(
            planet_args_database_list, planet_args_ctor_list, planet_args_units
        ):
            value = row[d_arg]
            if unit:
                value = value * unit
            planet_init_dict[p_arg] = value

        planet_init_dict["density"] = planet_density

        # name, distance, effective_temp, radius, mass,
        star_args_database_list = [
            "star_name",
            "star_distance",
            "star_teff",
            "star_radius",
            "star_mass",
        ]

        star_args_ctor_list = [
            "name",
            "distance",
            "effective_temp",
            "radius",
            "mass",
        ]

        star_args_units = [None, u.pc, u.K, u.R_sun, u.M_sun]

        # validate
        for arg in star_args_database_list:
            assert arg in row.keys(), f"missing {arg} in row {row}"

        star_init_dict = {}
        for d_arg, s_arg, unit in zip(
            star_args_database_list, star_args_ctor_list, star_args_units
        ):
            value = row[d_arg]
            if unit:
                value = value * unit
            star_init_dict[s_arg] = value

        # create star
        star = Star(**star_init_dict, planets=[Planet(**planet_init_dict)])

        return star

    @property
    def planet_angular_separation(self):
        separations = []
        for planet in self.planets:
            separations.append(np.arctan(planet.semi_major_axis / self.distance))
        return separations

    @property
    def bolometric_luminosity(self):
        return 4 * np.pi * self.radius**2 * c.sigma_sb * self.effective_temp**4


    def planet_eq_temps(self, albedo=0.0):
        """
        Assumes a thermal equilibrium with the star, such that the total incident flux is equal to the total emitted flux
        """
        eq_temps = []
        for planet in self.planets:
            T_eff = (
                self.bolometric_luminosity*(1-albedo)
                / (16 * np.pi * c.sigma_sb * planet.semi_major_axis**2)
            ) ** (1 / 4)

            eq_temps.append(T_eff)
        return eq_temps

    def planet_contrast_thermal(self, wavelength,albedo=0.0):
        contrasts = []
        temps = self.planet_eq_temps(albedo=albedo)
        radii = [planet.radius_lower_bound for planet in self.planets]
        for p_idx, planet in enumerate(self.planets):
            star_energy_density = spectral_energy_density(
                wavelength, self.effective_temp
            )
            planet_energy_density = spectral_energy_density(wavelength, temps[p_idx])

            contrast_ratio = (planet_energy_density * np.pi * radii[p_idx] ** 2) / (
                star_energy_density * np.pi * self.radius**2
            )

            contrast = np.log10(contrast_ratio.to(""))
            contrasts.append(contrast)
        return contrasts

    def planet_contrast_reflected(self, wavelength, albedo, phase_angle_factor):
        contrasts = []
        temps = self.planet_eq_temps
        radii = [planet.radius_lower_bound for planet in self.planets]
        for planet in self.planets:
            planet_area = np.pi * planet.radius_lower_bound**2
            light_shell_area = 4 * np.pi * planet.semi_major_axis**2

            stellar_energy_density = spectral_energy_density(
                wavelength, self.effective_temp
            )

            planet_energy_density = (
                stellar_energy_density
                * albedo
                * planet_area
                * phase_angle_factor
                / light_shell_area
            )

            contrast_ratio = planet_energy_density / stellar_energy_density

            contrasts.append(np.log10(contrast_ratio.to("")))
        return contrasts


class Planet:
    def __init__(self, name: str, M_sin_i, semi_major_axis, density) -> None:
        self.name = name
        self.semi_major_axis = semi_major_axis
        self.M_sin_i = M_sin_i
        self.density = density

    def __str__(self) -> str:
        return f"{self.name} at {self.semi_major_axis} with M_sin_i={self.M_sin_i}"

    @property
    def mass_lower_bound(self):
        return self.M_sin_i

    @property
    def volume_lower_bound(self):
        return self.mass_lower_bound / self.density

    @property
    def radius_lower_bound(self):
        return (3 * self.volume_lower_bound / (4 * np.pi)) ** (1 / 3)


if __name__ == "__main__":
    # p = Planet(
    #     "GJ 86 b",
    #     M_sin_i=4.27 * u.M_jup,
    #     semi_major_axis=0.1177 * u.au,
    #     density=1.64 * u.g / u.cm**3,
    # )

    # s = Star(
    #     "GJ 86",
    #     distance=10.9 * u.pc,
    #     effective_temp=5_350 * u.K,
    #     radius=0.855 * u.R_sun,
    #     mass=0.8 * u.M_sun,
    #     planets=[p],
    # )
    p = Planet(
        "HD 77946 b",
        M_sin_i=0.02637* u.M_jup,
        semi_major_axis=0.072 * u.au,
        density=1.64 * u.g / u.cm**3,
    )

    s = Star(
        "HD 77946",
        distance=99 * u.pc,
        effective_temp=6_046 * u.K,
        radius=1.31 * u.R_sun,
        mass=1.17* u.M_sun,
        planets=[p],
    )

    earth = Planet(
        "Earth",
        M_sin_i=1 * u.M_earth,
        semi_major_axis=1 * u.au,
        density=5.51 * u.g / u.cm**3,
    )

    sun = Star(
        "Sun",
        distance=0 * u.pc,
        effective_temp=5_780 * u.K,
        radius=1.0 * u.R_sun,
        mass=1.0 * u.M_sun,
        planets=[earth],
    )

    print([x.to(u.mas) for x in s.planet_angular_separation])

    print(s.planet_contrast_thermal(1.630 * u.micron))
    print(s.planet_contrast_thermal(3.450 * u.micron))
    print(s.planet_contrast_reflected(1.630 * u.micron,0.1,0.5))
    print(s.planet_contrast_reflected(3.450 * u.micron,0.1,0.5))
