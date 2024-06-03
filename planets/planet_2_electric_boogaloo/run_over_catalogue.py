# %%
import numpy as np
import time
import pyvo

# docs: http://voparis-tap-planeto.obspm.fr/__system__/dc_tables/show/tableinfo/exoplanet.epn_core
service = pyvo.dal.TAPService("http://voparis-tap-planeto.obspm.fr/tap")

query = "SELECT * FROM exoplanet.epn_core"

start = time.time()
results = service.search(query)
end = time.time()

print(f"Query took {end - start} seconds")

print(len(results))
print(len(results[0]))

np.save("myresults", results.to_table())


# %%
import planet_calcs


pandas_table = results.to_table().to_pandas()

# %%
import astropy.units as u
import astropy.constants as c

spectral_energy_density = planet_calcs.spectral_energy_density


def mass_to_radius(mass, density=1.64 * u.g / (u.cm**3)):
    volume = mass / density
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    return radius


def compute_contrast(database_row, wavel, albedo=0.0, density=1.64 * u.g / u.cm**3):
    star_temp = database_row["star_teff"] * u.K  # no errors
    star_radius = database_row["star_radius"] * u.Rsun  # no errors

    if np.isnan(database_row["mass"]):
        planet_mass_min, planet_mass, planet_mass_max = (
            planet_calcs.get_database_value_with_errors(
                database_row, "mass_sin_i", units=u.Mjup
            )
        )
        uses_sin_i = True
    else:
        planet_mass_min, planet_mass, planet_mass_max = (
            planet_calcs.get_database_value_with_errors(
                database_row, "mass", units=u.Mjup
            )
        )
        uses_sin_i = False
    # planet_mass_min, planet_mass, planet_mass_max = planet_calcs.get_database_value_with_errors(database_row, "mass_sin_i", units=u.Mjup)
    if np.isnan(planet_mass):
        return [np.nan * u.dimensionless_unscaled] * 3, False

    # infer radius
    planet_radius_min = mass_to_radius(planet_mass_min, density)
    planet_radius = mass_to_radius(planet_mass, density)
    planet_radius_max = mass_to_radius(planet_mass_max, density)

    star_bolometric_luminosity = 4 * np.pi * star_radius**2 * c.sigma_sb * star_temp**4

    # now eq temp
    semi_major_axis_min, semi_major_axis, semi_major_axis_max = (
        planet_calcs.get_database_value_with_errors(
            database_row, "semi_major_axis", units=u.AU
        )
    )

    planet_T_eq = (
        star_bolometric_luminosity
        * (1 - albedo)
        / (16 * np.pi * c.sigma_sb * semi_major_axis**2)
    ) ** (1 / 4)
    planet_T_eq_min = (
        star_bolometric_luminosity
        * (1 - albedo)
        / (16 * np.pi * c.sigma_sb * semi_major_axis_max**2)
    ) ** (1 / 4)
    planet_T_eq_max = (
        star_bolometric_luminosity
        * (1 - albedo)
        / (16 * np.pi * c.sigma_sb * semi_major_axis_min**2)
    ) ** (1 / 4)

    star_energy_density = spectral_energy_density(wavel, star_temp)
    planet_energy_density = spectral_energy_density(wavel, planet_T_eq)
    planet_energy_density_min = spectral_energy_density(wavel, planet_T_eq_min)
    planet_energy_density_max = spectral_energy_density(wavel, planet_T_eq_max)

    ratio = (planet_energy_density * np.pi * planet_radius**2) / (
        star_energy_density * np.pi * star_radius**2
    )
    ratio_min = (planet_energy_density_min * np.pi * planet_radius_min**2) / (
        star_energy_density * np.pi * star_radius**2
    )
    ratio_max = (planet_energy_density_max * np.pi * planet_radius_max**2) / (
        star_energy_density * np.pi * star_radius**2
    )

    return [ratio_min.to(""), ratio.to(""), ratio_max.to("")], uses_sin_i


wavel = 1.65 * u.micron

seperations = []
sep_errors = []
contrasts = []
contrast_errors = []
sin_i_mask = []
for index, row in pandas_table.iterrows():
    sep = planet_calcs.compute_angular_seperation(row)
    contrast, uses_sin_i = compute_contrast(row, wavel)

    sep_err = [sep[1].value - sep[0].value, sep[2].value - sep[1].value]
    if (np.array(sep_err) < 0).sum() > 0:
        seperations.append(np.nan)
        sep_errors.append([np.nan, np.nan])
    else:
        seperations.append(sep[1].value)
        sep_errors.append(sep_err)

    c_err = [
        contrast[1].value - contrast[0].value,
        contrast[2].value - contrast[1].value,
    ]

    if (np.array(c_err) < 0).sum() > 0:
        contrasts.append(np.nan)
        contrast_errors.append([np.nan, np.nan])
    else:
        contrasts.append(contrast[1].value)
        contrast_errors.append(c_err)

    sin_i_mask.append(uses_sin_i)


# # #add to pandas table
pandas_table["contrast"] = contrasts
pandas_table["seperation"] = seperations


# %%

sep_errors = np.array(sep_errors).T
contrast_errors = np.array(contrast_errors).T
seperations = np.array(seperations)
contrasts = np.array(contrasts)
sin_i_mask = np.array(sin_i_mask)

sep_errors.shape, sin_i_mask.shape

# %%
np.isnan(sep_errors).sum(axis=0), np.isnan(contrast_errors).sum(axis=0)
has_errors = np.logical_and(
    np.isnan(sep_errors).sum(axis=0) == 0, np.isnan(contrast_errors).sum(axis=0) == 0
)

# %%
sep_errors[:, has_errors]

# %%
pandas_table["dec"]

# %%
is_good_dec = np.array([pandas_table["dec"] < 20]).flatten()
is_good_dec

# %%

import matplotlib.pyplot as plt

# plt.scatter(seperations, contrasts, s=5)
# plt.errorbar(seperations[sin_i_mask], contrasts[sin_i_mask], xerr=sep_errors[sin_i_mask], yerr=contrast_errors[sin_i_mask], fmt="o", markersize=5,)

# sin_i_mask = np.logical_not(sin_i_mask)
# plt.errorbar(seperations[sin_i_mask], contrasts[sin_i_mask], xerr=sep_errors[sin_i_mask], yerr=contrast_errors[sin_i_mask], fmt="o", markersize=5,)
plt.errorbar(
    seperations[is_good_dec],
    contrasts[is_good_dec],
    xerr=sep_errors[:, is_good_dec],
    yerr=contrast_errors[:, is_good_dec],
    fmt="o",
    markersize=3,
    label="dec < 20°",
)
is_good_dec = np.logical_not(is_good_dec)
plt.errorbar(
    seperations[is_good_dec],
    contrasts[is_good_dec],
    xerr=sep_errors[:, is_good_dec],
    yerr=contrast_errors[:, is_good_dec],
    fmt="o",
    markersize=3,
    label="dec > 20°",
    alpha=0.2,
)
is_good_dec = np.logical_not(is_good_dec)


longest_baseline = 130 * u.m
contrast_ylimit = 10**-8
plt.xlim(0.9, 12)
plt.yscale("log")
# plt.xscale("log")
plt.ylim(contrast_ylimit, 10**-3)
plt.axvline(
    ((wavel / (2 * longest_baseline)) * u.rad).to(u.mas).value,
    color="black",
    linestyle="--",
    label="0.5 λ/longest baseline",
)
plt.legend()
plt.xlabel("Angular separation (milliarcseconds)")
plt.ylabel("Contrast of thermal emission in H band")
plt.savefig("contrast_vs_seperation.pdf")


# %%
pandas_table["log10contrast"] = np.log10(pandas_table["contrast"])

# %%
achievable = pandas_table[pandas_table["contrast"] > 5 * 10**-6]
achievable = achievable[
    achievable["seperation"]
    > ((wavel / (2 * longest_baseline)) * u.rad).to(u.mas).value
]
achievable = achievable[achievable["dec"] < 20]

table = achievable[["target_name", "log10contrast", "seperation", "dec"]]
print(
    table.sort_values("log10contrast", ascending=False).to_latex(
        index=False, float_format="{:.2f}".format
    )
)
