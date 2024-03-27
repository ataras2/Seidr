# %%
# from astroquery.gaia import Gaia
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
import astropy.constants as const

from scipy.optimize import fsolve

# %%
# read query output
# df = pd.read_csv("output.ecsv", header=40, delimiter=' ')
# df = pd.read_csv("output_w_mass.ecsv", header=46, delimiter=' ')
# df = pd.read_csv("output_w_mass_temp.ecsv", header=52, delimiter=' ')
df = pd.read_csv("output_w_mass_temp_large.ecsv", header=52, delimiter=" ")
df

# %%
# write out all sources to simbad query file
# with open('id_for_simbad.txt','w') as f:
#     for i in range(len(df)):
#         f.write("GAIA DR3 " + str(df['SOURCE_ID'][i]) + '\n')

# %%
# read simbad query result and match
simbad = pd.read_csv("simbad_result_large.txt", delimiter="\t", header=None)
simbad.columns = [
    "N",
    "Identifier",
    "typed_ident",
    "Otype",
    "RA",
    "DEC",
    "MagU",
    "MagB",
    "MagV",
    "MagR",
    "MagI",
    "MagH",
    "MagK",
    "SpType",
    "Ref",
    "Notes",
]


# %%
df["simbad_id"] = np.vectorize(lambda id: f"GAIA DR3 {id}")(df["SOURCE_ID"])

# %%

# # merge simbad result with gaia result
df = df.merge(simbad, left_on="simbad_id", right_on="typed_ident", how="inner")


# %%
df.shape

# %%
# # remove those with nans in H band
df = df.dropna(subset=["MagH"])
df = df[df["MagH"] != " "]
df["MagH"].to_numpy(float)
df.shape

# %%
plt.figure()
plt.hist(df["lum_flame"], bins=100)
plt.xlabel("Luminosity (L_sun)")
plt.ylabel("Number of stars")

# %%
df["planet_semi_major_axis"] = np.sqrt(df["lum_flame"])

# %%
df["planet_radius"] = 1  # u.Rjup


def contrast(planet_radius, planet_semimajor_axis, albedo):
    planet_radius = planet_radius * u.Rjup
    planet_semimajor_axis = planet_semimajor_axis * u.AU

    planet_area = np.pi * planet_radius**2
    light_shell_area = 4 * np.pi * planet_semimajor_axis**2
    light_reflected = planet_area / light_shell_area * albedo
    return light_reflected


def compute_sep(a, d):
    return (np.arctan((a * u.AU) / (d * u.pc))).to(u.mas).value


# %%
df["contrast"] = np.vectorize(contrast)(
    df["planet_radius"], df["planet_semi_major_axis"], 0.3
)

# %%

wavel = 1.65 * u.micron
longest_baseline = 130 * u.m
michelson_criterion = (0.5 * np.arctan(wavel / longest_baseline)).to(u.mas)

# %%
# make a gif for different values of the seperation

n_frames = 10
assert n_frames % 2 == 0
seperation_coeffs = np.linspace(0.05, 0.5, n_frames // 2)
seperation_coeffs = np.concatenate((seperation_coeffs, seperation_coeffs[::-1]))

coeff = seperation_coeffs.min()  # this will have the brightest stars
df["planet_semi_major_axis"] = np.sqrt(df["lum_flame"]) * coeff
df["angular_speraration"] = np.vectorize(compute_sep)(
    df["planet_semi_major_axis"], df["distance_gspphot"]
)
df["contrast"] = np.vectorize(contrast)(
    df["planet_radius"], df["planet_semi_major_axis"], 0.3
)

Figure = plt.figure()

scatter_pts = plt.scatter(
    df["angular_speraration"],
    df["contrast"],
    c=df["MagH"].to_numpy(float),
    cmap="viridis",
    s=2,
)

plt.colorbar(label="H band magnitude")
plt.xlabel("angular_speraration (mas)")
plt.ylabel("contrast")
plt.yscale("log")
plt.ylim(1e-8, 1e-2)
plt.xlim(0, 20)
plt.axhline(1e-5, color="k", linestyle="--")

plt.axvline(michelson_criterion.value, color="k", linestyle="--")

t = plt.title(f"Hypothetical planet contrast at {coeff:.3f}HZ distance")


def animate(frame_idx, scatter_pts, title):
    coeff = seperation_coeffs[frame_idx]
    df["planet_semi_major_axis"] = np.sqrt(df["lum_flame"]) * coeff
    df["angular_speraration"] = np.vectorize(compute_sep)(
        df["planet_semi_major_axis"], df["distance_gspphot"]
    )
    df["contrast"] = np.vectorize(contrast)(
        df["planet_radius"], df["planet_semi_major_axis"], 0.3
    )

    # if frame_idx > 0:
    #     scatter_pts.remove()
    #     title.remove()
    scatter_pts.set_offsets(
        np.column_stack((df["angular_speraration"], df["contrast"]))
    )
    # scatter_pts.set_array(df["MagH"].to_numpy(float))

    title.set_text(f"Hypothetical planet contrast at {coeff:.3f}HZ distance")

    return scatter_pts, title


from matplotlib.animation import FuncAnimation

anim_created = FuncAnimation(Figure, animate, frames=n_frames, fargs=(scatter_pts, t))

# %%
anim_created.save("planet_contrast_5.gif", fps=15)

# %%
# now think about expected detections as a function of distance
# seperation_coeffs = np.linspace(0.05, 1, 3)
seperation_coeffs = np.logspace(-1, 0, 25)

contrast_limit = 5e-5

contrast_limits = [5e-6, 1e-5, 2e-5, 4e-5, 8e-5]


plt.figure()

n_planets_for_contrast = {}
for c in contrast_limits:
    n_planets_for_contrast[c] = []

for coeff in seperation_coeffs:
    df["planet_semi_major_axis"] = np.sqrt(df["lum_flame"]) * coeff
    df["angular_speraration"] = np.vectorize(compute_sep)(
        df["planet_semi_major_axis"], df["distance_gspphot"]
    )
    df["contrast"] = np.vectorize(contrast)(
        df["planet_radius"], df["planet_semi_major_axis"], 0.3
    )

    for c in contrast_limits:
        n_pla = len(
            df[
                (df["contrast"] > c)
                & (df["angular_speraration"] > michelson_criterion.value)
                & (df["angular_speraration"] < 20)
            ]
        )
        n_planets_for_contrast[c].append(n_pla)

for c in contrast_limits:
    plt.plot(seperation_coeffs, n_planets_for_contrast[c], label=f"contrast > {c}")


plt.legend()
plt.xlabel("Seperation coeff (HZ distance)")
plt.ylabel("Number of planets detected")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1, 2e3)
plt.title(f"Number of hypothetical planets detected as a function of distance")
plt.show()


# %%
max_possible_planets = [max(n_planets_for_contrast[c]) for c in contrast_limits]

print("expectation values:")
print("nominal", [x * 0.137 / 100 for x in max_possible_planets])
print("worst", [x * 0.074 / 100 for x in max_possible_planets])
print("best", [x * 0.249 / 100 for x in max_possible_planets])

# %%
# look at distribution of magnitudes at the biggest peak
coeff = 0.2

contrast_limit = 0.5e-5

df["planet_semi_major_axis"] = np.sqrt(df["lum_flame"]) * coeff
df["angular_speraration"] = np.vectorize(compute_sep)(
    df["planet_semi_major_axis"], df["distance_gspphot"]
)
df["contrast"] = np.vectorize(contrast)(
    df["planet_radius"], df["planet_semi_major_axis"], 0.3
)

viable_subset = df[
    (df["contrast"] > contrast_limit)
    & (df["angular_speraration"] > michelson_criterion.value)
    & (df["angular_speraration"] < 20)
]

plt.figure()
plt.hist(viable_subset["MagH"].to_numpy(float), bins=30)
plt.xlabel("H band magnitude")
plt.ylabel("Number of stars")
plt.title(f"Magnitude of host stars")
plt.show()


plt.figure()
plt.hist(viable_subset["teff_gspphot"].to_numpy(float), bins=30)
plt.xlabel("teff_gspphot")
plt.ylabel("Number of stars")
plt.title(f"temperature of host star")
plt.show()

# periods


def period(a, star_mass):
    return (
        2
        * np.pi
        * np.sqrt((a * u.AU) ** 3 / (const.G * star_mass * u.M_sun)).to(u.day).value
    )


viable_subset["period"] = np.vectorize(period)(
    viable_subset["planet_semi_major_axis"], 0.12
)
plt.figure()
plt.hist(viable_subset["period"], bins=30)
plt.xlabel("period (days)")
plt.ylabel("Number of planets")
plt.title(f"Period of planets")
plt.show()


plt.figure()
plt.hist(viable_subset["planet_semi_major_axis"], bins=30)
plt.xlabel("semi_major_axis (au)")
plt.ylabel("Number of planets")
plt.show()


plt.figure()
plt.hist(np.log10(viable_subset["contrast"]), bins=30)
plt.xlabel("log10(contrast)")
plt.ylabel("Number of planets")
plt.title(f"contrast of planets")
plt.show()


# %%
period(0.03, 0.12)

# %%
len(viable_subset) * 0.137 / 100

# %%
# get the last entry of viable_subset
star = viable_subset.iloc[0]

# period order of magnitude calc
period = (
    2
    * np.pi
    * np.sqrt(
        (star["planet_semi_major_axis"] * u.AU) ** 3 / (const.G * 0.12 * u.M_sun)
    ).to(u.day)
)
period

# %%
# at what distance would a jupiter have to be around the sun to be detected
contrast_limit = 1e-5
planet_radius = 1

# fn = lambda x : (contrast(planet_radius, np.exp(x)*u.AU, 0.3).to("") - contrast_limit)*10**5


# x = fsolve(fn, -12, xtol=1e-6, full_output=True)
# print(x)

contrast(planet_radius, 0.1, 0.3).to("")

# %%
# now think about thermal effects

print(f"reflected contrast is {star['contrast']}")


def spectral_energy_density(wavelength, temperature):
    print(wavelength, temperature)
    return (
        2
        * const.h
        * const.c**2
        / wavelength**5
        / (np.exp(const.h * const.c / (wavelength * const.k_B * temperature)) - 1)
    )


planet_T_eff = (
    (
        star["lum_flame"]
        * u.Lsun
        * (1 - 0.3)
        / (16 * np.pi * const.sigma_sb * (star["planet_semi_major_axis"] * u.AU) ** 2)
    )
    ** (1 / 4)
).to(u.K)

star_energy_density = spectral_energy_density(wavel, 3000 * u.K)
planet_energy_density = spectral_energy_density(wavel, planet_T_eff)

contrast_ratio = (
    planet_energy_density * np.pi * (star["planet_radius"] * u.Rjup) ** 2
) / (star_energy_density * np.pi * 0.149 * u.Rsun**2)

contrast_ratio.to("")
