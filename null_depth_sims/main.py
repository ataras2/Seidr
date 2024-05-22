import jax
import jax.numpy as np
import dLux.utils as dlu
import matplotlib.pyplot as plt


# Overall plan:
# program inputs: companion properties
# fixed constants: detector, VLTI properties
# outputs: null depth distribution and hence SNR as ratio of companion light to starlight
# define inputs from baldr as a distribution of zernikies
# apply arbitrary correction from PL loop
# assume first n LP modes are injected (for n=1,3)
# Correction due to kernel nuller chip
# look at overall null depth

tscope_type = "UT"  # UT or AT


if tscope_type == "UT":
    diameter = 8.2
elif tscope_type == "AT":
    diameter = 1.8
else:
    raise ValueError("Invalid scope type")


# Chingaipe version
M_matrix = 0.25 * np.array(
    [
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],
        [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
        [1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j],
        [1 + 1j, -1 - 1j, -1 + 1j, 1 - 1j],
    ],
    dtype=np.complex64,
)

# Martinache version
# theta = np.pi / 2
# M_matrix = 0.25 * np.array(
#     [
#         [
#             1 + np.exp(1j * theta),
#             1 - np.exp(1j * theta),
#             -1 + np.exp(1j * theta),
#             -1 - np.exp(1j * theta),
#         ],
#         [
#             1 - np.exp(-1j * theta),
#             -1 - np.exp(-1j * theta),
#             1 + np.exp(-1j * theta),
#             -1 + np.exp(-1j * theta),
#         ],
#         [
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#         ],
#         [
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#         ],
#         [
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#         ],
#         [
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#             1 + np.exp(1j * theta),
#         ],
#     ],
#     dtype=np.complex64,
# )

# raise NotImplementedError


N_matrix = 0.5 * np.array(
    [
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)

K_matrix = np.array(
    [
        [1, -1, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 1, -1],
    ],
    dtype=np.float32,
)


# Inputs


# suppose we are now at the nuller chip

sigma_I = 0.001  # rms intensity error
sigma_phi = 10e-9  # rms phase error
n_beams = 4  # number of beams
n_runs = 1000

Tscope_positions = np.array(
    [
        [0.0, 0.0],
        [10.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0],
    ]
)

# define companion
contrast = 1e-2
wavelength = 1.6e-6  # meters


def nuller_given_position(pos, matrix):
    phases = (
        (Tscope_positions - Tscope_positions[2]) * dlu.arcsec2rad(pos) / wavelength
    ).sum(axis=1)

    fields = np.exp(1j * phases)

    # what does the detector see
    detector_outputs = np.abs(matrix @ fields) ** 2
    return detector_outputs


n_samp = 100
max_sep = 42
companion_positions = np.vstack(
    [np.linspace(-max_sep, max_sep, n_samp) * 1e-3, np.zeros(n_samp)]
).T  # mas
print(companion_positions)

oned_throughputs = jax.vmap(lambda c: nuller_given_position(c, N_matrix))(
    companion_positions,
)
print(oned_throughputs.shape)

plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, oned_throughputs)

oned_throughputs = jax.vmap(lambda c: nuller_given_position(c, M_matrix))(
    companion_positions,
)
print(oned_throughputs.shape)

plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, oned_throughputs)


plt.figure()
plt.plot(companion_positions[:, 0] * 1e3, (K_matrix @ oned_throughputs.T).T)
plt.show()


exit()


input_beam_amplitude = np.random.normal(1, sigma_I, (n_beams, n_runs))
input_beam_phase = np.random.normal(0, sigma_phi, (n_beams, n_runs))

input_beam_field = input_beam_amplitude * np.exp(1j * input_beam_phase)

detector_outputs = np.abs(M_matrix @ input_beam_field) ** 2

kernel_outputs = K_matrix @ detector_outputs

print(np.std(kernel_outputs, axis=1))

import matplotlib.pyplot as plt

plt.figure()
plt.hist(kernel_outputs[0], bins=50, alpha=0.5, label="kernel 1")
plt.hist(kernel_outputs[1], bins=50, alpha=0.5, label="kernel 2")
plt.hist(kernel_outputs[2], bins=50, alpha=0.5, label="kernel 3")
plt.legend()
plt.show()
