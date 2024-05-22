import numpy as np


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

# Inputs


# suppose we are now at the nuller chip

sigma_I = 0.001  # rms intensity error
sigma_phi = 10e-9  # rms phase error
n_beams = 4  # number of beams
n_runs = 1000

input_beam_amplitude = np.random.normal(1, sigma_I, (n_beams, n_runs))
input_beam_phase = (np.random.normal(0, sigma_phi, (n_beams, n_runs)))

input_beam_field = input_beam_amplitude * np.exp(1j * input_beam_phase)

M_matrix = 0.25 * np.array(
    [
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j],
        [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
        [1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j],
        [1 + 1j, -1 - 1j, -1 + 1j, 1 - 1j],
    ]
)

K_matrix = np.array(
    [
        [1, -1, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 1, -1],
    ],
    dtype=np.float32,
)

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
