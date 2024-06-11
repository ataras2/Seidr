"""
Make a gif of the abberrated psfs file
"""

import jax
import jax.numpy as np
import copy

import matplotlib.pyplot as plt

fname = "aberrated_psfs_many_zern_MMF"
data = np.load(fname + ".npz")

wavefronts = data["outputs"]
non_aberrated = data["non_aberrated"]

max_r = 6
n_frames = 100

Figure = plt.figure(figsize=(8, 4))


# add circle to show the PL input
circle = plt.Circle(
    (non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2),
    data["psf_npixels"] // (max_r * 2),
    fill=False,
    linestyle="--",
    color="w",
)

plt.subplot(1, 2, 1)
amp_img = plt.imshow(np.abs(non_aberrated), cmap="inferno")
# mark centre with a little cross
plt.plot(non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2, "+", color="r")
plt.colorbar()
plt.title("Amplitude")
plt.gca().add_artist(copy.copy(circle))

plt.subplot(1, 2, 2)
phase_img = plt.imshow(np.angle(non_aberrated), cmap="twilight")
plt.plot(non_aberrated.shape[0] // 2, non_aberrated.shape[1] // 2, "+", color="r")
plt.colorbar()
plt.title("Phase")
plt.gca().add_artist(copy.copy(circle))

plt.savefig(fname + ".png")


def animate(
    frame_idx,
):
    amp_img.set_data(np.abs(wavefronts[frame_idx]))
    phase_img.set_data(np.angle(wavefronts[frame_idx]))

    return amp_img, phase_img


from matplotlib.animation import FuncAnimation

anim_created = FuncAnimation(Figure, animate, frames=n_frames)

anim_created.save(fname + ".gif", fps=15)
