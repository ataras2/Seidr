"""
Answer the question: what does LP mode coupling look like for few mode fibers under 
correlated, random aberrations?

"""

import seidrsim
import correlatednoise

import jax
import jax.numpy as np
import time


n_zernikes = 6
sim = seidrsim.SeidrSim.make_default(type="mmf5", n_zernikes=n_zernikes)

n_images = 10_000
# n_images = 100
image_rate = 2e3  # Hz
image_times = np.arange(n_images) / image_rate

# correlated noise
correlation_time = 5e-3
ca = correlatednoise.CorrelatedNoise(
    correlation_time, np.array([0, 200, 200, 20, 20, 20]) * 1e-9
)


def zernike_to_lp_coupling(zernike_coeffs):
    sim.optics = sim.optics.set("aperture.coefficients", zernike_coeffs)
    return sim.propagate_injections(is_complex=True)[1]


start = time.time()
zernikes = ca.sample(image_times)
lp_couplings = jax.vmap(zernike_to_lp_coupling)(zernikes)
end = time.time()

print(f"Time taken: {end - start}, or {1000 * (end - start) / n_images} ms per image")


# save the data
np.savez(
    "lp_couplings.npz",
    lp_couplings=lp_couplings,
    image_times=image_times,
    correlation_time=correlation_time,
    rms_amplitudes=ca.rms_amplitudes,
    n_zernikes=n_zernikes,
    zernikes=zernikes,
)
