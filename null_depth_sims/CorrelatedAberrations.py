import jax.numpy as np
import jax.random as jr


class CorrelatedAberrations:
    """
    A class that can be used to generate correlated aberrations
    """

    def __init__(
        self,
        correlation_time,
        rms_amplitudes,
    ) -> None:
        self.correlation_time = correlation_time
        self.rms_amplitudes = rms_amplitudes

    def sample(self, sample_times, key=jr.PRNGKey(0)):
        """
        Sample the correlated aberrations
        """
        del_ts = np.diff(sample_times)

        # draw the inital value
        key, subkey = jr.split(key)
        noise = jr.normal(subkey, (1, len(self.rms_amplitudes))) * self.rms_amplitudes

        rs = np.exp(-del_ts / self.correlation_time)
        tilde_sigs = np.sqrt(1 - rs**2)[:,None] * self.rms_amplitudes[None, :]
        for i in range(1, len(sample_times)):

            key, subkey = jr.split(key)
            noise = np.vstack(
                [
                    noise,
                    rs[i - 1] * noise[-1]
                    + tilde_sigs[i - 1]
                    * jr.normal(subkey, (1, len(self.rms_amplitudes))),
                ]
            )

        return noise


if __name__ == "__main__":

    # test the class
    ca = CorrelatedAberrations(5e-3, np.array([1, 2, 3]) * 100 * 1e-9)

    sample_times = np.linspace(0, 50e-3, 500)

    noise = ca.sample(sample_times)

    print(noise.shape)

    import matplotlib.pyplot as plt

    plt.plot(sample_times, noise)

    ca_long = CorrelatedAberrations(200e-3, np.array([1, 2, 3]) * 100 * 1e-9)

    noise_long = ca_long.sample(sample_times)

    # reset colour
    plt.gca().set_prop_cycle(None)
    plt.plot(sample_times, noise_long, "--")

    plt.show()
