import numpy as np
from tvb_epilepsy.service.model_inversion.epileptor_params_factory \
                                            import generate_negative_lognormal_parameter


if __name__ == "__main__":

    x0 = generate_negative_lognormal_parameter("x0", -2.5 * np.ones(2,), -4.0, 1.0,
                                               sigma=None, sigma_scale=3,
                                               p_shape=(2,), use="scipy")

    print(x0)