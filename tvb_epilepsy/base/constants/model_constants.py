# coding=utf-8

import numpy as np

# Default model parameters
X0_DEF = 0.0
X0_CR_DEF = 1.0
E_DEF = 0.0
A_DEF = 1.0
B_DEF = 3.0
D_DEF = 5.0
SLOPE_DEF = 0.0
S_DEF = 6.0
GAMMA_DEF = 0.1
K_DEF = 2.5
I_EXT1_DEF = 3.1
I_EXT2_DEF = 0.45
YC_DEF = 1.0
TAU1_DEF = 0.5
TAU2_DEF = 10.0
TAU0_DEF = 30.0  # 2857.0
X1_DEF = -5.0 / 3.0
X1_EQ_CR_DEF = -4.0 / 3.0

WHITE_NOISE = "White"
COLORED_NOISE = "Colored"
NOISE_SEED = 42

TIME_DELAYS_FLAG = 0.0

MAX_DISEASE_VALUE = 1.0 - 10 ** -3

model_noise_intensity_dict = {
    "Epileptor": np.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorModel": np.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDP": np.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDPrealistic": np.array([0., 0., 1e-8, 0.0, 1e-8, 0., 1e-9, 1e-4, 1e-9, 1e-4, 1e-9]),
    "EpileptorDP2D": np.array([0., 1e-7])
}

# model_noise_type_dict = {
#     "Epileptor": WHITE_NOISE,
#     "EpileptorDP": WHITE_NOISE,
#     "EpileptorDPrealistic": COLORED_NOISE,
#     "EpileptorDP2D": WHITE_NOISE
# }

VOIS = {
    "EpileptorModel": ['x1', 'z', 'x2'],
    "Epileptor": ['x1', 'y1', 'z', 'x2', 'y2', 'g'],
    "EpileptorDP": ['x1', 'y1', 'z', 'x2', 'y2', 'g'],
    "EpileptorDPrealistic": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x0_t', 'slope_t', 'Iext1_t', 'Iext2_t', 'K_t'],
    "EpileptorDP2D": ['x1', 'z']
}
