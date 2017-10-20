# coding=utf-8
"""
Factory methods to build the wanted epileptor model.
Also, dictionaries to keep noise intensity and type for each model type.
"""

###
# Build TVB Epileptor
###
import numpy
from tvb.simulator.models import Epileptor

from tvb_epilepsy.base.constants import ADDITIVE_NOISE, MULTIPLICATIVE_NOISE
from tvb_epilepsy.base.computations.calculations_utils import calc_x0_val__to_model_x0
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDP, EpileptorDPrealistic


def build_tvb_model(model_configuration, zmode=numpy.array("lin")):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model_instance = Epileptor(x0=model_configuration.x0, Iext=model_configuration.Iext1, Iext2=model_configuration.Iext2,
                               Ks=K, c=model_configuration.yc,
                               a=model_configuration.a, b=model_configuration.b, d=model_configuration.d,
                               aa=model_configuration.s)

    return model_instance


###
# Build EpileptorDP2D
###
def build_ep_2sv_model(model_configuration, zmode=numpy.array("lin")):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP2D(x0=model_configuration.x0, Iext1=model_configuration.Iext1, K=K,
                          yc=model_configuration.yc, a=model_configuration.a, b=model_configuration.b,
                          d=model_configuration.d, zmode=zmode)

    return model


###
# Build EpileptorDP
###
def build_ep_6sv_model(model_configuration, zmode=numpy.array("lin")):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP(x0=model_configuration.x0, Iext1=model_configuration.Iext1, Iext2=model_configuration.Iext2,
                        K=K, yc=model_configuration.yc, a=model_configuration.a,
                        b=model_configuration.b, d=model_configuration.d, s=model_configuration.s,
                        gamma=model_configuration.gamma, zmode=zmode)

    return model


###
# Build EpileptorDPrealistic
###
def build_ep_11sv_model(model_configuration, zmode=numpy.array("lin"), pmode=numpy.array("z")):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDPrealistic(x0=model_configuration.x0, Iext1=model_configuration.Iext1,
                                 Iext2=model_configuration.Iext2, K=K, yc=model_configuration.yc,
                                 a=model_configuration.a, b=model_configuration.b, d=model_configuration.d,
                                 s=model_configuration.s, gamma=model_configuration.gamma, zmode=zmode, pmode=pmode)

    return model


# Model creator functions dictionary (factory)
model_build_dict = {
    "Epileptor": build_tvb_model,
    "EpileptorDP": build_ep_6sv_model,
    "EpileptorDPrealistic": build_ep_11sv_model,
    "EpileptorDP2D": build_ep_2sv_model
}

model_noise_intensity_dict = {
    "Epileptor": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorModel": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDP": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDPrealistic": numpy.array([0., 0., 1e-7, 0.0, 1e-7, 0., 1e-8, 1e-3, 1e-8, 1e-3, 1e-9])*0.1,
    "EpileptorDP2D": numpy.array([0., 5e-8])
}

model_noise_type_dict = {
    "Epileptor": ADDITIVE_NOISE,
    "EpileptorDP": ADDITIVE_NOISE,
    "EpileptorDPrealistic": MULTIPLICATIVE_NOISE,
    "EpileptorDP2D": ADDITIVE_NOISE
}
