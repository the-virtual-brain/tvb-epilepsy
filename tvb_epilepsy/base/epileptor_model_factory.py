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
from tvb_epilepsy.base.calculations import calc_rescaled_x0
from tvb_epilepsy.base.constants import ADDITIVE_NOISE, MULTIPLICATIVE_NOISE
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDP, EpileptorDPrealistic


def build_tvb_model(model_configuration, a=1.0, b=3.0, d=5.0, zmode="lin"):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1(), a, b - d, zmode=zmode)
    model_instance = Epileptor(x0=x0_transformed.flatten(), Iext=model_configuration.get_Iext1().flatten(),
                               Ks=model_configuration.get_K().flatten(), c=model_configuration.get_yc().flatten())

    return model_instance


###
# Build EpileptorDP2D
###
def build_ep_2sv_model(model_configuration, zmode=numpy.array("lin")):
    x0 = model_configuration.x0
    x0cr = model_configuration.x0cr
    rx0 = model_configuration.rx0

    model = EpileptorDP2D(x0=x0.T, Iext1=model_configuration.get_Iext1().T, K=model_configuration.get_K().T,
                          yc=model_configuration.get_yc().T, r=rx0.T, x0cr=x0cr.T, zmode=zmode)

    return model


###
# Build EpileptorDP
###
def build_ep_6sv_model(model_configuration, zmode=numpy.array("lin")):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1())
    model = EpileptorDP(x0=x0_transformed.T, Iext1=model_configuration.get_Iext1().T, K=model_configuration.get_K().T,
                        yc=model_configuration.get_yc().T, zmode=zmode)

    return model


###
# Build EpileptorDPrealistic
###
def build_ep_11sv_model(model_configuration, zmode=numpy.array("lin")):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1())
    model = EpileptorDPrealistic(x0=x0_transformed.T, Iext1=model_configuration.get_Iext1().T,
                                 K=model_configuration.get_K().T, yc=model_configuration.get_yc().T, zmode=zmode)

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
    "EpileptorDP": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDPrealistic": numpy.array([0., 0., 1e-7, 0.0, 1e-7, 0., 1e-8, 1e-3, 1e-8, 1e-3, 1e-9]),
    "EpileptorDP2D": numpy.array([0., 5e-5])
}

model_noise_type_dict = {
    "Epileptor": ADDITIVE_NOISE,
    "EpileptorDP": ADDITIVE_NOISE,
    "EpileptorDPrealistic": MULTIPLICATIVE_NOISE,
    "EpileptorDP2D": ADDITIVE_NOISE
}
