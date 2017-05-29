# coding=utf-8
"""
Factory methods to build the wanted epileptor model.
"""

###
# Build TVB Epileptor
###
import numpy
from tvb.simulator.models import Epileptor

from tvb_epilepsy.base.calculations import calc_rescaled_x0, calc_x0cr_r, calc_x0
from tvb_epilepsy.base.constants import ADDITIVE_NOISE, MULTIPLICATIVE_NOISE
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDP, EpileptorDPrealistic


def build_tvb_model(model_configuration, a=1.0, b=3.0, d=5.0, zmode="lin"):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1(), a, b - d, zmode=zmode)
    model_instance = Epileptor(x0=x0_transformed.flatten(), Iext=model_configuration.get_Iext1().flatten(),
                               Ks=model_configuration.get_K().flatten(), c=model_configuration.get_yc().flatten())

    noise_intensity = numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])
    noise_type = ADDITIVE_NOISE

    return model_instance, noise_intensity, noise_type


###
# Build EpileptorDP2D
###
def build_ep_2sv_model(model_configuration, zmode=numpy.array("lin")):
    if zmode == "lin":
        x0 = model_configuration.x0
        x0cr = model_configuration.x0cr
        rx0 = model_configuration.rx0
        # elif zmode == "sig":
        # Correct Ceq, x0cr, rx0 and x0 for sigmoidal fz(x1)

        # TODO: is this needed? Do the yc, Iext1 ever change?
        # (x0cr, rx0) = calc_x0cr_r(model_configuration.get_yc(), model_configuration.get_Iext1(), zmode=zmode)
        # x1EQ, zEQ, = model_configuration.get_equilibrum_points()
        # x0 = calc_x0(x1EQ, zEQ, model_configuration.get_K(), hypothesis.weights, x0cr, rx0, model="2d", zmode=zmode)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')
    model = EpileptorDP2D(x0=x0.T, Iext1=model_configuration.get_Iext1().T, K=model_configuration.get_K().T,
                          yc=model_configuration.get_yc().T, r=rx0.T, x0cr=x0cr.T, zmode=zmode)

    noise_intensity = numpy.array([0., 5e-5])
    noise_type = ADDITIVE_NOISE

    return model, noise_intensity, noise_type


###
# Build EpileptorDP
###
def build_ep_6sv_model(model_configuration, zmode=numpy.array("lin")):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1())
    model = EpileptorDP(x0=x0_transformed.T, Iext1=model_configuration.get_Iext1().T, K=model_configuration.get_K().T,
                        yc=model_configuration.get_yc().T, zmode=zmode)

    noise_intensity = numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])
    noise_type = ADDITIVE_NOISE

    return model, noise_intensity, noise_type


###
# Build EpileptorDPrealistic
###
def build_ep_11sv_model(model_configuration, zmode=numpy.array("lin")):
    x0_transformed = calc_rescaled_x0(model_configuration.get_x0_values(), model_configuration.get_yc(),
                                      model_configuration.get_Iext1())
    model = EpileptorDPrealistic(x0=x0_transformed.T, Iext1=model_configuration.get_Iext1().T,
                                 K=model_configuration.get_K().T, yc=model_configuration.get_yc().T, zmode=zmode)

    noise_intensity = numpy.array([0., 0., 1e-7, 0.0, 1e-7, 0., 1e-8, 1e-3, 1e-8, 1e-3, 1e-9])
    noise_type = MULTIPLICATIVE_NOISE

    return model, noise_intensity, noise_type


# Model creator functions dictionary (factory)
model_build_dict = {"Epileptor": build_tvb_model,
                    "EpileptorDP": build_ep_6sv_model,
                    "EpileptorDPrealistic": build_ep_11sv_model,
                    "EpileptorDP2D": build_ep_2sv_model}
