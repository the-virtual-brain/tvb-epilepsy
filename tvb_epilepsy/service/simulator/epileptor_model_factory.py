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
from tvb_epilepsy.service.simulator.simulator_java import EpileptorModel
from tvb_epilepsy.base.epileptor_models import EpileptorDP2D, EpileptorDP, EpileptorDPrealistic

AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)


AVAILABLE_DYNAMICAL_MODELS_NAMES = []
for model in AVAILABLE_DYNAMICAL_MODELS:
    AVAILABLE_DYNAMICAL_MODELS_NAMES.append(model._ui_name)

#TODO: Ensure that function signatures are the same. An Epileptor is build using model_configuration attributes. Take zmode it from model_configuration. Keep also pmode there or other values that can be received by kwargs.
def build_tvb_model(model_configuration):
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
def build_EpileptorDP2D(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP2D(x0=model_configuration.x0, Iext1=model_configuration.Iext1, K=K,
                          yc=model_configuration.yc, a=model_configuration.a, b=model_configuration.b,
                          d=model_configuration.d, zmode=model_configuration.zmode)
    return model


###
# Build EpileptorDP
###
def build_EpileptorDP(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP(x0=model_configuration.x0, Iext1=model_configuration.Iext1, Iext2=model_configuration.Iext2,
                        K=K, yc=model_configuration.yc, a=model_configuration.a,
                        b=model_configuration.b, d=model_configuration.d, s=model_configuration.s,
                        gamma=model_configuration.gamma, zmode=model_configuration.zmode)
    return model


###
# Build EpileptorDPrealistic
###
def build_EpileptorDPrealistic(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDPrealistic(x0=model_configuration.x0, Iext1=model_configuration.Iext1,
                                 Iext2=model_configuration.Iext2, K=K, yc=model_configuration.yc,
                                 a=model_configuration.a, b=model_configuration.b, d=model_configuration.d,
                                 s=model_configuration.s, gamma=model_configuration.gamma,
                                 zmode=model_configuration.zmode, pmode=numpy.array("z"))
    return model


# Model creator functions dictionary (factory)
model_build_dict = {
    "Epileptor": build_tvb_model,
    "EpileptorDP": build_EpileptorDP,
    "EpileptorDPrealistic": build_EpileptorDPrealistic,
    "EpileptorDP2D": build_EpileptorDP2D
}


EPILEPTOR_MODEL_NVARS = {
         "EpileptorModel": EpileptorModel._nvar,
         "Epileptor": Epileptor._nvar,
         "EpileptorDP": EpileptorDP._nvar,
         "EpileptorDPrealistic": EpileptorDPrealistic._nvar,
         "EpileptorDP2D": EpileptorDP2D._nvar
}


EPILEPTOR_MODEL_TAU1 = {
         "EpileptorModel": EpileptorModel.tt,
         "Epileptor": EpileptorDP().tau1,
         "EpileptorDP": EpileptorDP().tau1,
         "EpileptorDPrealistic": EpileptorDPrealistic().tau1,
         "EpileptorDP2D": EpileptorDP2D().tau1
}


EPILEPTOR_MODEL_TAU0 = {
         "EpileptorModel": 1.0 / EpileptorModel.r,
         "Epileptor": 1.0 / Epileptor().r,
         "EpileptorDP": EpileptorDP().tau0,
         "EpileptorDPrealistic": EpileptorDPrealistic().tau0,
         "EpileptorDP2D": EpileptorDP2D().tau0
}


model_noise_intensity_dict = {
    "Epileptor": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorModel":  1e-6,
    "EpileptorDP": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDPrealistic": numpy.array([0., 0., 1e-8, 0.0, 1e-8, 0., 1e-9, 1e-4, 1e-9, 1e-4, 1e-9]),
    "EpileptorDP2D": numpy.array([0., 1e-7])
}

VOIS = {
    "EpileptorModel": ['x1', 'z', 'x2'],
    "Epileptor": EpileptorDP().variables_of_interest.tolist(),
    "EpileptorDP": EpileptorDP().variables_of_interest.tolist(),
    "EpileptorDPrealistic": EpileptorDPrealistic().variables_of_interest.tolist(),
    "EpileptorDP2D": EpileptorDP2D().variables_of_interest.tolist()
}