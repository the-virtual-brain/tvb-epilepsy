# coding=utf-8
"""
Factory methods to build the wanted epileptor model.
Also, dictionaries to keep noise intensity and type for each model type.
"""

###
# Build TVB Epileptor
###
import numpy

from tvb_fit.tvb_epilepsy.service.simulator.simulator_java import JavaEpileptor
from tvb_fit.tvb_epilepsy.base.model.epileptor_models import EpileptorDP2D, EpileptorDP, EpileptorDPrealistic

from tvb_scripts.utils.log_error_utils import raise_value_error

from tvb.simulator.models import Epileptor


AVAILABLE_DYNAMICAL_MODELS = (Epileptor, JavaEpileptor, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)

AVAILABLE_DYNAMICAL_MODELS_NAMES = []
for model in AVAILABLE_DYNAMICAL_MODELS:
    AVAILABLE_DYNAMICAL_MODELS_NAMES.append(model._ui_name)

###
# Build TVB Epileptor
###
def build_tvb_model_from_model_config(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model_instance = Epileptor(x0=model_configuration.x0, Iext=model_configuration.Iext1, Iext2=model_configuration.Iext2,
                               Ks=K, c=model_configuration.yc, a=model_configuration.a, b=model_configuration.b,
                               d=model_configuration.d, aa=model_configuration.s,
                               tt=model_configuration.tau1, r=1.0/model_configuration.tau0)
    return model_instance


###
# Build EpileptorDP2D
###
def build_EpileptorDP2D_from_model_config(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP2D(x0=model_configuration.x0, Iext1=model_configuration.Iext1, K=K,
                          yc=model_configuration.yc, a=model_configuration.a, b=model_configuration.b,
                          d=model_configuration.d, tau1=model_configuration.tau1, tau0=model_configuration.tau0,
                          zmode=model_configuration.zmode)
    return model


###
# Build EpileptorDP
###
def build_EpileptorDP_from_model_config(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDP(x0=model_configuration.x0, Iext1=model_configuration.Iext1, Iext2=model_configuration.Iext2,
                        K=K, yc=model_configuration.yc, a=model_configuration.a,
                        b=model_configuration.b, d=model_configuration.d, s=model_configuration.s,
                        gamma=model_configuration.gamma, tau1=model_configuration.tau1, tau0=model_configuration.tau0,
                        zmode=model_configuration.zmode)
    return model


###
# Build EpileptorDPrealistic
###
def build_EpileptorDPrealistic_from_model_config(model_configuration):
    # We use the opposite sign for K with respect to all epileptor models
    K = -model_configuration.K
    model = EpileptorDPrealistic(x0=model_configuration.x0, Iext1=model_configuration.Iext1,
                                 Iext2=model_configuration.Iext2, K=K, yc=model_configuration.yc,
                                 a=model_configuration.a, b=model_configuration.b, d=model_configuration.d,
                                 s=model_configuration.s, gamma=model_configuration.gamma,
                                 tau1=model_configuration.tau1, tau0=model_configuration.tau0,
                                 zmode=model_configuration.zmode, pmode=model_configuration.pmode)
    return model

# Model creator functions dictionary (factory)
model_build_dict_from_model_config = {
    "Epileptor": build_tvb_model_from_model_config,
    "EpileptorDP": build_EpileptorDP_from_model_config,
    "EpileptorDPrealistic": build_EpileptorDPrealistic_from_model_config,
    "EpileptorDP2D":  build_EpileptorDP2D_from_model_config
}


def model_builder_from_model_config_fun(model_config, model_name=None):
    if not(isinstance(model_name, basestring)):
        model_name = model_config.model_name
    if model_name in AVAILABLE_DYNAMICAL_MODELS_NAMES:
        return model_build_dict_from_model_config[model_name](model_config)
    else:
        raise_value_error("Model name (%s) does not correspond to one of the available ones: %s"
                          % (str(input), str(AVAILABLE_DYNAMICAL_MODELS_NAMES)))


###
# Build TVB Epileptor
###
def build_tvb_model(**kwargs):
    # We use the opposite sign for K with respect to all epileptor models
    model_instance = Epileptor(**kwargs)
    return model_instance


###
# Build EpileptorDP2D
###
def build_EpileptorDP2D(**kwargs):
    # We use the opposite sign for K with respect to all epileptor models
    model_instance = Epileptor(**kwargs)
    return model_instance

###
# Build EpileptorDP
###
def build_EpileptorDP(**kwargs):
    # We use the opposite sign for K with respect to all epileptor models
    model_instance = Epileptor(**kwargs)
    return model_instance


###
# Build EpileptorDPrealistic
###
def build_EpileptorDPrealistic(**kwargs):
    # We use the opposite sign for K with respect to all epileptor models
    model_instance = Epileptor(**kwargs)
    return model_instance


# Model creator functions dictionary (factory)
model_build_dict= {
    "Epileptor": build_tvb_model,
    "EpileptorDP": build_EpileptorDP,
    "EpileptorDPrealistic": build_EpileptorDPrealistic,
    "EpileptorDP2D":  build_EpileptorDP2D
}


def model_builder_fun(model_name, **kwargs):
    if model_name in AVAILABLE_DYNAMICAL_MODELS_NAMES:
        return model_build_dict[model_name](**kwargs)
    else:
        raise_value_error("Model name (%s) does not correspond to one of the available ones: %s"
                          % (str(input), str(AVAILABLE_DYNAMICAL_MODELS_NAMES)))


EPILEPTOR_MODEL_NVARS = {
         "JavaEpileptor": JavaEpileptor._nvar,
         "Epileptor": Epileptor._nvar,
         "EpileptorDP": EpileptorDP._nvar,
         "EpileptorDPrealistic": EpileptorDPrealistic._nvar,
         "EpileptorDP2D": EpileptorDP2D._nvar
}


EPILEPTOR_MODEL_TAU1 = {
         "JavaEpileptor": JavaEpileptor.tt,
         "Epileptor": EpileptorDP().tau1,
         "EpileptorDP": EpileptorDP().tau1,
         "EpileptorDPrealistic": EpileptorDPrealistic().tau1,
         "EpileptorDP2D": EpileptorDP2D().tau1
}


EPILEPTOR_MODEL_TAU0 = {
         "JavaEpileptor": 1.0 / JavaEpileptor.r,
         "Epileptor": 1.0 / Epileptor().r,
         "EpileptorDP": EpileptorDP().tau0,
         "EpileptorDPrealistic": EpileptorDPrealistic().tau0,
         "EpileptorDP2D": EpileptorDP2D().tau0
}


model_noise_intensity_dict = {
    "Epileptor": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "JavaEpileptor": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
    "EpileptorDP": numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]),
                                                               # x0_t  K_t   slope_t Iext1_t Iext2_T
    "EpileptorDPrealistic": numpy.array([2.5e-4, 2.5e-4, 2e-6, 1e-7, 1e-7, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
    "EpileptorDP2D": numpy.array([0., 1e-7])
}

VOIS = {
    "JavaEpileptor": numpy.array(['x1', 'z', 'x2']),
    "Epileptor": EpileptorDP().variables_of_interest,
    "EpileptorDP": EpileptorDP().variables_of_interest,
    "EpileptorDPrealistic": EpileptorDPrealistic().variables_of_interest,
    "EpileptorDP2D": EpileptorDP2D().variables_of_interest
}