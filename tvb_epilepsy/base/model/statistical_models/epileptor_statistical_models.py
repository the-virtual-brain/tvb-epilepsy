from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration


class StatisticalModel(object):

    name = "vep"
    number_of_regions = 0
    parameters = {}
    model_config = ModelConfiguration()

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    def __init__(self, name='vep', number_of_regions=0, parameters={}, model_config=ModelConfiguration()):
        self.name = name
        self.number_of_regions = number_of_regions
        self.parameters = parameters
        self.model_config = model_config

    def __repr__(self):
        d = OrderedDict()
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(ikey) + ". " + key: val})
        return d

    def __str__(self):
        return formal_repr(self, sort_dict(self.__repr__()))

    # Overwrite the following two methods for models with parameters that are not covered by this formulation
    def get_prior(self, parameter_name):
        return self.parameters.get(parameter_name).mean, self.parameters.get(parameter_name)

    def get_truth(self, parameter_name):
        truth = getattr(self, parameter_name, None)
        if truth is None:
            truth = getattr(self.model_config, parameter_name, None)
        if truth is None:
            raise_value_error("Ground truth value for parameter " + parameter_name + " was not found!")


class ODEStatisticalModel(StatisticalModel):

    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER
    number_of_signals = 0
    time_length = 0
    dt = DT_DEF
    active_regions = []
    nonactive_regions = []

    @property
    def number_of_active_regions(self):
        return len(self.active_regions)

    @property
    def number_of_nonactive_regions(self):
        return len(self.nonactive_regions)

    def __init__(self, name='vep_ode', number_of_regions=0, parameters={}, model_config=ModelConfiguration(),
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER, number_of_signals=0, time_length=0, dt=DT_DEF,
                 active_regions=[]):
        super(ODEStatisticalModel, self).__init__(name, number_of_regions, parameters, model_config)
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(active_regions).tolist()
            self.number_of_active_regions_regions = len(self.active_regions)
            self.n_nonactive_regions = self.number_of_regions - self.number_of_active_regions_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")
        self.number_of_signals = number_of_signals
        self.time_length = time_length
        self.dt = dt
        if observation_model in [model for model in OBSERVATION_MODELS]:
            self.observation_model = observation_model
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_model) +
                              " is not one of the valid ones: " + str(OBSERVATION_MODELS._member_map_.values()) + "!")

    def __repr__(self):
        d = OrderedDict()
        d.update(super(ODEStatisticalModel, self).__repr__())
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(nKeys+ikey) + ". " + key: val})
        return d

    def __str__(self):
        return formal_repr(self, sort_dict(self.__repr__()))

    def update_active_regions(self, active_regions):
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(ensure_list(active_regions) + self.active_regions).tolist()
            self.number_of_active_regions_regions = len(self.active_regions)
            self.n_nonactive_regions = self.number_of_regions - self.number_of_active_regions_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")

    def get_truth(self, parameter_name):
        return getattr(self.model_config, parameter_name, None)


class SDEStatisticalModel(ODEStatisticalModel):

    sde_mode = SDE_MODES.NONCENTERED

    def __init__(self, name='vep_ode', number_of_regions=0, parameters={}, model_config=ModelConfiguration(),
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER, number_of_signals=0, time_length=0, dt=DT_DEF,
                 active_regions=[], sde_mode=SDE_MODES.NONCENTERED):
        super(SDEStatisticalModel, self).__init__(name, number_of_regions, parameters, model_config,
                                                  observation_model, number_of_signals, time_length, dt, active_regions)
        self.sde_mode = sde_mode

    def __repr__(self):
        d = OrderedDict()
        d.update(super(SDEStatisticalModel, self).__repr__())
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(nKeys+ikey) + ". " + key: val})
        return d

    def __str__(self):
        return formal_repr(self, sort_dict(self.__repr__()))
