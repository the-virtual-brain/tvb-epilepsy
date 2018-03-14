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
        self.model_config = ModelConfiguration()

    def __repr__(self):
        d = OrderedDict()
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(ikey) + ". " + key: val})
        return d

    def __str__(self):
        return formal_repr(self, sort_dict(self.__repr__()))


class ODEStatisticalModel(StatisticalModel):

    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER
    n_signals = 0
    n_times = 0
    dt = DT_DEF
    active_regions = []
    nonactive_regions = []

    @property
    def n_active(self):
        return len(self.active_regions)

    @property
    def non_active(self):
        return len(self.nonactive_regions)

    def __init__(self, name='vep_ode', number_of_regions=0, parameters={}, model_config=ModelConfiguration(),
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER, n_signals=0, n_times=0, dt=DT_DEF,
                 active_regions=[]):
        super(ODEStatisticalModel, self).__init__(name, number_of_regions, parameters, model_config)
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(active_regions).tolist()
            self.n_active_regions = len(self.active_regions)
            self.n_nonactive_regions = self.number_of_regions - self.n_active_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")
        self.n_signals = n_signals
        self.n_times = n_times
        self.dt = dt
        if observation_model in [model.value for model in OBSERVATION_MODELS]:
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
            self.n_active_regions = len(self.active_regions)
            self.n_nonactive_regions = self.number_of_regions - self.n_active_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")


class SDEStatisticalModel(ODEStatisticalModel):

    sde_mode = SDE_MODES.NONCENTERED

    def __init__(self, name='vep_ode', number_of_regions=0, parameters={}, model_config=ModelConfiguration(),
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER, n_signals=0, n_times=0, dt=DT_DEF,
                 active_regions=[], sde_mode=SDE_MODES.NONCENTERED):
        super(SDEStatisticalModel, self).__init__(name, number_of_regions, parameters, model_config,
                                                  observation_model, n_signals, n_times, dt, active_regions)
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
