import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, K_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.statistical_models.statistical_model import StatisticalModel
from tvb_epilepsy.base.model.statistical_models.parameter import Parameter


OBSERVATION_MODEL_EXPRESSIONS=["x1z_offset", "x1_offset", "x1"]
OBSERVATION_MODELS=["seeg_power", "seeg_logpower", "lfp_power", "lfp_logpower"]


class OdeStatisticalModel(StatisticalModel):

    def __init__(self, name, parameters, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                 euler_method="forward", observation_model="seeg_logpower", observation_expression="x1z_offset"):

        super(StatisticalModel, self).__init__(name, parameters, n_regions)

        if np.all(np.in1d(active_regions, range(self.n_regions))):
            self.active_regions = np.unique(active_regions).tolist()
            self.n_active_regions = len(active_regions)
            self.n_nonactive_regions = self.n_regions - self.n_active_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.n_regions) + ")!")
        self.n_signals = n_signals
        self.n_times = n_times
        self.dt = dt
        if np.in1d(euler_method, ["backward", "forward"]):
            self.euler_method = euler_method
        else:
            raise_value_error("Statistical model's euler_method " + str(euler_method) + " is not one of the valid ones: "
                              + str(["backward", "forward"]) + "!")
        if np.in1d(observation_expression, OBSERVATION_MODEL_EXPRESSIONS):
            self.observation_expression = observation_expression
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_expression) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODEL_EXPRESSIONS) + "!")
        if np.in1d(observation_model, OBSERVATION_MODELS):
            self.observation_model = observation_model
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_model) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODELS) + "!")

        # Integration
        sig_init_def = parameters.get("sig_init_def", 0.1)
        self.parameters.append(Parameter("sig_init",
                                         low=parameters.get("sig_init_lo", sig_init_def / 10.0),
                                         high=parameters.get("sig_init_hi", 3 * sig_init_def),
                                         loc=parameters.get("sig_init_loc", sig_init_def),
                                         scale=parameters.get("sig_init_sc", sig_init_def),
                                         shape=(1,),
                                         pdf="gamma"))
        # Observation model
        self.parameters.append(Parameter("eps",
                                         low=parameters.get("eps_lo", 0.0),
                                         high=parameters.get("eps_hi", 1.0),
                                         loc=parameters.get("eps_loc", 0.1),
                                         scale=parameters.get("eps_sc", 0.1),
                                         shape=(1,),
                                         pdf="gamma"))
        self.parameters.append(Parameter("scale_signal",
                                         low=parameters.get("scale_signal_lo", 0.1),
                                         high=parameters.get("scale_signal_hi", 2.0),
                                         loc=parameters.get("scale_signal_loc", 1.0),
                                         scale=parameters.get("scale_signal", 1.0),
                                         shape=(1,),
                                         pdf="gamma"))
        self.parameters.append(Parameter("offset_signal",
                                         low=parameters.get("offset_signal_lo", 0.0),
                                         high=parameters.get("offset_signal_hi", 1.0),
                                         loc=parameters.get("offset_signal_loc", 0.0),
                                         scale=parameters.get("offset_signal", 0.1),
                                         shape=(1,),
                                         pdf="gamma"))

        self.n_parameters = len(self.parameters)

    def update_active_regions(self, active_regions):
        if np.all(np.in1d(active_regions, range(self.n_regions))):
            self.active_regions = np.unique(ensure_list(active_regions) + self.active_regions).tolist()
            self.n_active_regions = len(active_regions)
            self.n_nonactive_regions = self.n_regions - self.n_active_regions
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.n_regions) + ")!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. type": self.type,
             "3. number of regions": self.n_regions,
             "4. active regions": self.active_regions,
             "5. number of active regions": self.n_active_regions,
             "6. number of nonactive regions": self.n_nonactive_regions,
             "7. number of observation signals": self.n_signals,
             "8. number of time points": self.n_times,
             "9. euler_method": self.euler_method,
             "10. observation_expression": self.observation_expression,
             "11. observation_model": self.observation_model,
             "12. number of parameters": self.n_parameters,
             "13. parameters": [p.__str__ for p in self.parameters.items()]}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "StatisicalModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)
