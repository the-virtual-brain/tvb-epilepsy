import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.statistical_model import StatisticalModel
from tvb_epilepsy.base.model.statistical_models.probability_distributions.gamma_distribution import GammaDistribution


OBSERVATION_MODEL_EXPRESSIONS=["x1z_offset", "x1_offset", "x1"]
OBSERVATION_MODELS=[ "seeg_logpower", "seeg_power", "lfp_power"]
EULER_METHODS = ["backward", "midpoint", "forward"]


class OdeStatisticalModel(StatisticalModel):

    def __init__(self, name, parameters, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                 euler_method="forward", observation_model="seeg_logpower", observation_expression="x1z_offset"):

        super(OdeStatisticalModel, self).__init__(name, parameters, n_regions)

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
        if np.in1d(euler_method.lower(), EULER_METHODS):
            if euler_method.lower() == "midpoint":
                warning("Midpoint Euler method is not implemented yet! Switching to default forward one!")
            self.euler_method = euler_method.lower()
        else:
            raise_value_error("Statistical model's euler_method " + str(euler_method) + " is not one of the valid ones: "
                              + str(["backward", "forward"]) + "!")
        if np.in1d(observation_expression.lower(), OBSERVATION_MODEL_EXPRESSIONS):
            self.observation_expression = observation_expression.lower()
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_expression) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODEL_EXPRESSIONS) + "!")
        if np.in1d(observation_model.lower(), OBSERVATION_MODELS):
            self.observation_model = observation_model.lower()
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_model) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODELS) + "!")

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
             "2. number of regions": self.n_regions,
             "3. active regions": self.active_regions,
             "4. number of active regions": self.n_active_regions,
             "5. number of nonactive regions": self.n_nonactive_regions,
             "6. number of observation signals": self.n_signals,
             "7. number of time points": self.n_times,
             "8. time step": self.dt,
             "9. euler_method": self.euler_method,
             "10. observation_expression": self.observation_expression,
             "11. observation_model": self.observation_model,
             "12. number of parameters": self.n_parameters,
             "13. parameters": self.parameters}
        return formal_repr(self, sort_dict(d))
