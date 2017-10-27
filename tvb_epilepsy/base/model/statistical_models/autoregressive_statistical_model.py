import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.model.statistical_models.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import OdeStatisticalModel
from tvb_epilepsy.base.h5_model import convert_to_h5_model





class AutoregressiveStatisticalModel(OdeStatisticalModel):

    def __init__(self, name, parameters, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                       euler_method="forward", observation_model="seeg_logpower",
                       observation_expression="x1z_offset"):

        super(OdeStatisticalModel, self).__init__(name, parameters, n_regions, active_regions, n_signals, n_times, dt,
                                                  euler_method, observation_model, observation_expression)

        # Integration
        sig_def = parameters.get("sig_def", 10**-4)
        self.parameters.append(Parameter("sig",
                                         low=parameters.get("sig_lo", sig_def / 10.0),
                                         high=parameters.get("sig_hi", 10 * sig_def),
                                         loc=parameters.get("sig_loc", sig_def),
                                         scale=parameters.get("sig_sc", sig_def),
                                         shape=(1,),
                                         pdf=parameters.get("pdf", "gamma")))

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
