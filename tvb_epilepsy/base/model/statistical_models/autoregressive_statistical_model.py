from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
from tvb_epilepsy.base.model.statistical_models.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import OdeStatisticalModel


class AutoregressiveStatisticalModel(OdeStatisticalModel):

    def __init__(self, name, parameters, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                       euler_method="forward", observation_model="seeg_logpower", observation_expression="x1z_offset"):

        super(AutoregressiveStatisticalModel, self).__init__(name, parameters, n_regions, active_regions, n_signals,
                                                             n_times, dt, euler_method, observation_model, observation_expression)

        # Further parameter setting:
        # State variables:
        self.parameters.append(Parameter("x1",
                                        low=parameters.get("x1_lo", -2.0),
                                        high=parameters.get("x1_hi", 2.0),
                                        loc=None,
                                        scale=None,
                                        shape=(self.n_times, self.n_active_regions),
                                        pdf="normal"))
        self.parameters.append(Parameter("z",
                                         low=parameters.get("z_lo", 2.0),
                                         high=parameters.get("z_hi", 5.0),
                                         loc=None,
                                         scale=None,
                                         shape=(self.n_times, self.n_active_regions),
                                         pdf="normal"))
        # Integration
        sig_def = parameters.get("sig_def", 10**-4)
        self.parameters.append(Parameter("sig",
                                         low=parameters.get("sig_lo", sig_def / 10.0),
                                         high=parameters.get("sig_hi", 10 * sig_def),
                                         loc=parameters.get("sig_loc", sig_def),
                                         scale=parameters.get("sig_sc", sig_def),
                                         shape=(1,),
                                         pdf=parameters.get("pdf", "gamma")))
        self.n_parameters = len(self.parameters)

