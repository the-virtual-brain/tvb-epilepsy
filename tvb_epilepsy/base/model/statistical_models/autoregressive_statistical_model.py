from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import OdeStatisticalModel
from tvb_epilepsy.base.model.statistical_models.probability_distributions.normal_distribution import NormalDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.gamma_distribution import GammaDistribution


class AutoregressiveStatisticalModel(OdeStatisticalModel):

    def __init__(self, name, parameters, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                       euler_method="forward", observation_model="seeg_logpower", observation_expression="x1z_offset"):

        super(AutoregressiveStatisticalModel, self).__init__(name, parameters, n_regions, active_regions, n_signals,
                                                             n_times, dt, euler_method, observation_model,
                                                             observation_expression)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(AutoregressiveStatisticalModel, self).__repr__()
