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

        # Further parameter setting:
        # State variables:
        self.parameters.append(parameters.get("x1", Parameter("x1",
                                                                low=parameters.get("x1_lo", -2.0),
                                                                high=parameters.get("x1_hi", 2.0),
                                                                probability_distribution=
                                                                    parameters.get("x1_pdf", NormalDistribution()),
                                                                shape=(self.n_times, self.n_active_regions))))
        self.parameters.append(parameters.get("z", Parameter("z",
                                                              low=parameters.get("z_lo", 2.0),
                                                              high=parameters.get("z_hi", 5.0),
                                                              probability_distribution=
                                                                parameters.get("z_pdf", NormalDistribution()),
                                                              shape=(self.n_times, self.n_active_regions))))

        # Integration
        parameter = parameters.get("sig")
        if parameter is None:
            sig_def = parameters.get("sig_def", 10**-4)
            probability_distribution = parameters.get("sig_pdf")
            if probability_distribution is None:
                probability_distribution = GammaDistribution()
                probability_distribution.compute_and_update_params({"mode": sig_def,
                                                                    "std": parameters.get("sig_sig", sig_def)})
                parameter = Parameter("sig",
                                      low=parameters.get("sig_lo", sig_def / 10.0),
                                      high=parameters.get("sig_hi", 10 * sig_def),
                                      probability_distribution=probability_distribution,
                                      shape=(1,))
        self.parameters.append(parameter)

        self.n_parameters = len(self.parameters)

