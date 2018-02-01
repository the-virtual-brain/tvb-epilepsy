
from tvb_epilepsy.base.constants.model_inversion_constants import SIG_EQ_DEF, OBSERVATION_MODEL_DEF, SIG_INIT_DEF
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


class SDEStatisticalModel(ODEStatisticalModel):

    def __init__(self, name="sde_vep", n_regions=0, active_regions=[], n_signals=0,
                       n_times=0, dt=1.0, sig_eq=SIG_EQ_DEF, sig_init=SIG_INIT_DEF,
                        # observation_expression="lfp", euler_method="forward",
                       observation_model=OBSERVATION_MODEL_DEF,  **defaults):
        super(SDEStatisticalModel, self).__init__(name, n_regions, active_regions, n_signals, n_times, dt,
                                                  sig_eq, sig_init, observation_model,
                                                  #  observation_expression, euler_method,
                                                  **defaults)
        self._add_parameters(**defaults)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(SDEStatisticalModel, self).__repr__()

    def _add_parameters(self, **defaults):
        for p in ["dX1t", "dZt", "sig"]:
            self.parameters.update({p: set_parameter(p, **defaults)})
