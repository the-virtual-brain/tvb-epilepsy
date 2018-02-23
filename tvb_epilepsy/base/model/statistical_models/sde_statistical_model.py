
from tvb_epilepsy.base.constants.model_inversion_constants import X1EQ_MIN, X1EQ_MAX, MC_SCALE, SIG_INIT_DEF, SIG_DEF, \
                                                                                                OBSERVATION_MODEL_DEF
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
#TODO: avoid service imported in model
from tvb_epilepsy.service.stochastic_parameter_builder import set_parameter


class SDEStatisticalModel(ODEStatisticalModel):

    def __init__(self, name="sde_vep", number_of_regions=0, active_regions=[], n_signals=0,
                       n_times=0, dt=1.0, x1eq_min=X1EQ_MIN, x1eq_max=X1EQ_MAX, MC_scale=MC_SCALE,
                       sig_init=SIG_INIT_DEF, sig=SIG_DEF,
                        # observation_expression="lfp", euler_method="forward",
                       observation_model=OBSERVATION_MODEL_DEF,  **defaults):
        super(SDEStatisticalModel, self).__init__(name, number_of_regions, active_regions, n_signals, n_times, dt,
                                                  x1eq_min, x1eq_max, MC_scale, sig_init, observation_model,
                                                  #  observation_expression, euler_method,
                                                  **defaults)
        self.sig = sig
        self._add_parameters(**defaults)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(SDEStatisticalModel, self).__repr__()

    def _add_parameters(self, **defaults):
        for p in ["dX1t", "dZt", "sig"]:
            self.parameters.update({p: set_parameter(p, **defaults)})
