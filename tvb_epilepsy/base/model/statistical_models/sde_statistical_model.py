
import numpy as np

from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF, X1_DEF
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


SIG_EQ_DEF = (X1_EQ_CR_DEF-X1_DEF)/10
SIG_INIT_DEF = SIG_EQ_DEF

class SDEStatisticalModel(ODEStatisticalModel):

    def __init__(self, name="sde_vep", n_regions=0, active_regions=[], n_signals=0,
                       n_times=0, dt=1.0, sig_eq=SIG_EQ_DEF, sig_init=SIG_INIT_DEF, # euler_method="forward",
                       observation_model="seeg_logpower", observation_expression="x1z_offset", **defaults):
        super(SDEStatisticalModel, self).__init__(name, n_regions, active_regions, n_signals, n_times, dt,
                                                  sig_eq, sig_init, observation_model, observation_expression,
                                                  **defaults) # euler_method,
        self._add_parameters(**defaults)
        self.context_str = "from " + construct_import_path(__file__) + " import SDEStatisticalModel"
        self.create_str = "SDEStatisticalModel('" + self.name + "')"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(SDEStatisticalModel, self).__repr__()

    def _add_parameters(self, **defaults):
        for p in ["z_dWt", "sig"]: #"x1_dWt",
            self.parameters.update({p: set_parameter(p, **defaults)})

    # def plot(self):
    #     figure_dir = os.path.join(FOLDER_FIGURES, "_ASM_" + self.name)
