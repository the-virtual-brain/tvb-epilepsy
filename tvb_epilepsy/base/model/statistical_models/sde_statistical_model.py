
import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


class SDEStatisticalModel(ODEStatisticalModel):

    def __init__(self, name="sde_vep", n_regions=0, active_regions=[], n_signals=0,
                       n_times=0, dt=1.0, euler_method="forward", observation_model="seeg_logpower",
                       observation_expression="x1z_offset", x1var="x1", zvar="z", **defaults):
        super(SDEStatisticalModel, self).__init__(name, n_regions, active_regions, n_signals, n_times, dt, euler_method,
                                                  observation_model, observation_expression, **defaults)
        self._add_parameters(x1var, zvar, **defaults)
        self.context_str = "from " + construct_import_path(__file__) + " import SDEStatisticalModel"
        self.create_str = "SDEStatisticalModel('" + self.name + "')"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(SDEStatisticalModel, self).__repr__()

    def _add_parameters(self, x1var="x1", zvar="z", **defaults):
        for p in [x1var, zvar]:
            self.parameters.update({p: set_parameter(p, optimize_pdf=False, **defaults)})
        self.parameters.update({"sig": set_parameter("sig", optimize_pdf=False, **defaults)})

    # def plot(self):
    #     figure_dir = os.path.join(FOLDER_FIGURES, "_ASM_" + self.name)
