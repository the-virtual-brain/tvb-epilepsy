
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel


class SDEStatisticalModel(ODEStatisticalModel):

    def __init__(self, name="sde_vep", parameters={}, n_regions=0, active_regions=[], n_signals=0, n_times=0, dt=1.0,
                       euler_method="forward", observation_model="seeg_logpower", observation_expression="x1z_offset",
                 **kwargs):
        super(SDEStatisticalModel, self).__init__(name, parameters, n_regions, active_regions, n_signals,
                                                  n_times, dt, euler_method, observation_model,
                                                  observation_expression, **kwargs)
        self.context_str = "from " + construct_import_path(__file__) + " import SDEStatisticalModel"
        self.create_str = "SDEStatisticalModel('" + self.name + "')"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return super(SDEStatisticalModel, self).__repr__()

    # def plot(self):
    #     figure_dir = os.path.join(FOLDER_FIGURES, "_ASM_" + self.name)
