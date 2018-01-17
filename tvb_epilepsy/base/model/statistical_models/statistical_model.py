
import matplotlib.pyplot as pl
from tvb_epilepsy.base.constants.configurations import SAVE_FLAG, SHOW_FLAG, FOLDER_FIGURES, FIG_FORMAT, \
                                                                                                    VERY_LARGE_PROTRAIT
from tvb_epilepsy.base.constants.model_inversion_constants import SIG_EQ_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
from tvb_epilepsy.base.utils.plot_utils import save_figure, check_show
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


class StatisticalModel(object):

    def __init__(self, name='vep', n_regions=0, sig_eq=SIG_EQ_DEF, **defaults):
        self.n_regions = n_regions
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Statistical model's type " + str(name) + " is not a string!")
        self.sig_eq = sig_eq
        # Parameter setting:
        self.parameters = {}
        self._generate_parameters(**defaults)
        self.n_parameters = len(self.parameters)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        d = {"1. type": self.name,
             "2. number of regions": self.n_regions,
             "3. equilibrium point x1 std": self.sig_eq,
             "4. number of parameters": self.n_parameters,
             "5. parameters": self.parameters}
        return formal_repr(self, sort_dict(d))

    def _generate_parameters(self, **defaults):
        for p in ["x1eq", "K", "tau1", "tau0", "MCsplit",  "MC", "sig_eq", "eps"]:
            self.parameters.update({p: set_parameter(p, **defaults)})

    def plot(self, figure_name="", figure_dir=FOLDER_FIGURES,
             save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_format=FIG_FORMAT):
        _, ax = pl.subplots(len(self.parameters), 2, figsize=VERY_LARGE_PROTRAIT)
        # ax = []
        for ip, p in enumerate(self.parameters.values()):
            ax[ip] = p.plot(ax=ax[ip], lgnd=False)
            # ax.append(p.plot_stochastic_parameter(lgnd=False))
        save_figure(save_flag, pl.gcf(), figure_name, figure_dir, figure_format)
        check_show(show_flag)
        return ax, pl.gcf()

