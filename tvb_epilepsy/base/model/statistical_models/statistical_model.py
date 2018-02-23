from tvb_epilepsy.base.constants.model_inversion_constants import X1EQ_MIN, X1EQ_MAX, MC_SCALE
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
#TODO: avoid service imported in model
from tvb_epilepsy.service.stochastic_parameter_builder import set_parameter


class StatisticalModel(object):

    def __init__(self, name='vep', number_of_regions=0, x1eq_min=X1EQ_MIN, x1eq_max=X1EQ_MAX, MC_scale=MC_SCALE, **defaults):
        self.number_of_regions = number_of_regions
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Statistical model's type " + str(name) + " is not a string!")
        self.x1eq_min = x1eq_min
        self.x1eq_max = x1eq_max
        self.MC_scale = MC_scale
        # Parameter setting:
        self.parameters = {}
        self._generate_parameters(**defaults)
        self.n_parameters = len(self.parameters)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        d = {"1. type": self.name,
             "2. number of regions": self.number_of_regions,
             "3. number of parameters": self.n_parameters,
             "4. parameters": self.parameters}
        return formal_repr(self, sort_dict(d))

    def _generate_parameters(self, **defaults):
        for p in ["x1eq_star", "K", "tau1", "tau0", "MCsplit", "MC", "eps"]:
            self.parameters.update({p: set_parameter(p, **defaults)})
