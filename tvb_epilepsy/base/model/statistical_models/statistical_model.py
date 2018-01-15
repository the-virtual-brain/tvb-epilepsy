from tvb_epilepsy.base.constants.model_inversion_constants import SIG_EQ_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, construct_import_path
from tvb_epilepsy.base.h5_model import convert_to_h5_model
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
        self.context_str = "from " + construct_import_path(__file__) + " import StatisticalModel"
        self.create_str = "StatisticalModel('" + self.name + "')"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        d = {"1. type": self.name,
             "2. number of regions": self.n_regions,
             "3. equilibrium point x1 std": self.sig_eq,
             "4. number of parameters": self.n_parameters,
             "5. parameters": self.parameters}
        return formal_repr(self, sort_dict(d))

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "StatisticalModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def _generate_parameters(self, **defaults):
        for p in ["x1eq", "K", "tau1", "tau0", "MCsplit",  "MC",  "eps"]: # "sig_eq",
            self.parameters.update({p: set_parameter(p, **defaults)})
