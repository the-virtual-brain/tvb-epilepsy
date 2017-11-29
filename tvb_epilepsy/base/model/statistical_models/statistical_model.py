
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, construct_import_path
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


class StatisticalModel(object):

    def __init__(self, name='vep', n_regions=0, **defaults):
        self.n_regions = n_regions
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Statistical model's type " + str(name) + " is not a string!")
        # Parameter setting:
        self.parameters = {}
        self.__generate_parameters(**defaults)
        self.n_parameters = len(self.parameters)
        self.context_str = "from " + construct_import_path(__file__) + " import StatisticalModel"
        self.create_str = "StatisticalModel('" + self.name + "')"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        d = {"1. type": self.name,
             "2. number of regions": self.n_regions,
             "3. number of parameters": self.n_parameters,
             "4. parameters": self.parameters}
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

    def __generate_parameters(self, **defaults):
        self.parameters.update({"x1eq": set_parameter("x1eq", optimize=False, **defaults)})
        for p in ["K", "tau1", "tau0", "MC", "sig_eq", "eps"]:
            self.parameters.update({p: set_parameter(p, optimize=True, **defaults)})
