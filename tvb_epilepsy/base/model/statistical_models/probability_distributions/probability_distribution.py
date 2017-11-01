
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class ProbabilityDistribution(object):

    name = ""
    params = {}
    n_params = 0.0
    constraint_string = ""
    mu = None
    median = None
    mode = None
    var = None
    std = None
    skew = None
    exkurt = None
    scipy_name = ""

    def __repr__(self):
        d = {"1. name": self.name,
             "2. params": self.params,
             "3. n_params": self.n_params,
             "4. constraint": self.constraint_string,
             "5. mu": self.mu,
             "6. median": self.median,
             "7. mode": self.mode,
             "8. var": self.var,
             "9. std": self.std,
             "10. skew": self.skew,
             "11. exkurt": self.exkurt,
             "12. scipy_name": self.scipy_name}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "ProbabilityDistributionModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

