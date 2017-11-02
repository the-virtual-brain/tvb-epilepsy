
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import root

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class ProbabilityDistribution(object):

    __metaclass__ = ABCMeta

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

    @abstractmethod
    def __init__(self):
        pass

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

    def __update_params__(self, **params):
        self.params = params
        self.n_params = len(self.params)
        if not (self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.mu = self.calc_mu()
        self.median = self.calc_median()
        self.mode = self.calc_mode()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    @abstractmethod
    def update_params(self, **params):
        pass

    @abstractmethod
    def constraint(self):
        pass

    @abstractmethod
    def scipy(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_mu_manual(self):
        pass

    @abstractmethod
    def calc_median_manual(self):
        pass

    @abstractmethod
    def calc_mode_manual(self):
        pass

    @abstractmethod
    def calc_var_manual(self):
        pass

    @abstractmethod
    def calc_std_manual(self):
        pass

    @abstractmethod
    def calc_skew_manual(self):
        pass

    @abstractmethod
    def calc_exkurt_manual(self):
        pass

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.calc_mu_manual()

    def calc_median(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().median()
        else:
            return self.calc_median_manual()

    def calc_mode(self, use="scipy"):
        if isequal_string(use, "scipy"):
            warning("No scipy calculation for mode! Switching to manual -following wikipedia- calculation!")
        self.calc_mode_manual()

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            return self.calc_var_manual()

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return self.calc_std_manual()

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            return self.calc_skew_manual()

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return self.calc_exkurt_manual()

    def compute_distributions_params(self, target_stats, **kwargs):
        if len(target_stats) != self.n_params:
            raise_value_error("Target parameters are " + str(len(target_stats)) +
                              ", whereas the characteristic parameters of distribution " + self.name +
                              " are " + str(self.n_params) + "!")
        fobjs = []
        p_keys = self.params.keys()
        for p_key, p_val in target_stats.iteritems():
            fobjs.append(
          lambda p: (getattr(self.__class__.__init__(**dict(zip(p_keys, p))), "calc_" + p_key)(**kwargs)) - p_val) ** 2
        fobj = lambda p: np.sum([f(p) for f in fobjs])
        sol = root(fobj, self.params.values(), method='lm', tol=10 ** (-12), callback=None, options=None)
        if sol.success:
            if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                raise_value_error("nan or inf values in solution x\n" + sol.message)
            return dict(zip(p_keys, sol.x))
        else:
            raise_value_error(sol.message)

    def compute_and_update_params(self, **target_stats):
        params = self.compute_distributions_params(**target_stats)
        self.update_params(**params)
