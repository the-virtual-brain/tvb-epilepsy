
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string, shape_to_size
from tvb_epilepsy.base.h5_model import convert_to_h5_model


AVAILABLE_DISTRIBUTIONS = ["uniform", "normal", "gamma", "lognormal", "exponential", "beta", "chisquare",
                           "binomial", "bernoulli", "poisson"]


class ProbabilityDistribution(object):

    __metaclass__ = ABCMeta

    name = ""
    n_params = 0.0
    pdf_shape = ()
    constraint_string = ""
    mean = None
    median = None
    mode = None
    var = None
    std = None
    skew = None
    kurt = None
    scipy_name = ""
    numpy_name = ""

    @abstractmethod
    def __init__(self):
        pass

    def __repr__(self):
        d = {"1. name": self.name,
             "2. params": self.params(),
             "3. n_params": self.n_params,
             "4. constraint": self.constraint_string,
             "5. pdf shape": self.pdf_shape,
             "5. mean": self.mean,
             "6. median": self.median,
             "7. mode": self.mode,
             "8. var": self.var,
             "9. std": self.std,
             "10. skew": self.skew,
             "11. kurt": self.kurt,
             "12. scipy_name": self.scipy_name,
             "13. numpy_name": self.numpy_name}
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
        self.__set_params__(**params)
        self.pdf_shape = self.__calc_shape__()
        self.n_params = len(self.params())
        if not (self.__check_constraints__()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params()) + " is not satisfied!")
        self.mean = self.calc_mean()
        self.median = self.calc_median()
        self.mode = self.calc_mode_manual()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.kurt = self.calc_kurt()

    def __set_params__(self, **params):
        for p_key, p_val in params.iteritems():
            setattr(self, p_key, p_val)

    def __calc_shape__(self):
        psum = np.array(0.0)
        for pval in self.params().values():
            psum += np.array(pval, dtype='f')
        return psum.shape

    def __check_constraints__(self):
        return np.all(self.constraints() > 0)

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def update_params(self, **params):
        pass

    @abstractmethod
    def scipy(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def numpy(self, size=(1,)):
        pass

    @abstractmethod
    def calc_mean_manual(self):
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
    def calc_kurt_manual(self):
        pass

    def calc_mean(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.calc_mean_manual()

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

    def calc_kurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return self.calc_kurt_manual()

    def compute_distributions_params(self, constraints=[], target_shape=None, **target_stats):
        if len(target_stats) != self.n_params:
            raise_value_error("Target parameters are " + str(len(target_stats)) +
                              ", whereas the characteristic parameters of distribution " + self.name +
                              " are " + str(self.n_params) + "!")
        i1 = np.ones(self.pdf_shape)
        try:
            if isinstance(target_shape, tuple):
                i1 *= np.ones(target_shape)
        except:
            raise_value_error("Target (" + str(target_shape) +
                              ") and distribution (" + str(self.pdf_shape) + ") shapes do not match/propagate!")
        try:
            for ts in target_stats.values():
                i1 *= np.ones(np.array(ts).shape)
        except:
            raise_value_error("Target statistics (" + str([np.array(ts).shape for ts in target_stats.values()]) +
                              ") and distribution (" + str(self.pdf_shape) + ") shapes do not match/propagate!")
        shape = i1.shape
        size = shape_to_size(shape)
        for ts_key in target_stats.keys():
            target_stats[ts_key] *= i1
        params_vector = []
        bounds = []
        p_keys = self.params().keys()
        for p_key in p_keys:
            self.__set_params__(**{p_key: np.array(self.params()[p_key]) * i1})
            params_vector += self.params()[p_key].flatten().tolist()
            bounds += [self.bounds()[p_key]] * size
        params_vector = np.array(params_vector).astype(np.float64)

        def construct_params_dict(p):
            p = p.astype(np.float64)
            params = {}
            for ik, p_key in enumerate(p_keys):
                params.update({p_key: np.reshape(p[ik * size:(ik + 1) * size], shape)})
            return params

        # Scalar objective  function
        def fobj(p):
            params = construct_params_dict(p)
            f = 0.0
            self.update_params(**params)
            for ts_key, ts_val in target_stats.iteritems():
                f += (getattr(self, "calc_" + ts_key)(use="manual") - ts_val) ** 2
            return f

        # Vector constraints function
        def fconstr(p):
            params = construct_params_dict(p)
            self.update_params(**params)
            return self.constraints()

        # Vector valued constraints' functions
        constraints.append({"type": "ineq", "fun": lambda p: fconstr(p)})
        # TODO solve the problem for integer parameters...
        sol = minimize(fobj, params_vector, constraints=constraints)
        if sol.success:
            if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                raise_value_error("nan or inf values in solution x\n" + sol.message)
            if sol.fun > 10 ** -3:
                warning("Not accurate solution! sol.fun = " + str(sol.fun))
            return dict(zip(p_keys, sol.x))
        else:
            raise_value_error(sol.message)

    def compute_and_update_params(self, target_shape=None, **target_stats):
        params = self.compute_distributions_params(target_shape, **target_stats)
        self.update_params(**params)


def generate_distribution(distrib_name, constraints=[], target_stats=None, target_shape=None, **kwargs):
    if np.in1d(distrib_name.lower(), AVAILABLE_DISTRIBUTIONS):
        exec("from ." + distrib_name.lower() + "_distribution import " + distrib_name.title() + "Distribution")
        distribution = eval(distrib_name.title() + "Distribution(**kwargs)")
        if isinstance(distribution, ProbabilityDistribution) and isinstance(target_stats, dict):
            distribution.compute_and_update_params(constraints, target_shape, **target_stats)
        return distribution
    else:
        raise_value_error(distrib_name + " is not one of the available distributions!: " + str(AVAILABLE_DISTRIBUTIONS))

