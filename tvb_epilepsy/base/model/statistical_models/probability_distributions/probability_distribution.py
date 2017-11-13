
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string, shape_to_size
from tvb_epilepsy.base.h5_model import convert_to_h5_model


AVAILABLE_DISTRIBUTIONS = ["uniform", "normal", "gamma", "lognormal", "exponential", "beta", "chisquare",
                           "binomial", "bernoulli", "poisson"]


class ProbabilityDistribution(object):

    __metaclass__ = ABCMeta

    type = ""
    n_params = 0.0
    p_shape = ()
    p_size = 0
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
        d = {"1. type": self.type,
             "2. pdf_params": self.pdf_params(),
             "3. n_params": self.n_params,
             "4. constraint": self.constraint_string,
             "5. shape": self.p_shape,
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
            filename = self.type + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def __update_params__(self, check_constraint=True, **params):
        if len(params) == 0:
            params = self.pdf_params()
        self.__set_params__(**params)
        self.p_shape = self.__calc_shape__()
        self.p_size = shape_to_size(self.p_shape)
        self.n_params = len(self.pdf_params())
        if check_constraint and not(self.__check_constraint__()):
            raise_value_error("Constraint for " + self.type + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.pdf_params()) + " is not satisfied!")
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

    def __check_constraint__(self):
        return np.all(self.constraint() > 0)

    def __calc_shape__(self):
        psum = np.zeros(self.p_shape)
        for pval in self.pdf_params().values():
            psum += np.array(pval, dtype='f')
        return psum.shape

    def __shape_parameters__(self, shape=None):
        if isinstance(shape, tuple):
            self.p_shape = shape
        i1 = np.ones(self.p_shape)
        for p_key in self.pdf_params().keys():
            setattr(self, p_key, getattr(self, p_key) * i1)
        self.__update_params__()

    @abstractmethod
    def pdf_params(self):
        pass

    @abstractmethod
    def update_params(self, **params):
        pass

    @abstractmethod
    def scipy(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def constraint(self):
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

    def compute_and_update_pdf_params(self, target_shape=None, **target_stats):
        if target_shape is None:
            target_shape = self.p_shape
        self.update_params(**(compute_pdf_params(self.type, target_stats, target_shape)))


def generate_distribution(distrib_type, target_shape=None, **target):
    if np.in1d(distrib_type.lower(), AVAILABLE_DISTRIBUTIONS):
        exec("from ." + distrib_type.lower() + "_distribution import " + distrib_type.title() + "Distribution")
        distribution = eval(distrib_type.title() + "Distribution()")
        if isinstance(target_shape, tuple):
            distribution.__shape_parameters__(target_shape)
        if len(target) > 0:
            try:
                distribution.update(**target)
            except:
                target = compute_pdf_params(distribution.type, target, target_shape)
                distribution.update_params(**target)
        return distribution
    else:
        raise_value_error(distrib_type + " is not one of the available distributions!: " + str(AVAILABLE_DISTRIBUTIONS))


# This function converts the parameters' vector to the parameters' dictionary
def construct_pdf_params_dict(p, pdf):
    # Make sure p in denormalized and of float64 type
    # TODO solve the problem for integer distribution parameters...
    p = p.astype(np.float64)
    params = {}
    for ik, p_key in enumerate(pdf.pdf_params().keys()):
        params.update({p_key: np.reshape(p[ik * pdf.p_size:(ik + 1) * pdf.p_size], pdf.p_shape)})
    return params


# Scalar objective  function
def fobj(p, pdf, target_stats):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(check_constraint=False, **params)
    f = 0.0
    for ts_key, ts_val in target_stats.iteritems():
        f += (getattr(pdf, "calc_" + ts_key)(use="manual") - ts_val) ** 2
    return f


# Vector constraint function. By default expr >= 0
def fconstr(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(check_constraint=False, **params)
    return pdf.constraint()


def compute_pdf_params(distrib_type, target_stats, target_shape=None):
    distribution = generate_distribution(distrib_type, target_shape=target_shape)
    # Check if the number of target stats is exactly the same as the number of distribution parameters to optimize:
    if len(target_stats) != distribution.n_params:
        raise_value_error("Target parameters are " + str(len(target_stats)) +
                          ", whereas the characteristic parameters of distribution " + distribution.type +
                          " are " + str(distribution.n_params) + "!")
    # Make sure that tha shapes of distribution, target stats and target p_shape are all matching one to the other:
    i1 = np.ones(distribution.p_shape)
    try:
        if isinstance(target_shape, tuple):
            i1 *= np.ones(target_shape)
    except:
        raise_value_error("Target (" + str(target_shape) +
                          ") and distribution (" + str(distribution.p_shape) + ") shapes do not propagate!")
    try:
        for ts in target_stats.values():
            i1 *= np.ones(np.array(ts).shape)
    except:
        raise_value_error("Target statistics (" + str([np.array(ts).shape for ts in target_stats.values()]) +
                          ") and distribution (" + str(distribution.p_shape) + ") shapes do not propagate!")
    for ts_key in target_stats.keys():
        target_stats[ts_key] *= i1
    if distribution.p_shape != i1.shape:
        distribution.__shape_parameters__(i1.shape)
    # Preparing initial conditions' parameters' vector:
    params_vector = []
    for p_val in distribution.pdf_params().values():
        params_vector += p_val.flatten().tolist()
    params_vector = np.array(params_vector).astype(np.float64)
    # Bounding initial condition:
    params_vector[np.where(params_vector > 10.0)[0]] = 10.0
    params_vector[np.where(params_vector < -10.0)[0]] = -10.0
    # Preparing contraints:
    constraints = {"type": "ineq", "fun": lambda p: fconstr(p, distribution)}
    # Run optimization
    sol = minimize(fobj, params_vector, args=(distribution, target_stats), method="COBYLA",
                   constraints=constraints, options={"tol": 10 ** -6, "catol": 10 ** -12})
    if sol.success:
        if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
            raise_value_error("nan or inf values in solution x\n" + sol.message)
        if sol.fun > 10 ** -6:
            warning("Not accurate solution! sol.fun = " + str(sol.fun))
        return construct_pdf_params_dict(sol.x, distribution)
    else:
        raise_value_error(sol.message)

