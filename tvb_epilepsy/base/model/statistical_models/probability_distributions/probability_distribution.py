
from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string, shape_to_size, \
    squeeze_array_to_scalar
from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.service.probability_distribution_factory import compute_pdf_params


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
        self._repr()

    def _repr(self):
        d = {"01. type": self.type,
             "02. pdf_params": self.pdf_params(),
             "03. n_params": self.n_params,
             "04. constraint": self.constraint_string,
             "05. shape": self.p_shape,
             "05. mean": self.mean,
             "06. median": self.median,
             "07. mode": self.mode,
             "08. var": self.var,
             "09. std": self.std,
             "10. skew": self.skew,
             "11. kurt": self.kurt,
             "12. scipy_name": self.scipy_name,
             "13. numpy_name": self.numpy_name}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self._repr()

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

    def __calc_shape__(self, params=None):
        if not(isinstance(params, dict)):
            params = self.pdf_params()
            p_shape = self.p_shape
        else:
            p_shape = ()
        psum = np.zeros(p_shape)
        for pval in params.values():
            psum = psum + np.array(pval, dtype='f')
        return psum.shape

    def __shape_parameters__(self, shape=None):
        if isinstance(shape, tuple):
            self.p_shape = shape
        i1 = np.ones(self.p_shape)
        for p_key in self.pdf_params().keys():
            setattr(self, p_key, getattr(self, p_key) * i1)
        self.__update_params__()

    def __squeeze_parameters__(self, update=False):
        params = self.pdf_params()
        for p_key, p_val in params.iteritems():
            params.update({p_key: squeeze_array_to_scalar(p_val)})
        if update:
            self.__set_params__(**params)
            self.__update_params__()
        return params

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
        return self.calc_mode_manual()

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
