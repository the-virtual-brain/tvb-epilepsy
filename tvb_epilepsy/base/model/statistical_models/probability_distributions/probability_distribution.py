from abc import ABCMeta, abstractmethod
import numpy as np
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string, shape_to_size, \
    squeeze_array_to_scalar
#TODO: avoid service imported in model
from tvb_epilepsy.service.probability_distribution_factory import compute_pdf_params


class ProbabilityDistribution(object):
    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    type = ""
    n_params = 0.0
    __p_shape = ()
    __p_size = 0
    constraint_string = ""
    __mean = None
    __median = None
    __mode = None
    __var = None
    __std = None
    __skew = None
    __kurt = None
    scipy_name = ""
    numpy_name = ""

    @property
    def mean(self):
        return self.__mean

    @property
    def median(self):
        return self.__median

    @property
    def mode(self):
        return self.__mode

    @property
    def var(self):
        return self.__var

    @property
    def std(self):
        return self.__std

    @property
    def skew(self):
        return self.__skew

    @property
    def kurt(self):
        return self.__kurt

    @property
    def p_shape(self):
        return self.__p_shape

    @property
    def p_size(self):
        return self.__p_size

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
             "05. shape": self.__p_shape,
             "05. mean": self.__mean,
             "06. median": self.__median,
             "07. mode": self.__mode,
             "08. var": self.__var,
             "09. std": self.__std,
             "10. skew": self.__skew,
             "11. kurt": self.__kurt,
             "12. scipy_name": self.scipy_name,
             "13. numpy_name": self.numpy_name}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self._repr()

    def __update_params__(self, loc=0.0, scale=1.0, use="scipy", check_constraint=True, **params):
        if len(params) == 0:
            params = self.pdf_params()
        self.__set_params__(**params)
        # params = self.__squeeze_parameters__(update=False, loc=loc, scale=scale, use=use)
        # self.__set_params__(**params)
        self.__p_shape = self.__update_shape__(loc, scale)
        self.__p_size = shape_to_size(self.p_shape)
        self.n_params = len(self.pdf_params())
        if check_constraint and not (self.__check_constraint__()):
            raise_value_error("Constraint for " + self.type + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.pdf_params()) + " is not satisfied!")
        self.__mean = self._calc_mean(loc, scale, use)
        self.__median = self._calc_median(loc, scale, use)
        self.__mode = self._calc_mode(loc, scale)
        self.__var = self._calc_var(loc, scale, use)
        self.__std = self._calc_std(loc, scale, use)
        self.__skew = self._calc_skew()
        self.__kurt = self._calc_kurt()

    def __set_params__(self, **params):
        for p_key, p_val in params.iteritems():
            setattr(self, p_key, p_val)

    def __check_constraint__(self):
        return np.all(self.constraint() > 0)

    def __update_shape__(self, loc=0.0, scale=1.0):
        try:
            shape = loc * scale * np.ones(self.p_shape)
            for p in self.pdf_params().values():
                shape *= p
            return self.p_shape
        except:
            return self.__calc_shape__(loc, scale)

    def __calc_shape__(self, loc=0.0, scale=1.0, params=None):
        if not (isinstance(params, dict)):
            params = self.pdf_params()
            p_shape = self.p_shape
        else:
            p_shape = ()
        psum = np.zeros(p_shape) * loc * scale
        for pval in params.values():
            psum = psum + np.array(pval, dtype='f')
        return psum.shape

    def __shape_parameters__(self, shape=None, loc=0.0, scale=1.0, use="scipy"):
        if isinstance(shape, tuple):
            self.__p_shape = shape
        i1 = np.ones((np.ones(self.p_shape) * loc * scale).shape)
        for p_key in self.pdf_params().keys():
            try:
                setattr(self, p_key, getattr(self, p_key) * i1)
            except:
                try:
                    setattr(self, p_key, np.reshape(getattr(self, p_key), self.p_shape))
                except:
                    raise_value_error("Neither propagation nor reshaping worked for distribution parameter " + p_key +
                                      " reshaping\nto shape " + str(self.p_shape) +
                                      "\nfrom shape " + str(getattr(self, p_key)) + "!")
        self.__update_params__(loc, scale, use)

    def __squeeze_parameters__(self, update=False, loc=0.0, scale=1.0, use="scipy"):
        params = self.pdf_params()
        for p_key, p_val in params.iteritems():
            params.update({p_key: squeeze_array_to_scalar(p_val)})
        if update:
            self.__set_params__(**params)
            self.__update_params__(loc, scale, use)
        return params

    @abstractmethod
    def pdf_params(self):
        pass

    @abstractmethod
    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        pass

    @abstractmethod
    def scipy(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def constraint(self):
        pass

    @abstractmethod
    def numpy(self, loc=0.0, scale=1.0, size=()):
        pass

    @abstractmethod
    def calc_mean_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_median_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_mode_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_var_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_std_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_skew_manual(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        pass

    def _calc_mean(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).stats(moments="m")
        else:
            return self.calc_mean_manual(loc, scale)

    def _calc_median(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).median()
        else:
            return self.calc_median_manual(loc, scale)

    def _calc_mode(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            self.logger.warning("No scipy calculation for mode! Switching to manual -following wikipedia- calculation!")
        return self.calc_mode_manual(loc, scale)

    def _calc_var(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).var()
        else:
            return self.calc_var_manual(loc, scale)

    def _calc_std(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).std()
        else:
            return self.calc_std_manual(loc, scale)

    def _calc_skew(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).stats(moments="s")
        else:
            return self.calc_skew_manual(loc, scale)

    def _calc_kurt(self, loc=0.0, scale=1.0, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy(loc, scale).stats(moments="k")
        else:
            return self.calc_kurt_manual(loc, scale)

    def compute_and_update_pdf_params(self, loc=0.0, scale=1.0, use="scipy", **target_stats):
        self.update_params(loc, scale, use, **(compute_pdf_params(self.type, target_stats, loc, scale, use)))
