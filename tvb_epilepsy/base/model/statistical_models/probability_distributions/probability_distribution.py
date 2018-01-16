from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as pl
from tvb_epilepsy.base.constants.configurations import SAVE_FLAG, SHOW_FLAG, FOLDER_FIGURES, FIG_FORMAT
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string, shape_to_size, \
    squeeze_array_to_scalar
from tvb_epilepsy.base.utils.plot_utils import save_figure, check_show
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

    def __update_params__(self, loc=0.0, scale=1.0, use="scipy", check_constraint=True, **params):
        if len(params) == 0:
            params = self.pdf_params()
        self.__set_params__(**params)
        # params = self.__squeeze_parameters__(update=False, loc=loc, scale=scale, use=use)
        self.__set_params__(**params)
        self.p_shape = self.__update_shape__(loc, scale)
        self.p_size = shape_to_size(self.p_shape)
        self.n_params = len(self.pdf_params())
        if check_constraint and not (self.__check_constraint__()):
            raise_value_error("Constraint for " + self.type + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.pdf_params()) + " is not satisfied!")
        self.mean = self._calc_mean(loc, scale, use)
        self.median = self._calc_median(loc, scale, use)
        self.mode = self._calc_mode(loc, scale)
        self.var = self._calc_var(loc, scale, use)
        self.std = self._calc_std(loc, scale, use)
        self.skew = self._calc_skew()
        self.kurt = self._calc_kurt()

    def __set_params__(self, **params):
        for p_key, p_val in params.iteritems():
            setattr(self, p_key, p_val)

    def __check_constraint__(self):
        return np.all(self.constraint() > 0)

    def __update_shape__(self, loc=0.0, scale=1.0):
        try:
            shape = loc * scale
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
            self.p_shape = shape
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
            warning("No scipy calculation for mode! Switching to manual -following wikipedia- calculation!")
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

    def _plot(self, loc=0.0, scale=1.0, x=np.array([]), ax=None, linestyle="-", lgnd=True):
        if len(x) < 1:
            x = np.linspace(self.scipy(loc, scale).ppf(0.01), self.scipy(loc, scale).ppf(0.99), 100)
        pdf = None
        while pdf is None:
            try:
                pdf = self.scipy(loc, scale).pdf(x)
            except:
                x = x[:, np.newaxis]
        x = np.tile(x, self.p_shape)
        if ax is None:
            _, ax = pl.subplots(1,1)
        for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
            ax.plot(xx.T, pp.T, linestyle=linestyle, linewidth=1, label=str(ip))
        if lgnd:
            pl.legend()
        return ax

    def plot_distribution(self, loc=0.0, scale=1.0, x=np.array([]), ax=None, linestyle="-", lgnd=True,
                          figure_name="", figure_dir=FOLDER_FIGURES, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
                          figure_format=FIG_FORMAT):
        ax = self._plot(loc, scale, x, ax, linestyle, lgnd)
        ax.set_title(self.type + " distribution")
        save_figure(save_flag, pl.gcf(), figure_name, figure_dir, figure_format)
        check_show(show_flag)
        return ax, pl.gcf()

