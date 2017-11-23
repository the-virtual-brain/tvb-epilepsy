import importlib

from abc import ABCMeta


from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, construct_import_path
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
                                                    ProbabilityDistribution
from tvb_epilepsy.service.probability_distribution_factory import compute_pdf_params


class StochasticParameterBase(Parameter, ProbabilityDistribution):
    __metaclass__ = ABCMeta

    def __init__(self, name="Parameter", low=MIN_SINGLE_VALUE, high=MAX_SINGLE_VALUE, p_shape=(), **pdf_params):
        Parameter.__init__(self, name, low, high, p_shape)
        self.context_str = "from " + construct_import_path(__file__) + " import StochasticParameterBase"
        self.create_str = "StochasticParameterBase('" + self.name + "')"
        self.update_str = "obj.update_params()"

    def __repr__(self):
        d = {"01. name": self.name,
             "02. low": self.low,
             "03. high": self.high,
             }
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        return Parameter._prepare_for_h5(self)

    def write_to_h5(self, folder, filename=""):
        Parameter.write_to_h5(self, folder, filename)

    def _string_generator(self):
        exec_str = "from tvb_epilepsy.base.model.statistical_models.stochastic_parameter " + \
                   "import generate_stochastic_parameter"
        eval_str = "generate_stochastic_parameter(" + self.name + \
                   ", probability_distribution=" + self.type + \
                   ", optimize=False)"
        d = {"exec": exec_str, "eval": eval_str}
        return d


def generate_stochastic_parameter(name="Parameter", low=-MAX_SINGLE_VALUE, high=MAX_SINGLE_VALUE, p_shape=(),
                                  probability_distribution="uniform", optimize=False, **pdf_params):
    pdf_module = importlib.import_module("tvb_epilepsy.base.model.statistical_models.probability_distributions." +
                                  probability_distribution.lower() + "_distribution")
    thisProbabilityDistribution = eval("pdf_module." + probability_distribution.title() + "Distribution")
    if optimize:
        pdf_params = compute_pdf_params(probability_distribution.lower(), pdf_params)

    class StochasticParameter(StochasticParameterBase, thisProbabilityDistribution):
        def __init__(self, name="Parameter", low=-MAX_SINGLE_VALUE, high=MAX_SINGLE_VALUE, p_shape=(), **pdf_params):
            StochasticParameterBase.__init__(self, name, low, high, p_shape)
            thisProbabilityDistribution.__init__(self, **pdf_params)
            self.context_str = "from " + construct_import_path(__file__) + " import generate_stochastic_parameter"
            self.create_str = "generate_stochastic_parameter('" + str(self.name) + \
                              "', probability_distribution='" + str(self.type) + "', optimize=False)"
            self.update_str = "obj.update_params()"

        def __str__(self):
            return StochasticParameterBase.__str__(self) + "\n" \
                   + "\n".join(thisProbabilityDistribution.__str__(self).splitlines()[1:])

    return StochasticParameter(name, low, high, p_shape, **pdf_params)




if __name__ == "__main__":
    sp = generate_stochastic_parameter("test", probability_distribution="gamma", optimize=False, shape=1.0, scale=2.0)
    print(sp)