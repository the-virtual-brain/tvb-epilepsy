from enum import Enum

from tvb_fit.tvb_epilepsy.base.model.epileptor_probabilistic_models \
    import EpiProbabilisticModel, ODEEpiProbabilisticModel, SDEEpiProbabilisticModel

# TODO: find a better solution here so that there is no reference from tvb_epilepsy to tvb_fit


class ProbabilisticModels(Enum):
    EPI_PROBABILISTIC_MODEL = {"name": EpiProbabilisticModel().__class__.__name__,
                               "instance": EpiProbabilisticModel()}
    ODE_EPI_PROBABILISTIC_MODEL = {"name": ODEEpiProbabilisticModel().__class__.__name__,
                                   "instance": ODEEpiProbabilisticModel()}
    SDE_EPI_PROBABILISTIC_MODEL = {"name": SDEEpiProbabilisticModel().__class__.__name__,
                                   "instance": SDEEpiProbabilisticModel()}