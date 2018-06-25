import numpy
from copy import deepcopy
from tvb_infer.tvb_lsa.lsa_config import CalculusConfig
from tvb_infer.tvb_epilepsy.base.constants.model_constants import *
from tvb_infer.base.utils.log_error_utils import raise_not_implemented_error
from tvb_infer.base.utils.data_structures_utils import formal_repr
from tvb_infer.tvb_epilepsy.service.pse.pse_service import ABCPSEService
from tvb_infer.tvb_lsa.lsa_service import LSAService


class LSAPSEService(ABCPSEService):
    task = "LSA"
    hypothesis = None

    def __init__(self, hypothesis=None, params_pse=None):
        super(LSAPSEService, self).__init__()
        self.hypothesis = hypothesis
        self.params_pse = params_pse
        self.prepare_params(params_pse)

    def __repr__(self):
        d = {"01. Task": self.task,
             "02. Main PSE object": self.hypothesis,
             "03. Parameters": numpy.array(["%s" % l for l in self.params_names]),
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def run_pse_parallel(self):
        raise_not_implemented_error("PSE parallel not implemented!", self.logger)

    def run(self, params, conn_matrix, model_config_service_input=None, lsa_service_input=None,
            yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF, x1eq_mode="optimize",
            lsa_method=CalculusConfig.LSA_METHOD, n_eigenvectors=CalculusConfig.EIGENVECTORS_NUMBER_SELECTION,
            weighted_eigenvector_sum=CalculusConfig.WEIGHTED_EIGENVECTOR_SUM):
        #try:
        # Copy and update hypothesis
        hypo_copy, model_configuration = self.update_hypo_model_config(self.hypothesis, params, conn_matrix,
                                                                       model_config_service_input, yc, Iext1, K, a,
                                                                       b, tau1, tau0, x1eq_mode)
        # Copy a LSAService and update it
        # ...create/update lsa service:
        if isinstance(lsa_service_input, LSAService):
            lsa_service = deepcopy(lsa_service_input)
        else:
            lsa_service = LSAService(lsa_method=lsa_method, eigen_vectors_number=n_eigenvectors,
                                     weighted_eigenvector_sum=weighted_eigenvector_sum)
        lsa_service.update_for_pse(params, self.params_paths, self.params_indices)
        lsa_hypothesis = lsa_service.run_lsa(hypo_copy, model_configuration)
        output = self.prepare_run_results(lsa_hypothesis, model_configuration)
        return True, output
       # except:
      #      return False, None

    def prepare_run_results(self, lsa_hypothesis, model_configuration=None):
        if model_configuration is None:
            return {"lsa_propagation_strengths": lsa_hypothesis.propagation_strenghts}

        return {"lsa_propagation_strengths": lsa_hypothesis.lsa_propagation_strengths,
                "x0_values": model_configuration.x0_values,
                "e_values": model_configuration.e_values, "x1eq": model_configuration.x1eq,
                "zeq": model_configuration.zeq, "Ceq": model_configuration.Ceq}
