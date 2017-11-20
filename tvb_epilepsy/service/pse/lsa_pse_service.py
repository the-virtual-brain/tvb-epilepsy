
from copy import deepcopy

import numpy

from tvb_epilepsy.base.utils.log_error_utils import raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr
from tvb_epilepsy.service.pse.pse_service import ABCPSEService


class LSAPSEService(ABCPSEService):
    hypothesis = None

    def __init__(self, hypothesis=None, params_pse=None):
        self.hypothesis = hypothesis
        self.params_pse = params_pse
        self.prepare_params(params_pse)

    def __repr__(self):
        d = {"01. Task": "LSA",
             "02. Main PSE object": self.hypothesis,
             "03. Number of computation loops": self.n_loops,
             "04. Parameters": numpy.array(["%s" % l for l in self.params_names]),
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def run_pse_parallel(self):
        raise_not_implemented_error("PSE parallel not implemented!", self.logger)

    def run(self, conn_matrix, params, lsa_service, model_config_service):
        try:
            # Copy and update hypothesis
            hypo_copy = deepcopy(self.hypothesis)
            hypo_copy.update_for_pse(params, self.params_paths, self.params_indices)
            # Create a ModelConfigService and update it
            model_configuration_service = deepcopy(model_config_service)
            model_configuration_service.update_for_pse(params, self.params_paths, self.params_indices)
            # Obtain Modelconfiguration
            if hypo_copy.type == "Epileptogenicity":
                model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hypo_copy,
                                                                                                    conn_matrix)
            else:
                model_configuration = model_configuration_service.configure_model_from_hypothesis(hypo_copy,
                                                                                                  conn_matrix)
            # Copy a LSAService and update it
            lsa_service = deepcopy(lsa_service)
            lsa_service.update_for_pse(params, self.params_paths, self.params_indices)
            lsa_hypothesis = lsa_service.run_lsa(hypo_copy, model_configuration)
            output = self.prepare_run_results(lsa_hypothesis, model_configuration)
            return True, output
        except:
            return False, None

    def prepare_run_results(self, lsa_hypothesis, model_configuration=None):
        if model_configuration is None:
            return {"lsa_propagation_strengths": lsa_hypothesis.propagation_strenghts}

        return {"lsa_propagation_strengths": lsa_hypothesis.lsa_propagation_strengths,
                "x0_values": model_configuration.x0_values,
                "e_values": model_configuration.e_values, "x1EQ": model_configuration.x1EQ,
                "zEQ": model_configuration.zEQ, "Ceq": model_configuration.Ceq}
