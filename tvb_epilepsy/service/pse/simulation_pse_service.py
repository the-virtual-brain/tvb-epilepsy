import numpy
from copy import deepcopy
from tvb_epilepsy.base.utils import raise_not_implemented_error, formal_repr
from tvb_epilepsy.custom.simulator_custom import custom_model_builder
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.pse.pse_service import ABCPSEService
from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB


class SimulationPSEService(ABCPSEService):
    simulator = None

    def __init__(self, simulator, params_pse=None):
        self.simulator = simulator
        self.params_pse = params_pse
        self.prepare_params(params_pse)

    def __repr__(self):
        d = {"01. Task": "Simulation",
             "02. Main PSE object": self.simulator,
             "03. Number of computation loops": self.n_loops,
             "04. Parameters": numpy.array(["%s" % l for l in self.params_names]),
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def run_pse_parallel(self):
        raise_not_implemented_error("PSE parallel not implemented!", self.logger)

    def run(self, conn_matrix, params, hypothesis):
        # Create new objects from the input simulator
        copy_simulator = deepcopy(self.simulator)
        copy_model = deepcopy(copy_simulator.model)

        try:
            # Copy hypothesis and update
            copy_hypo = deepcopy(hypothesis)
            copy_hypo.update_for_pse(params, self.params_paths, self.params_indices)

            # Create ModelConfigurationService and update
            model_configuration_service = ModelConfigurationService(copy_hypo.number_of_regions)
            model_configuration_service.update_for_pse(params, self.params_paths, self.params_indices)

            # Generate ModelConfiguration
            if copy_hypo.type == "Epileptogenicity":
                model_configuration = model_configuration_service.configure_model_from_E_hypothesis(copy_hypo,
                                                                                                    conn_matrix)
            else:
                model_configuration = model_configuration_service.configure_model_from_hypothesis(copy_hypo,
                                                                                                  conn_matrix)
            # Update simulator with new ModelConfiguration
            copy_simulator.model_configuration = model_configuration

            # Generate Model with new ModelConfiguration
            if isinstance(copy_simulator, SimulatorTVB):
                model = model_build_dict[copy_model._ui_name](model_configuration, zmode=copy_model.zmode)
            else:
                model = custom_model_builder(model_configuration)

            # Update model if needed
            self.set_object_attribute_recursively(model, params)
            copy_simulator.model = model

            # Update other possible remaining parameters, i.e., concerning the integrator, noise etc
            self.set_object_attribute_recursively(copy_simulator, params)

            copy_simulator.configure_initial_conditions()

            time, data, status = copy_simulator.launch()
            output = self.prepare_run_results(data, time)

            return True, output

        except:

            return False, None

    def prepare_run_results(self, data, time):
        return {"time": time, "data": data}
