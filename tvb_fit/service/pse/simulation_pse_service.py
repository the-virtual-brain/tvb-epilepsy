import numpy
from copy import deepcopy

from tvb_fit.service.pse.pse_service import ABCPSEService
from tvb_fit.service.model_configuration_builder import ModelConfigurationBuilder

from tvb_utils.log_error_utils import raise_not_implemented_error
from tvb_utils.data_structures_utils import formal_repr


class SimulationPSEService(ABCPSEService):
    task = "SIMULATION"
    simulator = None

    def __init__(self, simulator, params_pse=None):
        super(SimulationPSEService, self).__init__()
        self.simulator = simulator
        self.params_pse = params_pse
        self.prepare_params(params_pse)

    def __repr__(self):
        d = {"01. Task": self.task,
             "02. Main PSE object": self.simulator,
             "03. Number of computation loops": self.n_loops,
             "04. Parameters": numpy.array(["%s" % l for l in self.params_names]),
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def run_pse_parallel(self):
        raise_not_implemented_error("PSE parallel not implemented!", self.logger)

    def run(self, params, conn_matrix, model_config_builder_input=None, update_initial_conditions=True, **kwargs):
        # Create new objects from the input simulator
        simulator_copy = deepcopy(self.simulator)
        try:
            if isinstance(model_config_builder_input, ModelConfigurationBuilder):
                # Copy and update hypothesis
                model_configuration = self.update_model_config(params, conn_matrix, model_config_builder_input,
                                                               **kwargs)[0]
                # Update simulator with new EpileptorModelConfiguration
                simulator_copy.model_configuration = model_configuration
                # Confgure the new simulator_copy with the new EpileptorModelConfiguration
                simulator_copy.config_simulation_from_tvb_simulator(simulator_copy.simTVB)
            # Further/alternatively update model if needed
            self.update_object(simulator_copy.model, params, object_type="model")
            # Update other possible remaining parameters, i.e., concerning the integrator, noise etc
            self.update_object(simulator_copy, params, object_type="simTVB")
            # Now, recalculate the default initial conditions...
            # If initial conditions were parameters, then, this flag can be set to False
            if update_initial_conditions:
                simulator_copy.configure_initial_conditions()
            output, status = simulator_copy.launch()
            return status, output
        except:
            return False, None
