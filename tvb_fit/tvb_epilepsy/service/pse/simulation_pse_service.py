from copy import deepcopy
from tvb_fit.service.pse.simulation_pse_service import SimulationPSEService as SimulationPSEServiceBase
from tvb_fit.tvb_epilepsy.service.pse.pse_service import PSEService
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis


class SimulationPSEService(SimulationPSEServiceBase, PSEService):

    def __init__(self, simulator, params_pse=None):
        PSEService.__init__(self)
        SimulationPSEServiceBase.__init__(self, simulator, params_pse)

    def run(self, params, conn_matrix, model_config_builder_input=None, update_initial_conditions=True,
            hypothesis_input=None,  x1eq_mode="optimize"):
        # Create new objects from the input simulator
        simulator_copy = deepcopy(self.simulator)
        try:
            if isinstance(hypothesis_input, DiseaseHypothesis):
                # Copy and update hypothesis
                model_configuration = \
                    self.update_model_config(params, conn_matrix, model_config_builder_input,
                                             hypothesis_input, x1eq_mode)[0]
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

