import numpy
from copy import deepcopy
from tvb_epilepsy.base.constants.model_constants import K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.pse.pse_service import ABCPSEService
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB
from tvb_epilepsy.service.simulator.simulator_custom import custom_model_builder


class SimulationPSEService(ABCPSEService):
    task = "SIMULATION"
    simulator = None

    def __init__(self, simulator, params_pse=None):
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

    def run(self, params, conn_matrix, hypothesis_input=None, model_config_service_input=None,
            yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize",
            update_initial_conditions=True):
        # Create new objects from the input simulator
        simulator_copy = deepcopy(self.simulator)
        model_copy = deepcopy(simulator_copy.model)
        try:
            if isinstance(hypothesis_input, DiseaseHypothesis):
                # Copy and update hypothesis
                model_configuration = \
                    self.update_hypo_model_config(hypothesis_input, params, conn_matrix,
                                                  model_config_service_input, yc, Iext1, K, a, b, x1eq_mode)[1]
                # Update simulator with new ModelConfiguration
                simulator_copy.model_configuration = model_configuration
                # Generate Model with new ModelConfiguration
                if isinstance(simulator_copy, SimulatorTVB):
                    model = model_build_dict[model_copy._ui_name](model_configuration, zmode=model_copy.zmode)
                else:
                    model = custom_model_builder(model_configuration)
                simulator_copy.model = model
            # Update model if needed
            # TODO: check if the name "model" is correct!
            self.update_object(simulator_copy.model, params, object_type="model")
            # Update other possible remaining parameters, i.e., concerning the integrator, noise etc
            # TODO: check if the name "SimulatorTVB" is correct!
            self.update_object(simulator_copy, params, object_type="SimulatorTVB")
            # Now, recalculate the default initial conditions...
            # If initial conditions were parameters, then, this flag can be set to False
            if update_initial_conditions:
                simulator_copy.configure_initial_conditions()
            time, data, status = simulator_copy.launch()
            output = self.prepare_run_results(data, time)
            return True, output
        except:
            return False, None

    def prepare_run_results(self, data, time):
        return {"time": time, "data": data}
