"""
Service for launching TVB simulations.
"""
from copy import deepcopy
import numpy

from tvb_fit.service.simulator_tvb import SimulatorTVB as SimulatorTVBBase

from tvb_fit.tvb_epilepsy.base.constants.model_constants import TIME_DELAYS_FLAG
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import compute_initial_conditions_from_eq_point
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.service.simulator.epileptor_model_factory import model_builder_from_model_config_fun, VOIS

from tvb_utils.log_error_utils import initialize_logger

from tvb.simulator import integrators, simulator, coupling  # , noise, monitors


class SimulatorTVB(SimulatorTVBBase):
    """
    This class is used as a Wrapper over the TVB Simulator.
    It keeps attributes needed in order to create and configure a TVB Simulator object.
    """
    logger = initialize_logger(__name__)

    def _vep2tvb_connectivity(self):
        return self._vp2tvb_connectivity(TIME_DELAYS_FLAG)

    def get_vois(self):
        vois = super(SimulatorTVB, self).get_vois(VOIS[self.simTVB.model._ui_name])
        if self.simTVB.model._nvar == 2:
            return numpy.array([me.replace('-x1', 'source') for me in vois])
        else:
            return numpy.array([me.replace('x2 - x1', 'source') for me in vois])

    def config_simulation(self, noise, monitors):

        model = model_builder_from_model_config_fun(self.model_configuration)
        # The Epileptor doesn't expose by default all its state variables to the monitor...:
        if model._ui_name == "Epileptor":
            model.variables_of_interest = VOIS[model._ui_name].tolist()

        tvb_connectivity = self._vep2tvb_connectivity()

        tvb_coupling = coupling.Difference(a=1.0)

        integrator = getattr(integrators, self.settings.integrator_type) \
                                (dt=self.settings.integration_step, noise=noise)

        self.simTVB = simulator.Simulator(model=model, connectivity=tvb_connectivity, coupling=tvb_coupling,
                                          integrator=integrator, monitors=monitors,
                                          simulation_length=self.settings.simulation_length)

        self.simTVB.configure()

        self.configure_initial_conditions()

    def config_simulation_from_tvb_simulator(self, tvb_simulator):
        # Ignore simulation settings and use the input tvb_simulator
        self.simTVB = deepcopy(tvb_simulator)
        self.simTVB.model = model_builder_from_model_config_fun(self.model_configuration)
        self.simTVB.connectivity = self._vp2tvb_connectivity()
        self.simTVB.configure()
        self.configure_initial_conditions()

    def configure_initial_conditions(self):
        initial_conditions = self.model_configuration.initial_conditions
        if isinstance(initial_conditions, numpy.ndarray):
            if len(initial_conditions.shape) < 4:
                initial_conditions = numpy.expand_dims(initial_conditions, 2)
                initial_conditions = numpy.tile(initial_conditions, (self.simTVB.good_history_shape[0], 1, 1, 1))
            self.simTVB.initial_conditions = initial_conditions
        else:
            self.simTVB.initial_conditions = \
                compute_initial_conditions_from_eq_point(self.model_configuration,
                                                         history_length=self.simTVB.good_history_shape[0],
                                                         simulation_shape=True, epileptor_model=self.simTVB.model)

    def launch_simulation(self, report_every_n_monitor_steps=None, timeseries=Timeseries):
        return super(SimulatorTVB, self).launch_simulation(report_every_n_monitor_steps, timeseries)
