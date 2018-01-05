"""
Mechanism for launching TVB simulations.
"""

import sys
import time

import numpy
from tvb.datatypes import connectivity
from tvb.simulator import coupling, integrators, monitors, noise, simulator

from tvb_epilepsy.base.constants.module_constants import TIME_DELAYS_FLAG
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.simulators import ABCSimulator
from tvb_epilepsy.custom.read_write import epileptor_model_attributes_dict
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict


class SimulatorTVB(ABCSimulator):
    def __init__(self, connectivity, model_configuration, model, simulation_settings):
        self.model = model
        self.simulation_settings = simulation_settings
        self.model_configuration = model_configuration
        self.connectivity = connectivity

    @staticmethod
    def _vep2tvb_connectivity(vep_conn, model_connectivity=None):
        if model_connectivity is None:
            model_connectivity = vep_conn.normalized_weights
        return connectivity.Connectivity(use_storage=False, weights=model_connectivity,
                                         tract_lengths=TIME_DELAYS_FLAG*vep_conn.tract_lengths,
                                         region_labels=vep_conn.region_labels,
                                         centres=vep_conn.centres, hemispheres=vep_conn.hemispheres,
                                         orientations=vep_conn.orientations, areas=vep_conn.areas)

    def config_simulation(self, initial_conditions=None):

        if isinstance(self.model_configuration.model_connectivity, numpy.ndarray):
            tvb_connectivity = self._vep2tvb_connectivity(self.connectivity,
                                                          self.model_configuration.model_connectivity)
        else:
            tvb_connectivity = self._vep2tvb_connectivity(self.connectivity)
        tvb_coupling = coupling.Difference(a=1.)

        # Set noise:
        if isinstance(self.simulation_settings.noise_preconfig, noise.Noise):
            integrator = integrators.HeunStochastic(dt=self.simulation_settings.integration_step,
                                                    noise=self.simulation_settings.noise_preconfig)
        else:
            self.simulation_settings.noise_intensity = numpy.array(self.simulation_settings.noise_intensity)
            if self.simulation_settings.noise_intensity.size == 1:
                self.simulation_settings.noise_intensity = numpy.repeat(
                    numpy.squeeze(self.simulation_settings.noise_intensity), self.model.nvar)
            if numpy.min(self.simulation_settings.noise_intensity) > 0:
                thisNoise = noise.Additive(nsig=self.simulation_settings.noise_intensity,
                                           random_stream=numpy.random.RandomState(
                                               seed=self.simulation_settings.noise_seed))
                self.simulation_settings.noise_type = "Additive"
                integrator = integrators.HeunStochastic(dt=self.simulation_settings.integration_step, noise=thisNoise)
            else:
                integrator = integrators.HeunDeterministic(dt=self.simulation_settings.integration_step)
                self.simulation_settings.noise_type = "None"

        # Set monitors:
        what_to_watch = []
        if isinstance(self.simulation_settings.monitors_preconfig, monitors.Monitor):
            what_to_watch = (self.simulation_settings.monitors_preconfig,)
        elif isinstance(self.simulation_settings.monitors_preconfig, tuple) or isinstance(
                self.simulation_settings.monitors_preconfig, list):
            for monitor in self.simulation_settings.monitors_preconfig:
                if isinstance(monitor, monitors.Monitor):
                    what_to_watch.append(monitor)
                what_to_watch = tuple(what_to_watch)

        self.simTVB = simulator.Simulator(model=self.model, connectivity=tvb_connectivity, coupling=tvb_coupling,
                                          integrator=integrator, monitors=what_to_watch,
                                          simulation_length=self.simulation_settings.simulated_period)
        self.simTVB.configure()

        self.configure_initial_conditions(initial_conditions=initial_conditions)

    def launch_simulation(self, n_report_blocks=1):

        self.simTVB._configure_history(initial_conditions=self.simTVB.initial_conditions)

        status = True

        if n_report_blocks < 2:
            try:
                tavg_time, tavg_data = self.simTVB.run()[0]

            except Exception, error_message:
                status = False
                warning("Something went wrong with this simulation...:" + "\n" + error_message)
                return None, None, status

            return tavg_time, tavg_data, status

        else:

            sim_length = self.simTVB.simulation_length / self.simTVB.monitors[0].period
            block_length = sim_length / n_report_blocks
            curr_time_step = 0.0
            curr_block = 1.0

            # Perform the simulation
            tavg_data, tavg_time = [], []

            start = time.time()

            try:
                for tavg in self.simTVB():

                    curr_time_step += 1.0

                    if not tavg is None:
                        tavg_time.append(tavg[0][0])
                        tavg_data.append(tavg[0][1])

                    if curr_time_step >= curr_block * block_length:
                        end_block = time.time()
                        # TODO: correct this part to print percentage of simulation at the same line by erasing previous
                        print_this = "\r" + "..." + str(100 * curr_time_step / sim_length) + "% done in " + \
                                     str(end_block - start) + " secs"
                        sys.stdout.write(print_this)
                        sys.stdout.flush()
                        curr_block += 1.0
            except Exception, error_message:
                status = False
                warning("Something went wrong with this simulation...:" + "\n" + str(error_message))
                return None, None, status

            return numpy.array(tavg_time), numpy.array(tavg_data), status

    def _prepare_for_h5(self):

        attributes_dict = epileptor_model_attributes_dict[self.model._ui_name]
        for attr in attributes_dict:
            p = self.model.x0.shape
            field = getattr(self.model, attributes_dict[attr])
            if isinstance(field, (float, int, long, complex)) \
                    or (isinstance(field, (numpy.ndarray))
                        and numpy.all(str(field.dtype)[1] != numpy.array(["O", "S"])) and field.size == 1):
                setattr(self.model, attributes_dict[attr], field * numpy.ones(p))

        settings_h5_model = convert_to_h5_model(self.simulation_settings)
        epileptor_model_h5_model = convert_to_h5_model(self.model)

        epileptor_model_h5_model.append(settings_h5_model)
        epileptor_model_h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        epileptor_model_h5_model.add_or_update_metadata_attribute("Monitor expressions",
                                                                  self.simulation_settings.monitor_expressions)
        epileptor_model_h5_model.add_or_update_metadata_attribute("Variables names",
                                                                  self.simulation_settings.variables_names)

        return epileptor_model_h5_model

    def configure_model(self, **kwargs):
        self.model = model_build_dict[self.model._ui_name](self.model_configuration, **kwargs)

    def configure_initial_conditions(self, initial_conditions=None):

        if isinstance(initial_conditions, numpy.ndarray):
            self.simTVB.initial_conditions = initial_conditions

        else:
            self.simTVB.initial_conditions = self.prepare_initial_conditions(self.simTVB.good_history_shape[0])
