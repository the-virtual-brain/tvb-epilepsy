"""
Mechanism for launching TVB simulations.
"""

import sys
import time
import numpy
from tvb.datatypes import connectivity
from tvb.simulator import coupling, integrators, simulator
from tvb_infer.tvb_epilepsy.base.constants.model_constants import TIME_DELAYS_FLAG
from tvb_infer.base.utils.log_error_utils import initialize_logger
from tvb_infer.base.model.timeseries import Timeseries, TimeseriesDimensions
from tvb_infer.tvb_epilepsy.service.simulator.simulator import ABCSimulator
from tvb_infer.tvb_epilepsy.service.simulator.epileptor_model_factory import model_build_dict


class SimulatorTVB(ABCSimulator):
    """
    This class is used as a Wrapper over the TVB Simulator.
    It keeps attributes needed in order to create and configure a TVB Simulator object.
    """
    logger = initialize_logger(__name__)

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
                                         tract_lengths=TIME_DELAYS_FLAG * vep_conn.tract_lengths,
                                         region_labels=vep_conn.region_labels,
                                         centres=vep_conn.centres, hemispheres=vep_conn.hemispheres,
                                         orientations=vep_conn.orientations, areas=vep_conn.areas)

    def get_vois(self):
        # TODO: change 'lfp' for 'source'
        return [me.replace('x2 - x1', 'source') for me in self.simulation_settings.monitor_expressions]

    def config_simulation(self, noise, monitors, initial_conditions=None, **kwargs):

        if isinstance(self.model_configuration.model_connectivity, numpy.ndarray):
            tvb_connectivity = self._vep2tvb_connectivity(self.connectivity,
                                                          self.model_configuration.model_connectivity)
        else:
            tvb_connectivity = self._vep2tvb_connectivity(self.connectivity)

        tvb_coupling = coupling.Difference(a=1.)

        integrator = getattr(integrators, kwargs.get("integrator", "HeunStochastic"))(
                                                            dt=self.simulation_settings.integration_step, noise=noise)

        self.simTVB = simulator.Simulator(model=self.model, connectivity=tvb_connectivity, coupling=tvb_coupling,
                                          integrator=integrator, monitors=monitors,
                                          simulation_length=self.simulation_settings.simulated_period)
        self.simTVB.configure()

        self.configure_initial_conditions(initial_conditions=initial_conditions)

    def launch_simulation(self, report_every_n_monitor_steps=None):
        if report_every_n_monitor_steps >= 1:
            time_length_avg = numpy.round(self.simulation_settings.simulated_period / self.simTVB.monitors[0].period)
            n_report_blocks = max(report_every_n_monitor_steps * numpy.round(time_length_avg / 100), 1.0)
        else:
            n_report_blocks = 1

        self.simTVB._configure_history(initial_conditions=self.simTVB.initial_conditions)

        status = True

        if n_report_blocks < 2:
            try:
                tavg_time, tavg_data = self.simTVB.run()[0]

            except Exception, error_message:
                status = False
                self.logger.warning("Something went wrong with this simulation...:" + "\n" + error_message)
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
                self.logger.warning("Something went wrong with this simulation...:" + "\n" + str(error_message))
                return None, None, status

            tavg_time = numpy.array(tavg_time).flatten().astype('f')
            tavg_data = numpy.swapaxes(tavg_data, 1, 2).astype('f')
            # Variables of interest in a dictionary:
            sim_output = Timeseries(tavg_data, {TimeseriesDimensions.SPACE.value: self.connectivity.region_labels,
                                                TimeseriesDimensions.VARIABLES.value: self.get_vois()},
                                    tavg_time[0], numpy.diff(tavg_time).mean(), "ms")
            return sim_output, status

    def configure_model(self, **kwargs):
        self.model = model_build_dict[self.model._ui_name](self.model_configuration, **kwargs)

    def configure_initial_conditions(self, initial_conditions=None):

        if isinstance(initial_conditions, numpy.ndarray):
            self.simTVB.initial_conditions = initial_conditions

        else:
            self.simTVB.initial_conditions = self.prepare_initial_conditions(self.simTVB.good_history_shape[0])
