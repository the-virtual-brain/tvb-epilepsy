"""
Mechanism for launching TVB simulations.
"""
from abc import ABCMeta, abstractmethod

import sys
import time
import numpy
from tvb.datatypes import connectivity
from tvb.simulator.simulator import Simulator
# from tvb.model_config import coupling, integrators, model_config
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.model.timeseries import Timeseries, TimeseriesDimensions
from tvb_fit.tvb_epilepsy.service.simulator.simulator import ABCSimulator


class SimulatorTVB(ABCSimulator):
    __metaclass__ = ABCMeta

    """
    This class is used as a Wrapper over the TVB Simulator.
    It keeps attributes needed in order to create and configure a TVB Simulator object.
    """
    logger = initialize_logger(__name__)

    simTVB = Simulator()  # A tvb model_config
    model_configuration = None

    def __init__(self, model_configuration, simulatorTVB=Simulator()):
        self.simTVB = simulatorTVB
        self.model_configuration = model_configuration

    @staticmethod
    def _vp2tvb_connectivity(vp_conn, model_connectivity=None):
        if model_connectivity is None:
            model_connectivity = vp_conn.normalized_weights
        return connectivity.Connectivity(use_storage=False, weights=model_connectivity,
                                         tract_lengths=vp_conn.tract_lengths,
                                         region_labels=vp_conn.region_labels,
                                         centres=vp_conn.centres, hemispheres=vp_conn.hemispheres,
                                         orientations=vp_conn.orientations, areas=vp_conn.areas)

    def get_vois(self):
        # TODO: Confirm the path monitor.expression
        return [monitor.expression.replace('x2 - x1', 'source') for monitor in self.simTVB.monitors]

    def configure_initial_conditions(self, history_length=1, **kwargs):
        if isinstance(self.model_configuration.initial_conditions, numpy.ndarray):
            initial_conditions = numpy.expand_dims(self.model_configuration.initial_conditions, 2)
            self.simTVB.initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))

    def config_simulation(self, **kwargs):

        self.simTVB.connectivity = self._vp2tvb_connectivity(self.model_configuration.connectivity)

        #
        # self.simTVB = model_config.Simulator(model=self.model, connectivity=tvb_connectivity, coupling=tvb_coupling,
        #                                   integrator=integrator, monitors=monitors,
        #                                   simulation_length=self.simulation_settings.simulated_period)
        self.simTVB.configure()

        self.configure_initial_conditions(**kwargs)

    def launch_simulation(self, report_every_n_monitor_steps=None):
        if report_every_n_monitor_steps >= 1:
            time_length_avg = numpy.round(self.simTVB.simulation_length / self.simTVB.monitors[0].period)
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
            sim_output = \
                Timeseries(tavg_data,
                           {TimeseriesDimensions.SPACE.value: self.model_configuration.connectivity.region_labels,
                            TimeseriesDimensions.VARIABLES.value: self.get_vois()},
                           tavg_time[0], numpy.diff(tavg_time).mean(), "ms")
            return sim_output, status
