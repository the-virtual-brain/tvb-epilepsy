"""
Service for launching TVB simulations.
"""
from copy import deepcopy
import sys
import time
import numpy

from tvb_fit.base.constants import TIME_DELAYS_FLAG
from tvb_fit.service.simulator import ABCSimulator

from tvb_utils.log_error_utils import initialize_logger
from tvb_timeseries.model.timeseries import Timeseries, TimeseriesDimensions

from tvb.simulator import integrators, simulator, coupling, noise, monitors


class SimulatorTVB(ABCSimulator):
    """
    This class is used as a Wrapper over the TVB Simulator.
    It keeps attributes needed in order to create and configure a TVB Simulator object.
    """
    logger = initialize_logger(__name__)

    simTVB = None

    def __init__(self, model_configuration, connectivity, settings):
        super(SimulatorTVB, self).__init__(model_configuration, connectivity, settings)
        self.simTVB = None

    def get_vois(self, model_vois=None):
        if model_vois is None:
            model_vois = self.simTVB.model.variables_of_interest
        return self.settings.monitor_expressions(model_vois)

    @property
    def model(self):
        return self.simTVB.model

    # General choices are made here to be used as an example.
    def config_simulation(self, model):
        # TODO: generate model from self.model_configuration for every specific implementation
        tvb_connectivity = self._vp2tvb_connectivity(TIME_DELAYS_FLAG)

        tvb_coupling = coupling.Difference(a=1.0)

        noise_instance = noise.Additive(nsig=self.settings.noise_intensity,
                                        random_stream=numpy.random.RandomState(seed=self.settings.noise_seed))

        integrator = getattr(integrators, self.settings.integrator_type) \
                                (dt=self.settings.integration_step, noise=noise_instance)

        monitor = monitors.TemporalAverage()
        monitor.period = self.settings.monitor_sampling_period

        self.simTVB = simulator.Simulator(model=model, connectivity=tvb_connectivity,
                                          coupling=tvb_coupling, integrator=integrator,
                                          monitors=[monitor], simulation_length=self.settings.simulation_length)
        self.simTVB.configure()

        self.configure_initial_conditions()

    def config_simulation_from_tvb_simulator(self, tvb_simulator):
        # Ignore simulation settings and use the input tvb_simulator
        self.simTVB = deepcopy(tvb_simulator)
        self.simTVB.model = tvb_simulator.model  # TODO: compare this with self.model_configuration
        self.simTVB.connectivity = self._vp2tvb_connectivity(TIME_DELAYS_FLAG)
        self.simTVB.configure()
        self.configure_initial_conditions()

    def launch_simulation(self, report_every_n_monitor_steps=None, timeseries=Timeseries):
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

            except Exception as error_message:
                status = False
                self.logger.warning("Something went wrong with this simulation...:" + "\n" + str(error_message))
                return None, status

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
                return None, status

        tavg_time = numpy.array(tavg_time).flatten().astype('f')
        tavg_data = numpy.swapaxes(tavg_data, 1, 2).astype('f')

        return timeseries(# substitute with TimeSeriesRegion fot TVB like functionality
                          tavg_data, time=tavg_time,
                          connectivity=self.simTVB.connectivity,
                          labels_ordering=["Time", TimeseriesDimensions.VARIABLES.value, "Region", "Samples"],
                          labels_dimensions={TimeseriesDimensions.SPACE.value: self.connectivity.region_labels,
                                             TimeseriesDimensions.VARIABLES.value: self.get_vois()}, ts_type="Region"), \
               status

