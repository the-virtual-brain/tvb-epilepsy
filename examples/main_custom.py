"""
Entry point to run a custom simulation
"""

import os
import copy
import numpy

from tvb_epilepsy.base.constants import X0_DEF
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.base.plot_tools import plot_head, plot_hypothesis, plot_timeseries, plot_trajectories
from tvb_epilepsy.base.utils import initialize_logger, set_time_scales
from tvb_epilepsy.custom.read_write import read_ts
from tvb_epilepsy.custom.simulator_custom import setup_simulation

if __name__ == "__main__":
    logger = initialize_logger(__name__)

    # -------------------------------Reading data-----------------------------------

    logger.info("Reading from custom")
    data_folder = os.path.join("/WORK/Episense/trunk/demo-data", 'Head_TREC')
    from tvb_epilepsy.custom.readers_custom import CustomReader

    reader = CustomReader()

    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    logger.debug("Loaded Head " + str(head))
    logger.debug("Loaded Connectivity " + str(head.connectivity))

    # --------------------------Hypothesis and LSA-----------------------------------

    hyp_ep = Hypothesis(head.number_of_regions, head.connectivity.weights, "EP_Hypothesis")
    hyp_ep.K = 0.1 * hyp_ep.K
    logger.info("Configure the hypothesis e...")
    iE = numpy.array([7, 50])
    E = numpy.array([0.5, 0.8], dtype=numpy.float32)
    seizure_indices = numpy.array([7, 50], dtype=numpy.int32)

    hyp_ep.configure_e_hypothesis(iE, E, seizure_indices)
    logger.debug(str(hyp_ep))

    hyp_exc = copy.deepcopy(hyp_ep)
    hyp_exc.name = "Exc_Hypothesis"
    logger.info("Configure the hypothesis x0...")
    ii = numpy.array(range(head.number_of_regions), dtype=numpy.int32)
    ix0 = numpy.delete(ii, iE)

    hyp_exc.configure_x0_hypothesis(ix0, X0_DEF, seizure_indices)
    logger.debug(str(hyp_exc))

    # ------------------------------Simulation--------------------------------------

    (fs, dt, fsAVG, scale_time, sim_length, monitor_period,
     n_report_blocks, hpf_fs, hpf_low, hpf_high) = set_time_scales(fs=2 * 4096.0, dt=None, time_length=3000.0,
                                                                   scale_time=2.0, scale_fsavg=2.0,
                                                                   hpf_low=None, hpf_high=None)

    for hyp in (hyp_ep, hyp_exc):
        simulator_instance, settings, variables_names, model = \
            setup_simulation(data_folder, hyp, dt, sim_length, monitor_period, scale_time, noise_intensity=None)
        simulator_instance.config_simulation(hyp, os.path.join(simulator_instance.path, "Coonnectivity.h5"),
                                             vep_settings=settings)
        simulator_instance.launch_simulation(hyp)

    # -------------------------------Plotting---------------------------------------

    VERY_LARGE_SIZE = (30, 15)
    SAVE_FLAG = True

    plot_head(head, save_flag=SAVE_FLAG, figsize=VERY_LARGE_SIZE)

    for hyp in (hyp_ep, hyp_exc):
        plot_hypothesis(hyp, head.connectivity.region_labels, save_flag=SAVE_FLAG, figsize=VERY_LARGE_SIZE)

        ts_time, ts_data = read_ts(os.path.join(data_folder, hyp.name, "ts.h5"), data="data")

        plot_timeseries(ts_time, {'ts0': ts_data[:, :, 0], 'ts1': ts_data[:, :, 1], 'ts2': ts_data[:, :, 2]},
                        save_flag=True, title="Ts_" + hyp.name)

        plot_trajectories({'ts0': ts_data[:, :, 0], 'ts1': ts_data[:, :, 1], 'ts2': ts_data[:, :, 2]}, save_flag=True,
                          title="Traj_" + hyp.name)
