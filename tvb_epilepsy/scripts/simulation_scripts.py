import numpy as np

from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.custom.read_write import write_ts_epi, write_ts_seeg_epi
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D

###
# A helper function to make good choices for simulation settings for a custom simulator
###
def setup_custom_simulation_from_model_configuration(model_configuration, connectivity, dt, sim_length, monitor_period,
                                                     model_name, scale_time=1, noise_intensity=None):
    from tvb_epilepsy.custom.simulator_custom import EpileptorModel, custom_model_builder, \
        SimulationSettings, SimulatorCustom
    if model_name != EpileptorModel._ui_name:
        print "You can use only " + EpileptorModel._ui_name + "for custom simulations!"

    model = custom_model_builder(model_configuration)

    if noise_intensity is None:
        noise_intensity = 0  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                  scale_time=scale_time,
                                  noise_intensity=noise_intensity,
                                  monitor_sampling_period=monitor_period)

    simulator_instance = SimulatorCustom(connectivity, model_configuration, model, settings)

    return simulator_instance


###
# A helper function to make good choices for simulation settings, noise and monitors for a TVB simulator
###
def setup_TVB_simulation_from_model_configuration(model_configuration, connectivity, dt, sim_length, monitor_period,
                                                  model_name="EpileptorDP", zmode=np.array("lin"), scale_time=1,
                                                  noise_instance=None, noise_intensity=None, monitor_expressions=None,
                                                  monitors_instance=None):
    from tvb_epilepsy.base.constants import ADDITIVE_NOISE, NOISE_SEED
    from tvb_epilepsy.base.simulators import SimulationSettings
    from tvb_epilepsy.service.epileptor_model_factory import model_build_dict, model_noise_intensity_dict, \
        model_noise_type_dict
    from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB
    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic, EpileptorDP2D
    from tvb.datatypes import equations
    from tvb.simulator import monitors, noise
    from tvb.simulator.models import Epileptor

    model = model_build_dict[model_name](model_configuration, zmode=zmode)

    if isinstance(model, Epileptor):
        model.tt *= scale_time * 0.25
        model.r = 0.0001
    else:
        model.tau1 *= scale_time
        model.tau0 = 10000.0
        if isinstance(model, EpileptorDPrealistic):
            model.slope = 0.25
            model.pmode = np.array("z")

    if monitor_expressions is None:
        monitor_expressions = []
        for i in range(model._nvar):
            monitor_expressions.append("y" + str(i))
        if not (isinstance(model, EpileptorDP2D)):
            monitor_expressions.append("y3 - y0")

    if monitor_expressions is not None:
        model.variables_of_interest = monitor_expressions

    if monitors_instance is None:
        monitors_instance = monitors.TemporalAverage()

    if monitor_period is not None:
        monitors_instance.period = monitor_period

    default_noise_intensity = model_noise_intensity_dict[model_name]
    default_noise_type = model_noise_type_dict[model_name]

    if noise_intensity is None:
        noise_intensity = default_noise_intensity

    if noise_instance is not None:
        noise_instance.nsig = noise_intensity

    else:
        if default_noise_type is ADDITIVE_NOISE:
            noise_instance = noise.Additive(nsig=noise_intensity,
                                            random_stream=np.random.RandomState(seed=NOISE_SEED))
            noise_instance.configure_white(dt=dt)

        else:
            eq = equations.Linear(parameters={"a": 1.0, "b": 0.0})
            noise_instance = noise.Multiplicative(ntau=10, nsig=noise_intensity, b=eq,
                                                  random_stream=np.random.RandomState(seed=NOISE_SEED))
            noise_shape = noise_instance.nsig.shape
            noise_instance.configure_coloured(dt=dt, shape=noise_shape)

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt, scale_time=scale_time,
                                  noise_preconfig=noise_instance, noise_type=default_noise_type,
                                  noise_intensity=noise_intensity, noise_ntau=noise_instance.ntau,
                                  monitors_preconfig=monitors_instance, monitor_type=monitors_instance._ui_name,
                                  monitor_sampling_period=monitor_period, monitor_expressions=monitor_expressions,
                                  variables_names=model.variables_of_interest)

    simulator_instance = SimulatorTVB(connectivity, model_configuration, model, settings)

    return simulator_instance


def prepare_vois_ts_dict(vois, data):
    # Pack results into a dictionary:
    vois_ts_dict = dict()
    for idx_voi, voi in enumerate(vois):
        vois_ts_dict[voi] = data[:, idx_voi, :].astype('f')

    return vois_ts_dict


def prepare_ts_and_seeg_h5_file(folder, filename, model, projections, vois_ts_dict, hpf_flag, hpf_low, hpf_high, fsAVG,
                                dt):
    # High pass filter, and compute SEEG:
    if isinstance(model, EpileptorDP2D):
        raw_data = np.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x1"]])
        lfp_data = vois_ts_dict["x1"]

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['z'].dot(proj.T)

    else:
        if isinstance(model, EpileptorModel):
            lfp_data = vois_ts_dict["x2"] - vois_ts_dict["x1"]

        else:
            lfp_data = vois_ts_dict["lfp"]

        raw_data = np.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x2"]])

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['lfp'].dot(proj.T)
            if hpf_flag:
                for i in range(vois_ts_dict['seeg'].shape[0]):
                    vois_ts_dict['seeg_hpf%d' % i][:, i] = filter_data(
                        vois_ts_dict['seeg%d' % i][:, i], hpf_low, hpf_high,
                        fsAVG)
    # Write files:
    write_ts_epi(raw_data, dt, lfp_data, folder, filename)

    for i in range(len(projections)):
        write_ts_seeg_epi(vois_ts_dict['seeg%d' % i], dt, folder, filename)


def set_time_scales(fs=4096.0, time_length=100000.0, scale_time=1.0, scale_fsavg=None, report_every_n_monitor_steps=10,):
    dt = 1000.0 / fs
    if scale_fsavg is None:
        scale_fsavg = int(np.round(fs / 512.0))
    fsAVG = fs / scale_fsavg
    monitor_period = scale_fsavg * dt / scale_time
    sim_length = time_length / scale_time
    time_length_avg = np.round(sim_length / monitor_period)
    n_report_blocks = max(report_every_n_monitor_steps * np.round(time_length_avg / 100), 1.0)
    return dt, fsAVG, sim_length, monitor_period, n_report_blocks
