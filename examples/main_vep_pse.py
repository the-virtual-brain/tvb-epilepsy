"""
Entry point for working with VEP
"""
import os
import warnings

import numpy

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.constants import FOLDER_RES, FOLDER_FIGURES, SAVE_FLAG, SHOW_FLAG, SIMULATION_MODE, \
    TVB, DATA_MODE, VOIS, DATA_CUSTOM, X0_DEF, E_DEF
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.base.lsa_service import LSAService
from tvb_epilepsy.base.plot_tools import plot_nullclines_eq, plot_sim_results, \
    plot_hypothesis_model_configuration_and_lsa
from tvb_epilepsy.base.utils import initialize_logger, set_time_scales, calculate_projection, filter_data
from tvb_epilepsy.custom.read_write import write_h5_model, write_ts_epi, write_ts_seeg_epi
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.base.sample_service import DeterministicSampleService, StochasticSampleService, mean_std_to_low_high
from tvb_epilepsy.base.pse_service import PSE_service

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

if SIMULATION_MODE is TVB:
    from tvb_epilepsy.tvb_api.simulator_tvb import setup_simulation
else:
    from tvb_epilepsy.custom.simulator_custom import setup_simulation


def prepare_vois_ts_dict(vois, data):
    # Pack results into a dictionary:
    vois_ts_dict = dict()
    for idx_voi, voi in enumerate(vois):
        vois_ts_dict[voi] = data[:, idx_voi, :].astype('f')

    return vois_ts_dict


def prepare_ts_and_seeg_h5_file(hyp_name, model, projections, vois_ts_dict, hpf_flag, hpf_low, hpf_high, fsAVG, dt):
    # High pass filter, and compute SEEG:
    if isinstance(model, EpileptorDP2D):
        raw_data = numpy.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x1"]])
        lfp_data = vois_ts_dict["x1"]

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['z'].dot(proj.T)

    else:
        if isinstance(model, EpileptorModel):
            lfp_data = vois_ts_dict["x2"] - vois_ts_dict["x1"]

        else:
            lfp_data = vois_ts_dict["lfp"]

        raw_data = numpy.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x2"]])

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['lfp'].dot(proj.T)
            if hpf_flag:
                for i in range(vois_ts_dict['seeg'].shape[0]):
                    vois_ts_dict['seeg_hpf%d' % i][:, i] = filter_data(
                        vois_ts_dict['seeg%d' % i][:, i], hpf_low, hpf_high,
                        fsAVG)
    # Write files:
    write_ts_epi(raw_data, dt, lfp_data, path=os.path.join(FOLDER_RES, hyp_name + "_ep_ts.h5"))

    for i in range(len(projections)):
        write_ts_seeg_epi(vois_ts_dict['seeg%d' % i], dt, path=os.path.join(FOLDER_RES, hyp_name + "_ep_ts.h5"))


if __name__ == "__main__":

    logger = initialize_logger(__name__)

    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    # --------------------------Hypothesis definition-----------------------------------

    # Manual definition of hypothesis...:
    x0_indices = [20]
    x0_values = [0.9]
    E_indices = [70]
    E_values = [0.9]
    disease_values = x0_values + E_values
    disease_indices = x0_indices + E_indices

    # ...or reading a custom file:
    # ep_name = "ep_test1"
    # FOLDER_RES = os.path.join(data_folder, ep_name)
    # from tvb_epilepsy.custom.readers_custom import CustomReader
    # if not isinstance(reader, CustomReader):
    #     reader = CustomReader()
    # disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    # disease_indices, = numpy.where(disease_values > numpy.min([X0_DEF, E_DEF]))
    # disease_values = disease_values[disease_indices]
    # if disease_values.size > 1:
    #     inds_split = numpy.ceil(disease_values.size * 1.0 / 2).astype("int")
    #     x0_indices = disease_indices[:inds_split].tolist()
    #     E_indices = disease_indices[inds_split:].tolist()
    # else:
    #     x0_indices = disease_indices.tolist()
    #     E_indices = []
    # disease_indices = list(disease_indices)

    all_regions_indices = numpy.array(range(head.number_of_regions))

    stoch_sampler = StochasticSampleService(shape=(100,), distribution="norm", trunc_limits={"low": 0.0},
                                            random_seed=1000, loc=10.0, scale=3.0)
    K_samples, K_sample_stats = stoch_sampler.generate_samples(stats=True)

    # This is an example of Excitability Hypothesis:
    hyp_x0 = DiseaseHypothesis(head.connectivity, numpy.array(disease_values), x0_indices, [], [], [],
                               "Excitability", "Excitability_Hypothesis")
    healthy_indices = numpy.delete(all_regions_indices, x0_indices)
    stoch_sampler = StochasticSampleService(shape=(10,), distribution="norm", trunc_limits={"high": 1.0},
                                            random_seed=x0_indices[0], loc=0.9, scale=0.2)
    params_pse_x0 = [("hypothesis.x0", stoch_sampler.generate_samples(), x0_indices)]
    stoch_sampler = StochasticSampleService(shape=(10,), distribution="norm", trunc_limits={"low": 0.0, "high": 1.0},
                                            loc=0.3, scale=0.1)
    for ii in healthy_indices:
        stoch_sampler.random_seed = ii
        params_pse_x0.append(("model_configuration_service.x0", stoch_sampler.generate_samples(stats=False), [ii]))

    params_pse_x0.append(("model_configuration_service.K_unscaled", K_samples[:10], all_regions_indices))

    pse_hyp_x0 = PSE_service("LSA", hypothesis=hyp_x0, params_pse=params_pse_x0, grid_mode=False)

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = DiseaseHypothesis(head.connectivity, numpy.array(disease_values), [], E_indices, [], [],
                              "Epileptogenicity", "Epileptogenicity_Hypothesis")

    healthy_indices = numpy.delete(all_regions_indices, E_indices)
    stoch_sampler = StochasticSampleService(shape=(10,), distribution="norm", trunc_limits={"high": 1.0},
                                            random_seed=E_indices[0], loc=0.9, scale=0.2)
    params_pse_E = [("hypothesis.E", stoch_sampler.generate_samples(), E_indices)]
    stoch_sampler = StochasticSampleService(shape=(10,), distribution="norm", trunc_limits={"low": 0.0, "high": 1.0},
                                            loc=0.3, scale=0.1)
    for ii in healthy_indices:
        stoch_sampler.random_seed = ii
        params_pse_E.append(("model_configuration_service.E", stoch_sampler.generate_samples(stats=False), [ii]))

    params_pse_E.append(("model_configuration_service.K_unscaled", K_samples[:10], all_regions_indices))

    pse_hyp_E = PSE_service("LSA", hypothesis=hyp_E, params_pse=params_pse_E, grid_mode=False)

    if len(E_indices) > 0:
        # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
        hyp_x0_E = DiseaseHypothesis(head.connectivity, numpy.array(disease_values), x0_indices, E_indices, [], [],
                                     "Excitability", "Mixed_x0_E_Hypothesis")

        # Prepare parameter samples for pse:
        # Deterministic sampling for the 2 diseased regions to scan values around their mean value:
        low_x0, high_x0 = mean_std_to_low_high(mu=0.9, std=0.02)
        low_E, high_E = mean_std_to_low_high(mu=0.9, std=0.025)
        det_sampler = DeterministicSampleService(shape=(10,), low=low_x0, high=high_x0)
        x0_samples = det_sampler.generate_samples(stats=False)
        det_sampler = DeterministicSampleService(shape=(10,), low=low_E, high=high_E)
        E_samples = det_sampler.generate_samples(stats=False)
        # We generate a 2 dimensional grid outside pse_service, because the rest of the parameters will not be in
        # grid mode:
        x0_samples, E_samples = numpy.meshgrid(*[x0_samples, E_samples], sparse=False, indexing="ij")
        x0_samples = x0_samples.flatten()
        E_samples = E_samples.flatten()
        params_pse_x0_E = [("hypothesis.disease_values", x0_samples, [0]),
                           ("hypothesis.disease_values", E_samples, [1])]

        # Stochastic normal sampling for healthy regions around their x0 value
        stoch_sampler = StochasticSampleService(shape=(100,), distribution="uniform",
                                                sampling_module="numpy", low=0.0, high=0.25)
        healthy_indices = numpy.delete(all_regions_indices, disease_indices)
        for ii in healthy_indices:
            stoch_sampler.random_seed = ii
            params_pse_x0_E.append(("model_configuration_service.x0",
                                    stoch_sampler.generate_samples(stats=False), [ii]))

        # Add the global coupling K samples:
        params_pse_x0_E.append(("model_configuration_service.K_unscaled", K_samples, all_regions_indices))

        pse_hyp_x0_E = PSE_service("LSA", hypothesis=hyp_x0_E, params_pse=params_pse_x0_E, grid_mode=False)

        hypotheses = (hyp_x0, hyp_E, hyp_x0_E)

        pse_instances = {hyp_x0: pse_hyp_x0, hyp_E: pse_hyp_E, hyp_x0_E: pse_hyp_x0_E}

    else:

        hypotheses = (hyp_x0, hyp_E)

        pse_instances = {hyp_x0: pse_hyp_x0, hyp_E: pse_hyp_E}

    # --------------------------Projections computations-----------------------------------

    sensorsSEEG = []
    projections = []
    for sensors, projection in head.sensorsSEEG.iteritems():
        if projection is None:
            continue
        else:
            projection = calculate_projection(sensors, head.connectivity)
            head.sensorsSEEG[sensors] = projection
            print projection.shape
            sensorsSEEG.append(sensors)
            projections.append(projection)

    # --------------------------Simulation preparations-----------------------------------

    # TODO: maybe use a custom Monitor class
    fs = 2 * 4096.0
    scale_time = 2.0
    time_length = 10000.0
    scale_fsavg = 2.0
    report_every_n_monitor_steps = 10.0
    (dt, fsAVG, sim_length, monitor_period, n_report_blocks) = \
        set_time_scales(fs=fs, dt=None, time_length=time_length, scale_time=scale_time, scale_fsavg=scale_fsavg,
                        report_every_n_monitor_steps=report_every_n_monitor_steps)

    model_name = "EpileptorDP"

    # We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    hpf_flag = False
    hpf_low = max(16.0, 1000.0 / time_length)  # msec
    hpf_high = min(250.0, fsAVG)

    # --------------------------Hypothesis and LSA-----------------------------------

    for hyp in hypotheses:

        model_configuration_service = ModelConfigurationService(hyp.get_number_of_regions())
        if hyp.type == "Epileptogenicity":
            model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hyp)
        else:
            model_configuration = model_configuration_service.configure_model_from_hypothesis(hyp)

        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)

        plot_hypothesis_model_configuration_and_lsa(lsa_hypothesis, model_configuration,
                                                    n_eig=lsa_service.eigen_vectors_number)

        # write_h5_model(hyp.prepare_for_h5(), folder_name=FOLDER_RES, file_name=hyp.name + ".h5")
        write_h5_model(lsa_hypothesis.prepare_for_h5(), folder_name=FOLDER_RES, file_name=lsa_hypothesis.name + ".h5")
        write_h5_model(prepare_for_h5(lsa_service), folder_name=FOLDER_RES,
                       file_name=lsa_hypothesis.name + "_LSAConfig.h5")
        write_h5_model(model_configuration.prepare_for_h5(), folder_name=FOLDER_RES,
                       file_name=lsa_hypothesis.name + "_ModelConfig.h5")

        if isinstance(pse_instances[hyp], PSE_service):
            pse_propagation_strengths, execution_status = pse_instances[hyp].run_pse(grid_mode=False, model_configuration_service_input=model_configuration_service,
                                                                                     lsa_service_input=lsa_service)


        # # ------------------------------Simulation--------------------------------------
        #
        # simulator_instance = setup_simulation(model_configuration, head.connectivity, dt, sim_length, monitor_period,
        #                                       model_name, scale_time=scale_time, noise_intensity=10 ** -8)
        #
        # simulator_instance.config_simulation()
        # ttavg, tavg_data, status = simulator_instance.launch_simulation(n_report_blocks)
        #
        # write_h5_model(simulator_instance.prepare_for_h5(), folder_name=FOLDER_RES,
        #                file_name=lsa_hypothesis.name + "_sim_settings.h5")
        #
        # if not status:
        #     warnings.warn("Simulation failed!")
        #
        # else:
        #     tavg_data = tavg_data[:, :, :, 0]
        #
        #     vois = VOIS[model_name]
        #
        #     model = simulator_instance.model
        #
        #     logger.info("\nSimulated signal return shape: %s", tavg_data.shape)
        #     logger.info("Time: %s - %s", scale_time * ttavg[0], scale_time * ttavg[-1])
        #     logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())
        #
        #     time = scale_time * numpy.array(ttavg, dtype='float32')
        #     dt2 = numpy.min(numpy.diff(time))
        #
        #     vois_ts_dict = prepare_vois_ts_dict(vois, tavg_data)
        #
        #     prepare_ts_and_seeg_h5_file(lsa_hypothesis.name, model, projections, vois_ts_dict, hpf_flag, hpf_low,
        #                                 hpf_high, fsAVG, dt2)
        #
        #     vois_ts_dict['time'] = time
        #
        #     # Plot results
        #     if model.zmode is not numpy.array("lin"):
        #         plot_nullclines_eq(model_configuration, head.connectivity.region_labels,
        #                            special_idx=lsa_hypothesis.propagation_indices,
        #                            model=str(model.nvar) + "d", zmode=model.zmode,
        #                            figure_name=lsa_hypothesis.name + "_Nullclines and equilibria", save_flag=SAVE_FLAG,
        #                            show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES)
        #     plot_sim_results(model, lsa_hypothesis.propagation_indices, lsa_hypothesis.name, head, vois_ts_dict,
        #                      sensorsSEEG, hpf_flag)
        #
        #     # Save results
        #     vois_ts_dict['time_units'] = 'msec'
        #     # savemat(os.path.join(FOLDER_RES, hypothesis.name + "_ts.mat"), vois_ts_dict)
