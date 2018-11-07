# -*- coding: utf-8 -*-
import os

import numpy as np

from tvb_fit.base.constants import PriorsModes
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.utils.data_structures_utils import ensure_list, find_labels_inds, join_labels_indices_dict
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import K_UNSCALED_DEF, TAU1_DEF, TAU0_DEF
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import XModes, SDE_MODES, \
    OBSERVATION_MODELS, OBSERVATION_MODEL_DEF
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.probabilistic_models_builders import SDEProbabilisticModelBuilder
from tvb_fit.tvb_epilepsy.service.model_inversion_services import SDEModelInversionService
from tvb_fit.tvb_epilepsy.service.vep_stan_dict_builder import build_stan_model_data_dict
from tvb_fit.tvb_epilepsy.top.scripts.fitting_scripts import set_model_config_LSA, get_2D_simulation, \
    get_target_timeseries, set_target_timeseries, reconfigure_model_with_fit_estimates, run_fitting, \
    set_prior_parameters
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer


def path(name, base_path):
    return os.path.join(base_path , name + ".h5")


def set_hypotheses(head, config):
    hypotheses = []

    # Formulate a VEP hypothesis manually
    hyp_builder = HypothesisBuilder(head.connectivity.number_of_regions, config) # .set_normalize(1.5)

    # # Regions of Pathological Excitability hypothesis:
    # x0_indices = [6, 15, 52, 53] # [1, 26] #
    # x0_values = 2.5*np.array([0.9, 0.9, 0.5, 0.5])
    x0_indices = [6, 15]  # D, DK,: [1, 26] #
    x0_values = 2.5*np.array([0.9, 0.9])
    hyp_builder.set_x0_hypothesis(x0_indices, x0_values)

    # Regions of Model Epileptogenicity hypothesis:
    # e_indices = [6, 15, 52, 53]  # DK: [2, 25]
    e_indices = [52, 53]  # D, DK: [2, 25] #
    # e_values = np.array([0.9, 0.9, 0.5, 0.5])  # np.array([0.99] * 2)
    e_values = np.array([0.5, 0.5])  # np.array([0.99] * 2)
    hyp_builder.set_e_hypothesis(e_indices, e_values)

    # Regions of Connectivity hypothesis:
    # w_indices = []  # [(0, 1), (0, 2)]
    # w_values = []  # [0.5, 2.0]
    # hypo_builder.set_w_indices(w_indices).set_w_values(w_values)

    hypotheses.append(hyp_builder.build_hypothesis())

    # e_indices = [6, 15]  # [1, 2, 25, 26]
    # hypotheses.append(hyp_builder.build_hypothesis_from_file("postseeg", e_indices))
    # Change something manually if necessary
    # hypothesis2.x0_values = [0.01, 0.01]

    return tuple(hypotheses)


def main_fit_sim_hyplsa(stan_model_name, empirical_files, times_on, time_length, sim_times_on_off, sensors_lbls,
                        normal_flag=False, sim_source_type="fitting",
                        observation_model=OBSERVATION_MODEL_DEF, downsampling=2, preprocessing=[], normalization=None,
                        fitmethod="sample", pse_flag=True, fit_flag=True, test_flag=False,  config=Config(), **kwargs):

    # Prepare necessary services:
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    reader = H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)

    # Read head
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    sensors = ensure_list(head.get_sensors_by_name("distance"))[0]
    if isinstance(sensors, dict):
        sensors = sensors.values()[0]
        sensor_id = head.sensors_name_to_id(sensors.name)
    elif sensors is None:
        sensors = head.get_sensors_by_index()
        sensor_id = 0
    plotter.plot_head(head)

    # Set hypothesis:
    hyp = set_hypotheses(head, config)[0]

    config.out._out_base = os.path.join(config.out._out_base, hyp.name)
    plotter = Plotter(config)

    # Set model configuration and compute LSA
    model_configuration, lsa_hypothesis, pse_results = \
        set_model_config_LSA(head, hyp, reader, config, K_unscaled=3 * K_UNSCALED_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                             pse_flag=pse_flag, plotter=plotter, writer=writer)

    base_path = os.path.join(config.out.FOLDER_RES)

    # -----------Get prototypical simulated data from the fitting version of Epileptor(simulate if necessary) ----------
    source2D_file = path("Source2Dts", base_path)
    source2D_ts = get_2D_simulation(model_configuration, head, lsa_hypothesis, source2D_file, sim_times_on_off,
                                    config=config, reader=reader, writer=writer, plotter=plotter)

    # -------------------------- Get model_data and observation signals: -------------------------------------------

    target_data_file = path("FitTargetData", base_path)
    sim_target_file = path("ts_fit", base_path)
    empirical_target_file = path("ts_empirical", base_path)
    problstc_model_file = path("ProblstcModel", base_path)
    model_data_file = path("ModelData", base_path)

    if os.path.isfile(problstc_model_file) and os.path.isfile(model_data_file) and os.path.isfile(target_data_file):
        # Read existing probabilistic model and model data...
        probabilistic_model = reader.read_probabilistic_model(problstc_model_file)
        model_data = reader.read_dictionary(model_data_file)
        target_data = reader.read_timeseries(target_data_file)
    else:
        # Create model inversion service (stateless)
        model_inversion = SDEModelInversionService()
        model_inversion.normalization = normalization
        # Exclude ctx-l/rh-unknown regions from fitting
        model_inversion.active_regions_exlude = find_labels_inds(head.connectivity.region_labels, ["unknown"])

        # Generate probabilistic model and model data
        probabilistic_model = \
            SDEProbabilisticModelBuilder(model_name=stan_model_name, model_config=model_configuration,
                                         xmode=XModes.X1EQMODE.value, priors_mode=PriorsModes.INFORMATIVE.value,
                                         sde_mode=SDE_MODES.NONCENTERED.value, observation_model=observation_model,
                                         normal_flag=normal_flag,  K=np.mean(model_configuration.K)).\
                generate_model(generate_parameters=False)

        # Get by simulation and/or loading prototypical source 2D timeseries and the target (simulater or empirical)
        # time series for fitting
        signals, probabilistic_model, simulator = \
           get_target_timeseries(probabilistic_model, head, lsa_hypothesis, times_on, time_length,
                                 sensors_lbls, sensor_id, observation_model, sim_target_file, empirical_target_file,
                                 sim_source_type, downsampling, preprocessing, empirical_files, config, plotter)

        # Update active model's active region nodes
        e_values = pse_results.get("e_values_mean", model_configuration.e_values)
        lsa_propagation_strength = pse_results.get("lsa_propagation_strengths_mean",
                                                   lsa_hypothesis.lsa_propagation_strengths)
        model_inversion.active_e_th = 0.05
        probabilistic_model = \
            model_inversion.update_active_regions(probabilistic_model, sensors=sensors, e_values=e_values,
                                                  lsa_propagation_strengths=lsa_propagation_strength, reset=True)

        # Select and set the target time series
        target_data, probabilistic_model, model_inversion = \
            set_target_timeseries(probabilistic_model, model_inversion, signals, sensors, head,
                                  target_data_file, writer, plotter)

        #---------------------------------Finally set priors for the parameters-------------------------------------
        probabilistic_model = \
                set_prior_parameters(probabilistic_model, target_data, source2D_ts, None, problstc_model_file,
                                     [XModes.X0MODE.value, "x1_init", "z_init", "K", "tau1", "tau0", # "x1", "z"
                                      "sigma", "dWt", "epsilon", "scale", "offset"], normal_flag,
                                      writer=writer, plotter=plotter)

        # Construct the stan model data dict:
        model_data = build_stan_model_data_dict(probabilistic_model, target_data.squeezed,
                                                model_configuration.connectivity, time=target_data.time)

        # # ...or interface with INS stan models
        # from tvb_fit.service.model_inversion.vep_stan_dict_builder import \
        #   build_stan_model_data_dict_to_interface_ins
        # model_data = build_stan_model_data_dict_to_interface_ins(probabilistic_model, target_data.squeezed,
        #                                                          model_configuration.connectivity, gain_matrix,
        #                                                          time=target_data.time)
        writer.write_dictionary(model_data, model_data_file)

    # -------------------------- Fit or load results from previous fitting: -------------------------------------------

    estimates, samples, summary, info_crit = \
        run_fitting(probabilistic_model, stan_model_name, model_data, target_data, config, head,
                    hyp.all_disease_indices, ["K", "tau1", "tau0", "sigma", "epsilon", "scale", "offset"],
                    ["x0", "PZ", "x1eq", "zeq"], fit_flag, test_flag, base_path, fitmethod, n_chains_or_runs=10,
                    output_samples=100, num_warmup=100, min_samples_per_chain=100, max_depth=15, delta=0.95,
                    iter=100000, tol_rel_obj=1e-6, debug=1, simulate=0, writer=writer, plotter=plotter, **kwargs)


    # -------------------------- Reconfigure model after fitting:---------------------------------------------------
    model_configuration_fit = reconfigure_model_with_fit_estimates(head, model_configuration, probabilistic_model,
                                                                   estimates, base_path, writer, plotter)

    logger.info("Done!")


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    SUBJECT = "TVB3"
    head_folder = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", SUBJECT, "HeadD")
    SEEG_data = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "data/CC", "seeg", SUBJECT)

    if user_home == "/home/denis":
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "INScluster/fit")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)
        config.generic.C_COMPILER = "g++"
        config.generic.CMDSTAN_PATH = "/soft/stan/cmdstan-2.17.0"
    # elif user_home == "/Users/lia.domide":
    #     config = Config(head_folder="/WORK/episense/tvb-epilepsy/data/TVB3/Head",
    #                     raw_data_folder="/WORK/episense/tvb-epilepsy/data/TVB3/ts_seizure")
    #     config.generic.CMDSTAN_PATH = "/WORK/episense/cmdstan-2.17.1"

    else:
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results",
                              "fit/tests/empirical_singlestep_meancntrd_advi")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)
        config.generic.CMDSTAN_PATH = config.generic.CMDSTAN_PATH + "_precompiled"
    config.generic.PROBLSTC_MODELS_PATH = os.path.join(user_home, "VEPlocal/CC/tvb-epilepsy-cc-study/stan")

    # TVB3 larger preselection:
    sensors_lbls = join_labels_indices_dict({u"B'": np.arange(1, 5).tolist() + np.arange(12, 15).tolist(),
                                             u"F'": np.arange(1, 12).tolist(),
                                             u"G'": np.arange(1, 5).tolist() + np.arange(9, 16).tolist(),
                                             u"L'": np.arange(1, 14).tolist(),
                                             u"M'": np.arange(1, 4).tolist() + np.arange(7, 16).tolist(),
                                             u"O'": np.arange(1, 4).tolist() + np.arange(6, 13).tolist(),
                                             u"P'": np.arange(1, 4).tolist() + [8] + np.arange(10, 17).tolist(),
                                             u"R'": np.arange(1, 5).tolist() + np.arange(7, 10).tolist()})

    # Simulation times_on_off
    #  for "fitting" simulations with tau0=30.0
    sim_times_on_off = [70.0, 120.0] # e_hypo, [100, 130] for x0_hypo, and e_x0_hypo
    EMPIRICAL = True
    sim_source_type = "paper"
    observation_model = OBSERVATION_MODELS.SEEG_POWER.value  #OBSERVATION_MODELS.SEEG_LOGPOWER.value  #OBSERVATION_MODELS.SOURCE_POWER.value  #
    if EMPIRICAL:
        seizures_files = ['SZ1_0001.edf', 'SZ2_0001.edf']  # 'SZ2_0001.edf'
        times_on = [9700.0, 13700.0] # (np.array([15.0, 30.0]) * 1000.0).tolist() # for SZ1
        time_length = 25600.0
        # times_on_off = (np.array([15.0, 38.0]) * 1000.0).tolist()  # for SZ2
        # sensors_filename = "SensorsSEEG_116.h5"
        # # TVB4 preselection:
        # sensors_lbls = [u"D5", u"D6", u"D7",  u"D8", u"D9", u"D10", u"Z9", u"Z10", u"Z11", u"Z12", u"Z13", u"Z14",
        #                 u"S1", u"S2", u"S3", u"D'3", u"D'4", u"D'10", u"D'11", u"D'12", u"D'13", u"D'14"]
        # sensors_inds = [4, 5, 6, 7, 8, 9, 86, 87, 88, 89, 90, 91, 94, 95, 96, 112, 113, 119, 120, 121, 122, 123]
        # # TVB4:
        # seizure = 'SZ3_0001.edf'
        # sensors_filename = "SensorsSEEG_210.h5"
        # times_on_off = [20.0, 100.0]
    else:
        if sim_source_type == "paper":
            times_on = [1500.0] # fot e_x0_hypo # [1500.0] # for x0_hypo # [1200.0] # for e_hypo #
            time_length = 700.0 # for x0_hypo, and e_x0_hypo # 500.0 # for e_hypo #
        else:
            times_on_off = sim_times_on_off # for "fitting" simulations with tau0=30.0
            # times_on_off = [1100.0, 1300.0]  # for "fitting" simulations with tau0=300.0
            times_on = sim_times_on_off[0]
            time_length = sim_times_on_off[1] - sim_times_on_off[0]
    normalization = "zscore"
    preprocessing = []
    downsampling = 2
    normal_flag = False
    stan_model_name = "vep_sde_cc_meancntrd"
    fitmethod = "advi"   # ""  # "sample"  # "advi" or "opt"
    pse_flag = True
    fit_flag = True
    test_flag = False
    if EMPIRICAL:
        main_fit_sim_hyplsa(stan_model_name,
                            [os.path.join(config.input.RAW_DATA_FOLDER, seizure_file)
                                             for seizure_file in seizures_files],
                            times_on, time_length, sim_times_on_off, sensors_lbls,
                            normal_flag, sim_source_type, observation_model,
                            downsampling, preprocessing, normalization, fitmethod, pse_flag, fit_flag, test_flag, config)
    else:
        main_fit_sim_hyplsa(stan_model_name, [], times_on, time_length, sim_times_on_off, sensors_lbls,
                            normal_flag, sim_source_type, observation_model,
                            downsampling, preprocessing, normalization, fitmethod, pse_flag, fit_flag, test_flag, config)

