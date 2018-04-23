# encoding=utf8


from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.model_inversion.statistical_models_builders import SDEStatisticalModelBuilder
from tvb_epilepsy.service.model_inversion.model_inversion_services import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.stan.pystan_service import PyStanService
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_dict_to_interface_ins, \
                                                                                        convert_params_names_from_ins
from tvb_epilepsy.top.scripts.fitting_scripts import *
from tvb_epilepsy.plot.plotter import Plotter

def set_hypotheses(head, config):
    # Formulate a VEP hypothesis manually
    hyp_builder = HypothesisBuilder(head.connectivity.number_of_regions, config).set_normalize(0.99)

    # Regions of Pathological Excitability hypothesis:
    x0_indices = [2, 25]
    x0_values = [0.01, 0.01]
    hyp_builder.set_x0_hypothesis(x0_indices, x0_values)

    # Regions of Model Epileptogenicity hypothesis:
    e_indices = list(range(head.connectivity.number_of_regions))
    e_indices.remove(2)
    e_indices.remove(25)
    e_values = np.zeros((head.connectivity.number_of_regions,)) + 0.01
    e_values[[1, 26]] = 0.99
    e_values = np.delete(e_values, [2, 25]).tolist()
    print(e_indices, e_values)
    hyp_builder.set_e_hypothesis(e_indices, e_values)

    # Regions of Connectivity hypothesis:
    # w_indices = []  # [(0, 1), (0, 2)]
    # w_values = []  # [0.5, 2.0]
    # hypo_builder.set_w_indices(w_indices).set_w_values(w_values)

    hypothesis1 = hyp_builder.build_hypothesis()

    e_indices = [1, 26]  # [1, 2, 25, 26]
    hypothesis2 = hyp_builder.build_hypothesis_from_file("clinical_hypothesis_postseeg", e_indices)
    # Change something manually if necessary
    # hypothesis2.x0_values = [0.01, 0.01]

    return (hypothesis1, hypothesis2)


def main_fit_sim_hyplsa(stan_model_name="vep_sde", empirical_file="",
                        observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value, sensors_lbls=[], sensor_id=0,
                        times_on_off=[], fitmethod="optimizing", stan_service="CmdStan",
                        fit_flag=True, config=Config(), **kwargs):

    def path(name):
        if len(name) > 0:
            return base_path + "_" + name + ".h5"
        else:
            return base_path + ".h5"

    # Prepare necessary services:
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    reader = H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)

    # Read head
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    sensors = head.get_sensors_id(sensor_ids=sensor_id)
    plotter.plot_head(head)

    # Set hypotheses:
    hypotheses = set_hypotheses(head, config)

    # ------------------------------Stan model and service--------------------------------------
    model_code_path = os.path.join(config.generic.STATS_MODELS_PATH, stan_model_name + ".stan")
    if isequal_string(stan_service, "CmdStan"):
        stan_service = CmdStanService(model_name=stan_model_name, model_code_path=model_code_path, fitmethod=fitmethod,
                                      config=config)
    else:
        stan_service = PyStanService(model_name=stan_model_name, model_code_path=model_code_path, fitmethod=fitmethod,
                                     config=config)
    stan_service.set_or_compile_model()

    for hyp in hypotheses[1:]:
        base_path = os.path.join(config.out.FOLDER_RES, hyp.name)
        # Set model configuration and compute LSA
        model_configuration, lsa_hypothesis = set_model_config_LSA(head, hyp, reader, config, K_unscaled=3*K_DEF)

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        # Create model inversion service (stateless)
        stats_model_file = path("StatsModel")
        model_data_file = path("ModelData")
        target_data_file = path("TargetData")
        if os.path.isfile(stats_model_file) and os.path.isfile(model_data_file) and os.path.isfile(target_data_file):
            # Read existing statistical model and model data...
            statistical_model = reader.read_statistical_model(stats_model_file)
            model_data = stan_service.load_model_data_from_file(model_data_path=model_data_file)
            target_data = reader.read_timeseries(target_data_file)
        else:
            model_inversion = SDEModelInversionService()

            # ...or generate a new statistical model and model data
            statistical_model = \
                SDEStatisticalModelBuilder(model_name="vep_sde", model_config=model_configuration,
                                           parameters=[XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value,
                                                        "tau1", "tau0","K", "x1init", "zinit", "sigma_init", "sigma_eq",
                                                        "sigma", "dX1t", "dZt", "epsilon", "scale", "offset"],
                                           xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                                           sde_mode=SDE_MODES.NONCENTERED.value, observation_model=observation_model,).\
                                                                                                       generate_model()

            # Update active model's active region nodes
            statistical_model = model_inversion.update_active_regions(statistical_model,
                                                                      e_values=lsa_hypothesis.e_values,
                                                                      lsa_propagation_strength=
                                                                            lsa_hypothesis.lsa_propagation_strengths,
                                                                      reset=True)

            # Now some scripts for settting and preprocessing target signals:
            if os.path.isfile(empirical_file):
                # -------------------------- Get empirical data (preprocess edf if necessary) --------------------------
                signals = set_empirical_data(empirical_file, path("ts_empirical"),
                                             head, sensors_lbls, sensor_id, times_on_off,
                                             label_strip_fun=lambda s: s.split("POL ")[-1], plotter=plotter,
                                             title_prefix=hyp.name)
            else:
                # -------------------------- Get simulated data (simulate if necessary) -------------------------------
                signals, simulator = \
                    set_simulated_target_data(path("ts"), model_configuration, head, lsa_hypothesis, statistical_model,
                                              sensor_id, times_on_off, config, plotter, title_prefix=hyp.name, **kwargs)
                statistical_model.ground_truth.update({"tau1": np.mean(simulator.model.tt),
                                                       "tau0": 1.0 / np.mean(simulator.model.r),
                                                       "sigma": np.mean(simulator.simulation_settings.noise_intensity)})

            # -------------------------- Select and set target data from signals ---------------------------------------
            if statistical_model.observation_model in OBSERVATION_MODELS.SEEG.value:
                model_inversion.auto_selection = "correlation-power"
                model_inversion.sensors_per_electrode = 2
            target_data, statistical_model, gain_matrix = \
                model_inversion.set_target_data_and_time(signals, statistical_model, head=head, sensors=sensors)

            plotter.plot_statistical_model(statistical_model, hyp.name + " Statistical Model")
            plotter.plot_raster({'Target Signals': target_data.squeezed}, target_data.time_line,
                                time_units=target_data.time_unit, title=hyp.name + ' Target Signals raster',
                                offset=0.1, labels=target_data.space_labels)
            plotter.plot_timeseries({'Target Signals': target_data.squeezed}, target_data.time_line,
                                    time_units=target_data.time_unit,
                                    title=hyp.name + ' Target Signals', labels=target_data.space_labels)

            writer.write_statistical_model(statistical_model, model_configuration.number_of_regions, stats_model_file)
            writer.write_timeseries(target_data, target_data_file)

            # Interface with INS stan models
            if stan_model_name.find("vep-fe-rev") >= 0:
                model_data = build_stan_model_dict_to_interface_ins(statistical_model, target_data.squeezed,
                                                                    gain_matrix, time=target_data.time_line)
            writer.write_dictionary(model_data, model_data_file)

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        num_warmup = 20
        if False:
            ests, samples, summary = stan_service.fit(debug=0, simulate=0, model_data=model_data, merge_outputs=False,
                                                      chains=2, refresh=1, num_warmup=num_warmup, num_samples=30,
                                                      max_depth=10, delta=0.8, save_warmup=1, plot_warmup=1, **kwargs)
            writer.write_generic(ests, path("FitEst"))
            writer.write_generic(samples, path("FitSamples"))
            if summary is not None:
                writer.write_generic(summary, path("FitSummary"))
        else:
            ests, samples, summary = stan_service.read_output()
            if fitmethod.find("sampl") >= 0:
                plotter.plot_HMC(samples, skip_samples=num_warmup, figure_name=hyp.name + " HMC NUTS trace")

        # Interface with INS stan models
        if stan_model_name.find("vep-fe-rev") >= 0:
            ests, samples, Rhat, model_data = \
                convert_params_names_from_ins([ests, samples, stan_service.get_Rhat(summary), model_data])

        # Pack fit samples time series into timeseries objects:
        samples, target_data = samples_to_timeseries(samples, model_data, target_data, head.connectivity.region_labels)

        # -------------------------- Plot fitting results: ------------------------------------------------------------
        plotter.plot_fit_results(ests, samples, model_data, target_data, statistical_model, stats={"Rhat": Rhat},
                                 pair_plot_params=["tau1", "K", "sigma", "epsilon", "scale", "offset"],
                                 region_violin_params=["x0", "x1init", "zinit"], regions_mode="active",
                                 regions_labels=head.connectivity.region_labels, trajectories_plot=True,
                                 connectivity_plot=False, skip_samples=num_warmup, title_prefix=hyp.name)


        # -------------------------- Reconfigure model after fitting:---------------------------------------------------
        for id_est, est in enumerate(ensure_list(ests)):
            fit_model_configuration_builder = \
                ModelConfigurationBuilder(hyp.number_of_regions, K=est["K"] * hyp.number_of_regions)
            x0_values_fit = model_configuration.x0_values
            x0_values_fit[statistical_model.active_regions] = \
                fit_model_configuration_builder._compute_x0_values_from_x0_model(est['x0'])
            hyp_fit = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions).\
                                          set_name('fit' + str(id_est) + "_" + hyp.name).\
                                          set_x0_hypothesis(list(statistical_model.active_regions),
                                                            x0_values_fit[statistical_model.active_regions]).\
                                          build_hypothesis()
            base_path = os.path.join(config.out.FOLDER_RES, hyp_fit.name)
            writer.write_hypothesis(hyp_fit, path(""))

            model_configuration_fit = \
                fit_model_configuration_builder.build_model_from_hypothesis(hyp_fit,  # est["MC"]
                                                                            model_configuration.model_connectivity)

            writer.write_model_configuration(model_configuration_fit, path("ModelConfig"))

            # Plot nullclines and equilibria of model configuration
            plotter.plot_state_space(model_configuration_fit, region_labels=head.connectivity.region_labels,
                                     special_idx=statistical_model.active_regions, model="6d", zmode="lin",
                                     figure_name=hyp_fit.name + "_Nullclines and equilibria")
        logger.info("Done!")


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    head_folder = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")
    SEEG_data = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "data/CC", "TVB3",
                             "raw/seeg/ts_seizure")

    if user_home == "/home/denis":
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "INScluster")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data,
                        output_base=output, separate_by_run=False)
        config.generic.C_COMPILER = "g++"
        config.generic.CMDSTAN_PATH = "/soft/stan/cmdstan-2.17.0"

    elif user_home == "/Users/lia.domide":
        config = Config(head_folder="/WORK/episense/tvb-epilepsy/data/TVB3/Head",
                        raw_data_folder="/WORK/episense/tvb-epilepsy/data/TVB3/ts_seizure")
        config.generic.CMDSTAN_PATH = "/WORK/episense/cmdstan-2.17.1"

    else:
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "fit")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data,
                        output_base=output, separate_by_run=False)

    # TVB3 larger preselection:
    sensors_lbls = \
        [u"B'1", u"B'2", u"B'3", u"B'4",
         u"F'1", u"F'2", u"F'3", u"F'4", u"F'5", u"F'6", u"F'7", u"F'8", u"F'9", u"F'10", u"F'11",
         u"G'1", u"G'2", u"G'3", u"G'4", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"G'13", u"G'14", u"G'15",
         u"L'1", u"L'2", u"L'3", u"L'4", u"L'5", u"L'6", u"L'7", u"L'8", u"L'9", u"L'10", u"L'11", u"L'12", u"L'13",
         u"M'1", u"M'2", u"M'3", u"M'7", u"M'8", u"M'9", u"M'10", u"M'11", u"M'12", u"M'13", u"M'14", u"M'15",
         u"O'1", u"O'2", u"O'3", u"O'6", u"O'7", u"O'8", u"O'9", u"O'10", u"O'11", u"O'12", # u"O'13"
         u"P'1", u"P'2", u"P'3", u"P'8", u"P'10", u"P'11", u"P'12", u"P'13", u"P'14", u"P'15", u"P'16",
         u"R'1", u"R'2", u"R'3", u"R'4", u"R'7", u"R'8", u"R'9",
         ]
    sensors_inds = [0, 1, 2, 3,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    28, 29, 30, 31, 36, 37, 38, 39, 40, 41, 42,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                    58, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                    74, 75, 76, 79, 80, 81, 82, 83, 84, 85, # 86,
                    90, 91, 92, 97, 99, 100, 101, 102, 103, 104, 105,
                    106, 107, 108, 109, 112, 113, 114
                    ]
    # TVB3 preselection:
    # sensors_lbls = [u"G'1", u"G'2", u"G'3", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"M'6", u"M'7", u"M'8", u"L'4",
    #                 u"L'5",  u"L'6", u"L'7", u"L'8", u"L'9"]
    # sensors_inds = [28, 29, 30, 35, 36, 37, 38, 39, 63, 64, 65, 47, 48, 49, 50, 51, 52]
    # TVB3 selection:
    # sensors_lbls = [u"G'1", u"G'2", u"G'11", u"G'12", u"M'7", u"M'8", u"L'5", u"L'6"]
    # sensors_inds = [28, 29, 38, 39, 64, 65, 48, 49]
    seizure = 'SZ1_0001.edf'
    times_on_off = [15.0, 35.0] * 1000
    # sensors_filename = "SensorsSEEG_116.h5"
    # # TVB4 preselection:
    # sensors_lbls = [u"D5", u"D6", u"D7",  u"D8", u"D9", u"D10", u"Z9", u"Z10", u"Z11", u"Z12", u"Z13", u"Z14",
    #                 u"S1", u"S2", u"S3", u"D'3", u"D'4", u"D'10", u"D'11", u"D'12", u"D'13", u"D'14"]
    # sensors_inds = [4, 5, 6, 7, 8, 9, 86, 87, 88, 89, 90, 91, 94, 95, 96, 112, 113, 119, 120, 121, 122, 123]
    # # TVB4:
    # seizure = 'SZ3_0001.edf'
    # sensors_filename = "SensorsSEEG_210.h5"
    # times_on_off = [20.0, 100.0]
    EMPIRICAL = True
    # stats_model_name = "vep_sde"
    stan_model_name = "vep-fe-rev-09dp"
    fitmethod = "sample"
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    fit_flag = True
    if EMPIRICAL:
        main_fit_sim_hyplsa(stan_model_name=stan_model_name, observation_model=observation_model,
                            empirical_file=os.path.join(config.input.RAW_DATA_FOLDER, seizure),
                            sensors_lbls=sensors_lbls, times_on_off=times_on_off, fitmethod=fitmethod,
                            stan_service="CmdStan", fit_flag=fit_flag, config=config)
    else:
        main_fit_sim_hyplsa(stan_model_name=stan_model_name, observation_model=observation_model,
                            sensors_lbls=sensors_lbls, times_on_off=[50.0, 550.0], fitmethod=fitmethod,
                            stan_service="CmdStan", fit_flag=fit_flag, config=config)
