# encoding=utf8


from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.model_inversion.statistical_models_builders import SDEStatisticalModelBuilder
from tvb_epilepsy.service.model_inversion.model_inversion_services import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_dict
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_dict_to_interface_ins
from tvb_epilepsy.top.scripts.fitting_scripts import *


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


def main_fit_sim_hyplsa(stan_model_name="vep_sde", empirical_file=None, dynamical_model = "EpileptorDP2D",
                        observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER,  times_on_off=[], time_units="msec",
                        sensors_lbls=[], sensors_inds=[], n_electrodes=None, sensors_per_electrode=2,
                        fitmethod="optimizing", stan_service="CmdStan", fit_flag=True, config=Config(), **kwargs):
    # Prepare necessary services:
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    reader = H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)

    # Read head
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    # plotter.plot_head(head)

    # Set hypotheses:
    hypotheses = set_hypotheses(head, config)

    # ------------------------------Stan model and service--------------------------------------
    stan_service = build_stan_service_and_model(stan_service, stan_model_name, fitmethod, config)
    # -------------------------------Reading data-----------------------------------

    for hyp in hypotheses[1:]:

        # Set model configuration and compute LSA
        model_configuration, lsa_hypothesis = set_model_config_LSA(head, hyp, reader, writer, plotter, config,
                                                                   K_unscaled=3*K_DEF)

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        # Create model inversion service (stateless)
        model_inversion = SDEModelInversionService(statistical_model.number_of_regions, **kwargs)
        stats_model_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_StatsModel.h5")
        model_data_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelData.h5")
        if os.path.isfile(stats_model_file) and os.path.isfile(model_data_file):
            # Read existing statistical model and model data...
            statistical_model = reader.read_generic(stats_model_file)
            statistical_model.model_config = model_configuration
            model_data = stan_service.load_model_data_from_file(model_data_path=model_data_file)
        else:
            # ...or generate a new statistical model and model data
            statistical_model = \
                SDEStatisticalModelBuilder(model_name="vep_sde", model_config=model_configuration,
                                           parameters=[XModes.X0MODE.value, "tau1", "K",
                                                        "x1init", "zinit", "sigma_init", "dX1t", "dZt",
                                                        "sigma", "epsilon", "scale_signal", "offset_signal"],
                                           xmode=XModes.X0MODE, priors_mode="informative",
                                           sigma_init=SIG_INIT_DEF, epsilon=EPSILON_DEF, sigma=SIGMA_DEF,
                                           sde_mode=SDE_MODES.NONCENTERED,  observation_model=observation_model). \
                                                                        generate_statistical_model(model_configuration)

            # Update active model's active region nodes
            statistical_model = model_inversion.update_active_regions(statistical_model, methods=["e_values", "LSA"],
                                                                      active_regions_th=0.2, reset=True)
            # plotter.plot_statistical_model(statistical_model, "Statistical Model")

            # Now set data:
            if os.path.isfile(empirical_file):
                signals_ts_dict, manual_selection, signals_labels = \
                                set_empirical_data(head, hyp.name, model_inversion, empirical_file,
                                                   sensors_inds, sensors_lbls, dynamical_model,
                                                   times_on_off, time_units, plotter)
            else:
                # -------------------------- Get simulated data (simulate if necessary) -------------------------------
                signals_ts_dict, manual_selection, signals_labels = \
                    set_simulated_data(head, hyp.name, lsa_hypothesis, model_configuration, model_inversion,
                                       statistical_model, sensors_inds, stan_model_name, dynamical_model, config)
            # -------------------------- Select and set observation signals -----------------------------------
            signals, time, stats_model, signals_labels, target_data = \
                    model_inversion.set_target_data_and_time(signals_ts_dict, statistical_model, dynamical_model,
                                                             sensors=head.get_sensors(), signals_labels=signals_labels,
                                                             manual_selection=manual_selection,
                                                             auto_selection="correlation-power", # auto_selection=False,
                                                             n_electrodes=n_electrodes,
                                                             sensors_per_electrode=sensors_per_electrode,
                                                             group_electrodes=True, normalization="baseline-amplitude",
                                                             plotter=plotter)
            # if len(model_inversion.signals_inds) < head.get_sensors_id().number_of_sensors:
            #     statistical_model = \
            #             model_inversion.update_active_regions_seeg(statistical_model)

            plot_target_signals(signals_ts_dict, signals, time, signals_labels, hyp.name,
                                model_inversion, statistical_model, writer, plotter, config)

            # Create model_data for stan
            model_data = build_stan_model_dict(statistical_model, signals, model_inversion,
                                                       time=time, sensors=head.get_sensors(), gain_matrix=None)
            writer.write_dictionary(model_data, os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelData.h5"))

            # Interface with INS stan models
            if stan_model_name.find("vep-fe-rev") >= 0:
                model_data, x0_star_mu, x_init_mu, z_init_mu = \
                    build_stan_model_dict_to_interface_ins(statistical_model, signals, model_inversion,
                                                       time=time, sensors=head.get_sensors(), gain_matrix=None)
                k_str = 'k'
            else:
                x0_star_mu = None
                x_init_mu = None
                z_init_mu = None
                k_str = 'K'

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        num_warmup = 200
        if fit_flag:
            ests, samples, summary = stan_service.fit(debug=0, simulate=0, model_data=model_data, merge_outputs=False,
                                                      chains=4, refresh=1, num_warmup=num_warmup, num_samples=300,
                                                      max_depth=10, delta=0.8, save_warmup=1, plot_warmup=1, **kwargs)
            writer.write_generic(ests, config.out.FOLDER_RES, hyp.name + "_fit_est.h5")
            writer.write_generic(samples, config.out.FOLDER_RES, hyp.name + "_fit_samples.h5")
            if summary is not None:
                writer.write_generic(summary, config.out.FOLDER_RES, hyp.name + "_fit_summary.h5")
                if isinstance(summary, dict):
                    R_hat = summary.get("R_hat", None)
                    if R_hat is not None:
                        R_hat = {"R_hat": R_hat}
        else:
            ests, samples, summary = stan_service.read_output()
            if isinstance(summary, dict):
                R_hat = summary.get("R_hat", None)
                if R_hat is not None:
                    R_hat = {"R_hat": R_hat}
            if fitmethod.find("sampl") >= 0:
                Plotter(config).plot_HMC(samples, skip_samples=num_warmup)
        ests = ensure_list(ests)

        # -------------------------- Plot fitting results: ------------------------------------------------------------
        plot_fitting_results(ests, samples, R_hat, stan_model_name, model_data, statistical_model, model_inversion,
                             model_configuration, lsa_hypothesis, plotter, x0_star_mu, x_init_mu, z_init_mu)


        # -------------------------- Reconfigure model after fitting:---------------------------------------------------
        for id_est, est in enumerate(ensure_list(ests)):
            fit_model_configuration_builder = \
                ModelConfigurationBuilder(hyp.number_of_regions, K=est[k_str] * hyp.number_of_regions)
            x0_values_fit = model_configuration.x0_values
            x0_values_fit[statistical_model.active_regions] = \
                fit_model_configuration_builder._compute_x0_values_from_x0_model(est['x0'])
            hyp_fit = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions).\
                                          set_name('fit' + str(id_est) + "_" + hyp.name).\
                                          set_x0_hypothesis(list(statistical_model.active_regions),
                                                            x0_values_fit[statistical_model.active_regions]).\
                                          build_hypothesis()
            writer.write_hypothesis(hyp_fit, os.path.join(config.out.FOLDER_RES, hyp_fit.name + ".h5"))

            model_configuration_fit = \
                fit_model_configuration_builder.build_model_from_hypothesis(hyp_fit,  # est["MC"]
                                                                            model_configuration.model_connectivity)

            writer.write_model_configuration(model_configuration_fit,
                                             os.path.join(config.out.FOLDER_RES, hyp_fit.name + "_ModelConfig.h5"))

            # Plot nullclines and equilibria of model configuration
            plotter.plot_state_space(model_configuration_fit,
                                     region_labels=model_inversion.region_labels,
                                     special_idx=statistical_model.active_regions,
                                     model="6d", zmode="lin",
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
                        output_base=output, separate_by_run=True)
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
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER
    fit_flag = True
    n_electrodes = 8
    sensors_per_electrode = 2
    if EMPIRICAL:
        main_fit_sim_hyplsa(stats_model_name=stan_model_name, observation_model=observation_model,
                            empirical_file=os.path.join(config.input.RAW_DATA_FOLDER, seizure),
                            times_on_off=times_on_off, time_units="sec", sensors_inds=sensors_inds, n_electrodes=8,
                            ensors_per_electrode=2, fitmethod=fitmethod, stan_service="CmdStan", fit_flag=fit_flag,
                            config=config)
    else:
        main_fit_sim_hyplsa(stats_model_name=stan_model_name, observation_model=observation_model,
                            times_on_off=[1000.0, 19000.0], sensors_inds=sensors_inds, n_electrodes=8,
                            sensors_per_electrode=2, fitmethod=fitmethod, stan_service="CmdStan", fit_flag=fit_flag,
                            config=config)
