
# import os
# from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
# from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
# from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
# from matplotlib import pyplot

from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter

if __name__ == "__main__":

    K = set_parameter("K", optimize_pdf=True, use="manual", K_lo=0.0, K_hi=10, K_pdf="lognormal",
                      K_pdf_params={"skew": 0.0, "mean": 1.0}, K_mean=1.0, K_std=0.1)

    print("Done")
    # stats_model_name = "vep-fe-rev-05"
    # model_code_path = os.path.join(FOLDER_VEP_HOME, stats_model_name + ".stan")
    # FOLDER_RES = os.path.join(FOLDER_VEP_HOME, stats_model_name + "_res")
    # # fitmethod = "sample"
    # # output_filepath=os.path.join(FOLDER_RES, "output.csv")
    # # diagnose_filepath = os.path.join(FOLDER_RES, "diagnostic.csv")
    # #
    # # stan_service = CmdStanService(model_name=stats_model_name, model=None, model_code=None,
    # #                               model_dir=FOLDER_VEP_HOME, model_code_path=model_code_path,
    # #                               fitmethod=fitmethod, random_seed=12345, init="random")
    # #
    # # est = stan_service.read_output_csv(output_filepath, merge=False)
    # # diagnostics = stan_service.read_output_csv(diagnose_filepath, merge=False)
    # # convert_to_h5_model(est).write_to_h5(FOLDER_RES, "estimate.h5")
    # # convert_to_h5_model(diagnostics).write_to_h5(FOLDER_RES, "diagnostics.h5")
    # est = read_h5_model(os.path.join(FOLDER_RES, "estimate.h5")).convert_from_h5_model()
    # # est = ensure_list(est)
    # # for id_est, this_est in enumerate(est):
    # #     model_inversion.plot_fit_results(this_est, statistical_model, signals, time=None,
    # #                                      seizure_indices=lsa_hypothesis.get_regions_disease(),
    # #                                      trajectories_plot=True, id_est=str(id_est))
    # #     # -------------------------- Reconfigure model after fitting:---------------------------------------------------
    # #     fit_model_configuration_service = \
    # #         ModelConfigurationService(hyp.number_of_regions, K=this_est['K'] * hyp.number_of_regions)
    # #     x0_values_fit = \
    # #         fit_model_configuration_service._compute_x0_values_from_x0_model(this_est['x0'])
    # #     hyp_fit = \
    # #         DiseaseHypothesis(head.connectivity.number_of_regions,
    # #                           excitability_hypothesis={tuple(range(model_configuration.n_regions)): x0_values_fit},
    # #                           epileptogenicity_hypothesis={}, connectivity_hypothesis={},
    # #                           name='fit' + str(id_est) + "_" + hyp.name)
    # #     model_configuration_fit = fit_model_configuration_service.configure_model_from_hypothesis(hyp_fit,
    # #                                                                                               this_est["MC"])
    # #     model_configuration_fit.write_to_h5(results_dir, hyp_fit.name + "_ModelConfig.h5")
    # #     # Plot nullclines and equilibria of model configuration
    # #     model_configuration_service.plot_state_space(model_configuration_fit,
    # #                                                    model_configuration_service.region_labels,
    # #                                                    special_idx=statistical_model.active_regions,
    # #                                                    model="6d", zmode="lin",
    # #                                                    figure_name=hyp_fit.name + "_Nullclines and equilibria")
