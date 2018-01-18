
# import os
import numpy as np
# from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
# from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
# from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
# from matplotlib import pyplot

from tvb_epilepsy.base.constants.model_constants import *
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter

if __name__ == "__main__":
    # K_mean = 10*K_DEF/87
    # K_std = np.min([K_mean - K_MIN, K_MAX - K_mean]) / 6.0
    # K = set_parameter("K", optimize_pdf=True, use="manual", K_lo=K_MIN, K_hi=K_MAX, K_pdf="lognormal",
    #                   K_pdf_params={"skew": 0.0, "mean": K_mean/K_std}, K_mean=K_mean, K_std=K_std) # K_mean=K_mean,
    #
    # tau1_mean = 0.5
    # tau1_std = np.min([tau1_mean - TAU1_MIN, TAU1_MAX - tau1_mean]) / 6.0
    # tau1 = set_parameter("tau1", optimize_pdf=True, use="manual", tau1_lo=TAU1_MIN, tau1_hi=TAU1_MAX, tau1_pdf="lognormal",
    #                      tau1_pdf_params={"skew": 0.0, "mean": tau1_mean / tau1_std}, tau1_mean=tau1_mean, tau1_std=tau1_std)
    # tau1.plot_stochastic_parameter(figure_name="tau1 parameter")
    #
    # tau0_mean = 10.0
    # tau0_std = np.min([tau0_mean - TAU0_MIN, TAU0_MAX - tau0_mean]) / 6.0
    # tau0 = set_parameter("tau0", optimize_pdf=True, use="manual", tau0_lo=TAU0_MIN, tau0_hi=TAU0_MAX, tau0_pdf="lognormal",
    #                      tau0_pdf_params={"skew": 0.0, "mean": tau0_mean / tau0_std}, tau0_mean=tau0_mean, tau0_std=tau0_std)
    # tau0.plot_stochastic_parameter(figure_name="tau0 parameter")

    sig_std = SIG_DEF / 6.0
    sig = set_parameter("sig", optimize_pdf=True, use="manual", sig_lo=0.1*SIG_DEF, sig_hi=10*SIG_DEF,
                         sig_pdf="gamma",
                         sig_pdf_params={"skew": 0.0, "mean": SIG_DEF/sig_std}, sig_mean=SIG_DEF, sig_std=sig_std) #
    sig.plot_stochastic_parameter(figure_name="sig parameter")
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
