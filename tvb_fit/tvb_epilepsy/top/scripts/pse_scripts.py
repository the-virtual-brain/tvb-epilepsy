import os
import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.model_constants import MAX_DISEASE_VALUE
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_fit.samplers.probabilistic_sampler import ProbabilisticSampler
from tvb_fit.tvb_epilepsy.top.scripts.lsa_scripts import lsa_done, run_lsa, from_hypothesis_to_model_config_lsa

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import list_of_dicts_to_dicts_of_ndarrays, \
    dicts_of_lists_to_lists_of_dicts, linear_index_to_coordinate_tuples
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard


LOG = initialize_logger(__name__)


###
# These functions are helper functions to run parameter search exploration (pse) for Linear Stability Analysis (LSA).
###

def run_lsa_pse(hypothesis, modelconfig, modelconfig_builder, lsa_service, region_labels, random_seed=0,
                lsa_pse_path="", figname="", hypo_figsfolder="",
                writer=None, plotter=None, logger=None, config=Config(), **lsa_pse_params):
    if not lsa_done(hypothesis):
        from tvb_fit.tvb_epilepsy.top.workflow import hypo_prefix
        hypo_prfx = hypo_prefix(hypothesis.name, hypothesis.type) + "_"
        hypo_path = os.path.join(os.path.dirname(lsa_pse_path), hypo_prfx + ".h5")
        hypothesis, lsa_service = run_lsa(hypothesis, modelconfig, lsa_service, region_labels,
                                          hypo_path, hypo_prfx +"LSA", hypo_figsfolder,
                                          writer, plotter, config)
    n_samples = lsa_pse_params.get("n_samples", 100)
    param_range = lsa_pse_params.get("param_range", 0.1)
    all_regions_indices = np.arange(hypothesis.number_of_regions)
    healthy_regions_indices = np.delete(all_regions_indices, hypothesis.regions_disease_indices).tolist()
    # This a specific example function. Overload for different applications
    pse_params = {"path": [], "indices": [], "name": [], "samples": []}
    lsa_pse_sampler = \
        ProbabilisticSampler(n_samples=n_samples, random_seed=random_seed)

    # First build from the hypothesis the input parameters of the parameter search exploration.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)

    # x0 parameters'sampling
    for ii in range(len(hypothesis.x0_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(region_labels[hypothesis.x0_indices[ii]]) + " Excitability")

        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.x0_values[ii],  # loc
                                                        param_range / 3.0),  # scale
                                                        probability_distribution="norm",
                                                        high=MAX_DISEASE_VALUE, shape=(1,)))

    # e parameters'sampling
    for ii in range(len(hypothesis.e_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(region_labels[hypothesis.e_indices[ii]]) + " Epileptogenicity")
        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.e_values[ii],  # loc
                                                        param_range / 3.0),  # scale
                                                       probability_distribution="norm",
                                                       high=MAX_DISEASE_VALUE, shape=(1,)))

    # w parameters'sampling
    for ii in range(len(hypothesis.w_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(hypothesis.w_indices[ii], modelconfig.connectivity.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(region_labels[inds[0][0]]) + "-" +
                                      str(region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds) + "]")
        # Now generate samples using a truncated normal distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.w_values[ii],  # loc
                                                        param_range * hypothesis.w_values[ii]),  # scale
                                                        probability_distribution="norm", low=0.0, shape=(1,)))

    # Global coupling jitter
    kloc = modelconfig_builder.K_unscaled[0]
    pse_params["path"].append("model_configuration_builder.K_unscaled")
    pse_params["indices"].append(all_regions_indices)
    # Now generate samples using a truncated normal distribution
    pse_params["samples"].append(
        lsa_pse_sampler.generate_samples(parameter=(0.1 * kloc,  # loc
                                                    2 * kloc),  # scale
                                                    probability_distribution="uniform", low=1.0, shape=(1,)))
    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)

    # Add a random jitter to the healthy regions...:
    n_params = len(healthy_regions_indices)
    samples = lsa_pse_sampler.generate_samples(parameter=(0.0,  # loc
                                                          param_range / 10),  # scale
                                                          probability_distribution="norm", shape=(n_params,))

    for ii in range(n_params):
        pse_params_list.append({"path": "model_configuration_builder.e_values", "samples": samples[ii],
                                "indices": [healthy_regions_indices[ii]], "name": "e_values"})

    # Now run pse service to generate output samples:
    lsa_pse_service = LSAPSEService(hypothesis=hypothesis, params_pse=pse_params_list)
    lsa_pse_results, execution_status = \
        lsa_pse_service.run_pse(modelconfig.connectivity, False, modelconfig_builder, lsa_service)
    logger.info(lsa_pse_service.__repr__())
    lsa_pse_results = list_of_dicts_to_dicts_of_ndarrays(lsa_pse_results)

    # Compute statistical estimates across samples:
    for key in lsa_pse_results.keys():
        lsa_pse_results[key + "_mean"] = np.mean(lsa_pse_results[key], axis=0)
        lsa_pse_results[key + "_std"] = np.std(lsa_pse_results[key], axis=0)

    # Plot samples
    if plotter:
        plotter.plot_lsa(hypothesis, modelconfig,
                         lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, region_labels, lsa_pse_results,
                         title=figname, lsa_service=lsa_service)
        if os.path.isdir(hypo_figsfolder) and (hypo_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(hypo_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("LSA_PSE")))

    if writer:
        writer.write_dictionary(lsa_pse_results, lsa_pse_path)

    return lsa_pse_results, lsa_pse_service, lsa_pse_sampler


def pse_from_lsa_hypothesis(lsa_hypothesis, modelconfig, modelconfig_builder, lsa_service, region_labels,
                            n_samples=100, param_range=0.1, writer=None, plotter=None, logger=LOG, config=Config(),
                            **pse_params):
    lsa_pse_params = {"n_samples": n_samples, "param_range": param_range}
    lsa_pse_params.update(pse_params)
    return run_lsa_pse(lsa_hypothesis, modelconfig, modelconfig_builder, lsa_service, region_labels, random_seed=0,
                       lsa_pse_path=os.path.join(config.out.FOLDER_RES, "LSA_PSE.h5"),
                       writer=writer, plotter=plotter, logger=logger, config=config, **lsa_pse_params)


def pse_from_hypothesis(hypothesis, head, n_samples=100, param_range=0.1, lsa_params={}, model_config_kwargs={},
                       writer=None, plotter=None, logger=LOG, config=Config(), **pse_params):
    model_configuration, lsa_hypothesis, model_configuration_builder, lsa_service = \
        from_hypothesis_to_model_config_lsa(hypothesis, head, model_params=model_config_kwargs, lsa_params=lsa_params,
                                            writer=writer, plotter=plotter, logger=logger, config=Config())

    pse_results, pse_service, pse_sampler = \
        pse_from_lsa_hypothesis(lsa_hypothesis, model_configuration, model_configuration_builder, lsa_service,
                                head.connectivity.region_labels, n_samples, param_range,
                                writer=writer, plotter=plotter,logger=logger, config=config, **pse_params)

    return model_configuration, lsa_service, lsa_hypothesis, pse_results, pse_service, pse_sampler

