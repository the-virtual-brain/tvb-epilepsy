import numpy as np

from tvb_epilepsy.base.constants.module_constants import MAX_DISEASE_VALUE
from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.data_structures_utils import list_of_dicts_to_dicts_of_ndarrays, \
    dicts_of_lists_to_lists_of_dicts, linear_index_to_coordinate_tuples
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.scripts.hypothesis_scripts import start_lsa_run
from tvb_epilepsy.service.pse_service import PSEService
from tvb_epilepsy.service.sampling.stochastic_sampling_service import StochasticSamplingService


###
# These functions are helper functions to run parameter search exploration (pse) for Linear Stability Analysis (LSA).
###
def pse_from_lsa_hypothesis(lsa_hypothesis, model_connectivity, region_labels,
                            n_samples, param_range=0.1, global_coupling=[],
                            healthy_regions_parameters=[],
                            model_configuration_service=None, lsa_service=None,
                            save_flag=False, folder_res=FOLDER_RES, filename=None, logger=None, **kwargs):
    if logger is None:
        logger = initialize_logger(__name__)
    all_regions_indices = range(lsa_hypothesis.number_of_regions)
    disease_indices = lsa_hypothesis.get_regions_disease_indices()
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    pse_params = {"path": [], "indices": [], "name": [], "samples": []}
    sampler = StochasticSamplingService(n_samples=n_samples, random_seed=kwargs.get("random_seed", None))
    # First build from the hypothesis the input parameters of the parameter search exploration.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)
    for ii in range(len(lsa_hypothesis.x0_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(region_labels[lsa_hypothesis.x0_indices[ii]]) + " Excitability")

        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            sampler.generate_samples(parameter=(lsa_hypothesis.x0_values[ii],  # loc
                                                param_range / 3.0),  # scale
                                     probability_distribution="norm",
                                     high=MAX_DISEASE_VALUE, shape=(1,)))
        # pse_params["samples"].append(
        #     sampler.generate_samples(parameter=(lsa_hypothesis.x0_values[ii] - param_range,  # loc
        #                                         2 * param_range),                            # scale
        #                              probability_distribution="uniform",
        #                              high=MAX_DISEASE_VALUE, shape=(1,)))
    for ii in range(len(lsa_hypothesis.e_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(region_labels[lsa_hypothesis.e_indices[ii]]) + " Epileptogenicity")

        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            sampler.generate_samples(parameter=(lsa_hypothesis.e_values[ii],  # loc
                                                param_range / 3.0),  # scale
                                     probability_distribution="norm",
                                     high=MAX_DISEASE_VALUE, shape=(1,)))
        # pse_params["samples"].append(
        #     sampler.generate_samples(parameter=(lsa_hypothesis.e_values[ii] - param_range,  # loc
        #                                         2 * param_range),  # scale
        #                              probability_distribution="uniform",
        #                              high=MAX_DISEASE_VALUE, shape=(1,)))
    for ii in range(len(lsa_hypothesis.w_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(lsa_hypothesis.w_indices[ii], model_connectivity.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(region_labels[inds[0][0]]) + "-" +
                                      str(region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds), + "]")
        # Now generate samples using a truncated normal distribution
        pse_params["samples"].append(
            sampler.generate_samples(parameter=(lsa_hypothesis.w_values[ii],  # loc
                                                param_range * lsa_hypothesis.w_values[ii]),  # scale
                                     probability_distribution="norm", low=0.0, shape=(1,)))
    kloc = model_configuration_service.K_unscaled[0]
    for val in global_coupling:
        pse_params["path"].append("model_configuration_service.K_unscaled")
        inds = val.get("indices", all_regions_indices)
        if np.all(inds == all_regions_indices):
            pse_params["name"].append("Global coupling")
        else:
            pse_params["name"].append("Afferent coupling[" + str(inds) + "]")
        pse_params["indices"].append(inds)

        # Now generate samples susing a truncated normal distribution
        pse_params["samples"].append(
            sampler.generate_samples(parameter=(1.0,   # loc
                                                100),  # scale
                                     probability_distribution="uniform", low=1.0, shape=(1,)))
        # pse_params["samples"].append(
        #     sampler.generate_samples(parameter=(kloc,  # loc
        #                                         30 * param_range),  # scale
        #                              probability_distribution="norm", low=0.0, shape=(1,)))
    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)
    # Add a random jitter to the healthy regions if required...:
    for val in healthy_regions_parameters:
        inds = val.get("indices", healthy_indices)
        name = val.get("name", "x0_values")
        n_params = len(inds)
        samples = sampler.generate_samples(parameter=(0.0,  # loc
                                                      param_range / 10),  # scale
                                           probability_distribution="norm", shape=(n_params,))
        for ii in range(n_params):
            pse_params_list.append({"path": "model_configuration_service." + name, "samples": samples[ii],
                                    "indices": [inds[ii]], "name": name})
    # Now run pse service to generate output samples:
    pse = PSEService("LSA", hypothesis=lsa_hypothesis, params_pse=pse_params_list)
    pse_results, execution_status = pse.run_pse(model_connectivity, grid_mode=False, lsa_service_input=lsa_service,
                                                model_configuration_service_input=model_configuration_service)
    # Call to new PSEService:
    # pse = LSAPSEService(lsa_hypothesis, pse_params_list)
    # pse_results, execution_status = pse.run_pse(model_connectivity, False, lsa_service, model_configuration_service)
    pse_results = list_of_dicts_to_dicts_of_ndarrays(pse_results)
    for key in pse_results.keys():
        pse_results[key + "_mean"] = np.mean(pse_results[key], axis=0)
        pse_results[key + "_std"] = np.std(pse_results[key], axis=0)
    if save_flag:
        logger.info(pse.__repr__())
        if not(isinstance(filename, basestring)):
            filename = "LSA_PSA"
        pse.write_to_h5(folder_res, filename + "_pse_service.h5")
        h5_model = convert_to_h5_model(pse_results)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", lsa_hypothesis.number_of_regions)
        h5_model.write_to_h5(folder_res, filename + ".h5")
    return pse_results, pse_params_list


def pse_from_hypothesis(hypothesis, model_connectivity, region_labels, n_samples, param_range=0.1, global_coupling=[],
                        healthy_regions_parameters=[], save_flag=False, folder_res=FOLDER_RES, filename=None,
                        logger=None, **kwargs):
    if logger is None:
        logger = initialize_logger(__name__)
    # Compute lsa for this hypothesis before the parameter search:
    logger.info("Running hypothesis: " + hypothesis.name)
    model_configuration_service, model_configuration, lsa_service, lsa_hypothesis = \
        start_lsa_run(hypothesis, model_connectivity, logger)
    pse_results, pse_params_list = pse_from_lsa_hypothesis(lsa_hypothesis, model_connectivity, region_labels,
                                                           n_samples, param_range, global_coupling,
                                                           healthy_regions_parameters,
                                                           model_configuration_service, lsa_service,
                                                           save_flag, folder_res=folder_res, filename=filename,
                                                           logger=logger, **kwargs)
    return model_configuration, lsa_service, lsa_hypothesis, pse_results, pse_params_list
