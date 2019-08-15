import os
import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.model_constants import MAX_DISEASE_VALUE
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_fit.samplers.salib_sampler_interface import SalibSamplerInterface
from tvb_fit.samplers.probabilistic_sampler import ProbabilisticSampler
from tvb_fit.service.sensitivity_analysis_service import SensitivityAnalysisService, METHODS
from tvb_fit.tvb_epilepsy.top.scripts.hypothesis_scripts import start_lsa_run

from tvb_utils.log_error_utils import initialize_logger, raise_value_error
from tvb_utils.data_structures_utils import list_of_dicts_to_dicts_of_ndarrays, \
    dicts_of_lists_to_lists_of_dicts, linear_index_to_coordinate_tuples
from tvb_io.h5_writer import H5Writer


# These functions are helper functions to run sensitivity analysis parameter search exploration (pse)
# for Linear Stability Analysis (LSA).
def sensitivity_analysis_pse_from_lsa_hypothesis(n_samples, lsa_hypothesis, connectivity_matrix,
                                                 model_configuration_builder, lsa_service, region_labels,
                                                 method="sobol", half_range=0.1, global_coupling=[],
                                                 healthy_regions_parameters=[],
                                                 save_services=False, config=Config(), **kwargs):
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    method = method.lower()
    if np.in1d(method, METHODS):
        if np.in1d(method, ["delta", "dgsm"]):
            sampler = "latin"
        elif method == "sobol":
            sampler = "saltelli"
        elif method == "fast":
            sampler = "fast_sampler"
        else:
            sampler = method
    else:
        raise_value_error("Method " + str(method) + " is not one of the available methods " + str(METHODS) + " !")
    all_regions_indices = range(lsa_hypothesis.number_of_regions)
    disease_indices = lsa_hypothesis.regions_disease_indices
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    pse_params = {"path": [], "indices": [], "name": [], "low": [], "high": []}
    n_inputs = 0
    # First build from the hypothesis the input parameters of the sensitivity analysis.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)
    for ii in range(len(lsa_hypothesis.x0_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(region_labels[lsa_hypothesis.x0_indices[ii]]) +
                                  " Excitability")
        pse_params["low"].append(lsa_hypothesis.x0_values[ii] - half_range)
        pse_params["high"].append(np.min([MAX_DISEASE_VALUE, lsa_hypothesis.x0_values[ii] + half_range]))
    for ii in range(len(lsa_hypothesis.e_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(region_labels[lsa_hypothesis.e_indices[ii]]) +
                                  " Epileptogenicity")
        pse_params["low"].append(lsa_hypothesis.e_values[ii] - half_range)
        pse_params["high"].append(np.min([MAX_DISEASE_VALUE, lsa_hypothesis.e_values[ii] + half_range]))
    for ii in range(len(lsa_hypothesis.w_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(lsa_hypothesis.w_indices[ii], connectivity_matrix.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(region_labels[inds[0][0]]) + "-" +
                                      str(region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds), + "]")
            pse_params["low"].append(np.max([lsa_hypothesis.w_values[ii] - half_range, 0.0]))
            pse_params["high"].append(lsa_hypothesis.w_values[ii] + half_range)
    for val in global_coupling:
        n_inputs += 1
        pse_params["path"].append("model.configuration.service.K_unscaled")
        inds = val.get("indices", all_regions_indices)
        if np.all(inds == all_regions_indices):
            pse_params["name"].append("Global coupling")
        else:
            pse_params["name"].append("Afferent coupling[" + str(inds) + "]")
        pse_params["indices"].append(inds)
        pse_params["low"].append(val.get("low", 0.0))
        pse_params["high"].append(val.get("high", 2.0))
    # Now generate samples suitable for sensitivity analysis
    sampler = SalibSamplerInterface(n_samples=n_samples, sampler=sampler, random_seed=kwargs.get("random_seed", None))
    input_samples = sampler.generate_samples(low=pse_params["low"], high=pse_params["high"], **kwargs)
    n_samples = input_samples.shape[1]
    pse_params.update({"samples": [np.array(value) for value in input_samples.tolist()]})
    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)
    # Add a random jitter to the healthy regions if required...:
    sampler = ProbabilisticSampler(n_samples=n_samples, random_seed=kwargs.get("random_seed", None))
    for val in healthy_regions_parameters:
        inds = val.get("indices", healthy_indices)
        name = val.get("name", "x0_values")
        n_params = len(inds)
        samples = sampler.generate_samples(parameter=(kwargs.get("loc", 0.0),  # loc
                                                      kwargs.get("scale", 2 * half_range)),  # scale
                                           probability_distribution="uniform", low=0.0, shape=(n_params,))
        for ii in range(n_params):
            pse_params_list.append({"path": "model_configuration_builder." + name, "samples": samples[ii],
                                    "indices": [inds[ii]], "name": name})
    # Now run pse service to generate output samples:
    pse = LSAPSEService(hypothesis=lsa_hypothesis, params_pse=pse_params_list)
    pse_results, execution_status = pse.run_pse(connectivity_matrix, False, model_configuration_builder, lsa_service)
    pse_results = list_of_dicts_to_dicts_of_ndarrays(pse_results)
    # Now prepare inputs and outputs and run the sensitivity analysis:
    # NOTE!: Without the jittered healthy regions which we don' want to include into the sensitivity analysis!
    inputs = dicts_of_lists_to_lists_of_dicts(pse_params)
    outputs = [{"names": ["LSA Propagation Strength"], "values": pse_results["lsa_propagation_strengths"]}]
    sensitivity_analysis_service = SensitivityAnalysisService(inputs, outputs, method=method,
                                                              calc_second_order=kwargs.get("calc_second_order", True),
                                                              conf_level=kwargs.get("conf_level", 0.95))
    results = sensitivity_analysis_service.run(**kwargs)
    if save_services:
        logger.info(pse.__repr__())
        writer = H5Writer()
        writer.write_pse_service(pse, os.path.join(config.out.FOLDER_RES, method + "_test_pse_service.h5"))
        logger.info(sensitivity_analysis_service.__repr__())
        writer.write_sensitivity_analysis_service(sensitivity_analysis_service,
                                                  os.path.join(config.out.FOLDER_RES, method + "_test_sa_service.h5"))
    return results, pse_results


def sensitivity_analysis_pse_from_hypothesis(n_samples, hypothesis, connectivity_matrix, region_labels,
                                             method="sobol", half_range=0.1, global_coupling=[],
                                             healthy_regions_parameters=[], save_services=False, config=Config(),
                                             model_config_kwargs={}, **kwargs):
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    # Compute lsa for this hypothesis before sensitivity analysis:
    logger.info("Running hypothesis: " + hypothesis.name)
    model_configuration_builder, model_configuration, lsa_service, lsa_hypothesis = \
        start_lsa_run(hypothesis, connectivity_matrix, config, **model_config_kwargs)
    results, pse_results = sensitivity_analysis_pse_from_lsa_hypothesis(n_samples, lsa_hypothesis, connectivity_matrix,
                                                                        model_configuration_builder, lsa_service,
                                                                        region_labels, method, half_range,
                                                                        global_coupling, healthy_regions_parameters,
                                                                        save_services, config, **kwargs)
    return model_configuration_builder, model_configuration, lsa_service, lsa_hypothesis, results, pse_results
