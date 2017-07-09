import warnings
import numpy as np

from SALib.analyze import sobol, delta, fast, morris, dgsm,  ff

from tvb_epilepsy.base.utils import list_of_dicts_to_dicts_of_ndarrays


METHODS = ["sobol", "latin", "delta", "dgsm", "fast", "fast_sampler", "morris", "ff", "fractional_factorial"]


# TODO: make sensitivity_analysis_from_hypothesis() helper function

# TODO: make example-testing function __main__

# TODO: make __repr__ and prepare h5_model functions

# This function is a helper function to run sensitivity analysis parameter search exploration (pse)
# for Linear Stability Analysis (LSA).

def sensitivity_analysis_pse_from_hypothesis(hypothesis, n_samples, method="latin", half_range=0.1, global_coupling=[],
                                             healthy_regions_parameters=[], model_configuration=None,
                                             model_configuration_service=None, lsa_service=None, **kwargs):

    from tvb_epilepsy.base.constants import MAX_DISEASE_VALUE
    from tvb_epilepsy.base.utils import linear_index_to_coordinate_tuples, dicts_of_lists_to_lists_of_dicts
    from tvb_epilepsy.base.sample_service import StochasticSampleService
    from tvb_epilepsy.base.pse_service import PSE_service
    from tvb_epilepsy.base.sensitivity_analysis_service import SensitivityAnalysisService

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
        raise ValueError(
            "Method " + str(method) + " is not one of the available methods " + str(METHODS) + " !")

    all_regions_indices = range(hypothesis.get_number_of_regions())
    disease_indices = hypothesis.get_regions_disease_indices()
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()

    pse_params = {"path": [], "indices": [], "name": [], "bounds": []}
    n_inputs = 0

    # First build from the hypothesis the input parameters of the sensitivity analysis.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)
    for ii in range(len(hypothesis.x0_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.x0_indices[ii]]) +
                                  " Excitability")
        pse_params["bounds"].append([hypothesis.x0_values[ii]-half_range,
                       np.min([MAX_DISEASE_VALUE, hypothesis.x0_values[ii]+half_range])])

    for ii in range(len(hypothesis.e_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.x0_indices[ii]]) +
                                  " Epileptogenicity")
        pse_params["bounds"].append([hypothesis.e_values[ii]-half_range,
                       np.min([MAX_DISEASE_VALUE, hypothesis.e_values[ii]+half_range])])

    for ii in range(len(hypothesis.w_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(hypothesis.w_indices[ii], hypothesis.connectivity.weights.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(hypothesis.connectivity.region_labels[inds[0][0]]) + "-" +
                                str(hypothesis.connectivity.region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds), + "]")
            pse_params["bounds"].append([np.max([hypothesis.w_values[ii]-half_range, 0.0]),
                                                 hypothesis.w_values[ii]+half_range])

    for val in global_coupling:
        n_inputs += 1
        pse_params["path"].append("model.configuration.service.K_unscaled")
        inds = val.get("indices", all_regions_indices)
        if np.all(inds == all_regions_indices):
            pse_params["name"].append("Global coupling")
        else:
            pse_params["name"].append("Afferent coupling[" + str(inds) + "]")
        pse_params["indices"].append(inds)
        pse_params["bounds"].append(val["bounds"])

    # Now generate samples suitable for sensitivity analysis
    sampler = StochasticSampleService(n_samples=n_samples, n_outputs=n_inputs, sampler=sampler, trunc_limits={},
                                      sampling_module="salib", random_seed=kwargs.get("random_seed", None),
                                      bounds=pse_params["bounds"])

    input_samples = sampler.generate_samples(**kwargs)
    n_samples = input_samples.shape[1]
    pse_params.update({"samples": [np.array(value) for value in input_samples.tolist()]})

    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)

    # Add a random jitter to the healthy regions if required...:
    for val in healthy_regions_parameters:
        inds = val.get("indices", healthy_indices)
        name = val.get("name", "x0")
        n_params = len(inds)
        sampler = StochasticSampleService(n_samples=n_samples, n_outputs=n_params, sampler="uniform",
                                          trunc_limits={"low": 0.0}, sampling_module="scipy",
                                          random_seed=kwargs.get("random_seed", None),
                                          loc=kwargs.get("loc", 0.0), scale=kwargs.get("scale", 2*half_range))

        samples = sampler.generate_samples(**kwargs)
        for ii in range(n_params):
            pse_params_list.append({"path": "model_configuration_service." + name, "samples": samples[ii],
                                    "indices": [inds[ii]], "name": name})

    # Now run pse service to generate output samples:

    pse = PSE_service("LSA", hypothesis=hypothesis, params_pse=pse_params_list)
    pse_results, execution_status = pse.run_pse(grid_mode=False, lsa_service_input=lsa_service,
                                                model_configuration_service_input=model_configuration_service)

    pse_results = list_of_dicts_to_dicts_of_ndarrays(pse_results)

    # Now prepare inputs and outputs and run the sensitivity analysis:
    # NOTE!: Without the jittered healthy regions which we don' want to include into the sensitivity analysis!
    inputs = dicts_of_lists_to_lists_of_dicts(pse_params)

    outputs = [{"names": ["LSA Propagation Strength"], "values": pse_results["propagation_strengths"]}]
    sensitivity_analysis_service = SensitivityAnalysisService(inputs, outputs, method=method,
                                                              calc_second_order=kwargs.get("calc_second_order", True),
                                                              conf_level=kwargs.get("conf_level", 0.95))

    results = sensitivity_analysis_service.run(**kwargs)

    return results, pse_results


class SensitivityAnalysisService(object):

    def __init__(self, inputs, outputs, method="delta", calc_second_order=True, conf_level=0.95):

        self._set_method(method)
        self._set_calc_second_order(calc_second_order)
        self._set_conf_level(conf_level)

        self.n_samples = []
        self.input_names = []
        self.input_bounds = []
        self.input_samples = []
        self.n_inputs = len(inputs)

        for input in inputs:

            self.input_names.append(input["name"])

            samples = np.array(input["samples"]).flatten()
            self.n_samples.append(samples.size)
            self.input_samples.append(samples)

            self.input_bounds.append(input.get("bounds", [samples.min(), samples.max()]))

        if np.all(np.array(self.n_samples) == self.n_samples[0]):
            self.n_samples = self.n_samples[0]
        else:
            raise ValueError("Not all input parameters have equal number of samples!: " + str(self.n_samples))

        self.input_samples = np.array(self.input_samples).T

        self.n_outputs = 0
        self.output_values = []
        self.output_names = []

        for output in outputs:

            if output["values"].size == self.n_samples:
                n_outputs = 1
                self.output_values.append(output["values"].flatten())
            else:
                if output["values"].shape[0] == self.n_samples:
                    self.output_values.append(output["values"])
                    n_outputs = output["values"].shape[1]
                elif output["values"].shape[1] == self.n_samples:
                    self.output_values.append(output["values"].T)
                    n_outputs = output["values"].shape[0]
                else:
                    raise ValueError("Non of the dimensions of output samples: " + str(output["values"].shape) +
                                     " matches n_samples = " + str(self.n_samples) + " !")
            self.n_outputs += n_outputs

            if n_outputs > 1 and len(output["names"]) == 1:
                self.output_names += np.array(["%s[%d]" % l for l in zip(np.repeat(output["names"][0], n_outputs),
                                                                         range(n_outputs))]).tolist()
            else:
                self.output_names += output["names"]

        self.output_values = np.vstack(self.output_values)

        self.problem = {}
        self.other_parameters = {}

    def _set_method(self, method):
        method = method.lower()
        if np.in1d(method, METHODS):
            self.method = method
        else:
            raise ValueError(
                "Method " + str(method) + " is not one of the available methods " + str(METHODS) + " !")

    def _set_calc_second_order(self, calc_second_order):
        if isinstance(calc_second_order, bool):
            self.calc_second_order = calc_second_order
        else:
            raise ValueError("calc_second_order = " + str(calc_second_order) + "is not a boolean as it should!")

    def _set_conf_level(self, conf_level):
        if isinstance(conf_level, float) and conf_level > 0.0 and conf_level < 1.0:
            self.conf_level = conf_level
        else:
            raise ValueError("conf_level = " + str(conf_level) +
                             "is not a float in the (0.0, 1.0) interval as it should!")

    def _update_parameters(self, method=None, calc_second_order=None, conf_level=None):

        if method is not None:
            self._set_method(method)

        if calc_second_order is not None:
            self._calc_set_second_order(calc_second_order)

        if conf_level is not None:
            self._set_conf_level(conf_level)


    def run(self, input_ids=None, output_ids=None, method=None, calc_second_order=None, conf_level=None, **kwargs):

        self._update_parameters(method, calc_second_order, conf_level)

        self.other_parameters = kwargs

        if input_ids is None:
            input_ids = range(self.n_inputs)

        self.problem = {"num_vars": len(input_ids),
                        "names": np.array(self.input_names)[input_ids].tolist(),
                        "bounds": np.array(self.input_bounds)[input_ids].tolist()}

        if output_ids is None:
            output_ids = range(self.n_outputs)

        n_outputs = len(output_ids)

        if self.method.lower() == "sobol":
            warnings.warn("'sobol' method requires 'saltelli' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # calc_second_order (bool): Calculate second-order sensitivities (default True)
            # num_resamples (int): The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float): The confidence interval level (default 0.95)
            # print_to_console (bool): Print results directly to console (default False)
            self.analyzer = lambda output: sobol.analyze(self.problem, output, **kwargs)

        elif np.in1d(self.method.lower(), ["latin", "delta"]):
            warnings.warn("'latin' sampling scheme is recommended for 'delta' method!")
            # Additional keyword parameters and their defaults:
            # num_resamples (int): The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float): The confidence interval level (default 0.95)
            # print_to_console (bool): Print results directly to console (default False)
            self.analyzer = lambda output: delta.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                         **kwargs)

        elif np.in1d(self.method.lower(), ["fast", "fast_sampler"]):
            warnings.warn("'fast' method requires 'fast_sampler' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # M (int): The interference parameter,
            #           i.e., the number of harmonics to sum in the Fourier series decomposition (default 4)
            # print_to_console (bool): Print results directly to console (default False)
            # grid_jump (int): The grid jump size, must be identical to the value passed to
            #                   SALib.sample.morris.sample() (default 2)
            # num_levels (int): The number of grid levels, must be identical to the value passed to
            #                   SALib.sample.morris (default 4)
            self.analyzer = lambda output: fast.analyze(self.problem, output, **kwargs)

        elif np.in1d(self.method.lower(), ["ff", "fractional_factorial"]):
            warnings.warn("'fractional_factorial' method requires 'fractional_factorial' sampling scheme!")
            self.analyzer = lambda output: ff.analyze(self.problem, self.input_samples[:, input_ids], output, **kwargs)
            # Additional keyword parameters and their defaults:
            # second_order (bool, default=False): Include interaction effects
            # print_to_console (bool, default=False): Print results directly to console

        elif self.method.lower().lower() == "morris":
            warnings.warn("'morris' method requires 'morris' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # num_resamples (int): The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float): The confidence interval level (default 0.95)
            # print_to_console (bool): Print results directly to console (default False)
            self.analyzer = lambda output: morris.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                          **kwargs)

        elif self.method.lower() == "dgsm":
            # num_resamples (int): The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float): The confidence interval level (default 0.95)
            # print_to_console (bool): Print results directly to console (default False)
            self.analyzer = lambda output: dgsm.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                        **kwargs)
        else:
            raise ValueError(
                "Method " + str(self.method) + " is not one of the available methods " + str(METHODS) + " !")

        output_names = []
        results = []
        for io in output_ids:
            output_names.append(self.output_names[io])
            results.append(self.analyzer(self.output_values[:, io]))

         # TODO: Adjust list_of_dicts_to_dicts_of_ndarrays to handle ndarray concatenation
        results = list_of_dicts_to_dicts_of_ndarrays(results)

        results.update({"output_names": output_names})

        return results


# Test and example function:

if __name__ == "__main__":

    import os
    from tvb_epilepsy.base.constants import DATA_CUSTOM, FOLDER_RES
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader
    from tvb_epilepsy.custom.read_write import write_h5_model
    from tvb_epilepsy.base.h5_model import prepare_for_h5
    from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
    from tvb_epilepsy.base.model_configuration_service import ModelConfigurationService
    from tvb_epilepsy.base.lsa_service import LSAService
    from tvb_epilepsy.base.plot_tools import plot_lsa_pse

    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    head = reader.read_head(data_folder)

    # --------------------------Hypothesis definition-----------------------------------

    n_samples = 100

    #
    # Manual definition of hypothesis...:
    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]
    disease_indices = x0_indices + e_indices
    n_disease = len(disease_indices)

    n_x0 = len(x0_indices)
    n_e = len(e_indices)
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)
    # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
    hyp_x0_E = DiseaseHypothesis(head.connectivity, excitability_hypothesis={tuple(x0_indices): x0_values},
                                 epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                 connectivity_hypothesis={})

    print "Running hypothesis: " + hyp_x0_E.name

    print "creating model configuration..."
    model_configuration_service = ModelConfigurationService(hyp_x0_E.get_number_of_regions())
    model_configuration = model_configuration_service.configure_model_from_hypothesis(hyp_x0_E)

    print "running LSA..."
    lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
    lsa_hypothesis = lsa_service.run_lsa(hyp_x0_E, model_configuration)

    print "running sensitivity analysis PSE LSA..."
    for m in METHODS:
        # try:
        sa_results, pse_results = \
            sensitivity_analysis_pse_from_hypothesis(lsa_hypothesis, n_samples, method="sobol", half_range=0.1,
                                                     global_coupling=[{"indices": all_regions_indices,
                                                                       "bounds":
                                                                   [0.0, 2*model_configuration_service.K_unscaled[0]]}],
                                                    healthy_regions_parameters=
                                                     [{"name": "x0", "indices": healthy_indices}],
                                                     model_configuration=model_configuration,
                                                     model_configuration_service=model_configuration_service,
                                                     lsa_service=lsa_service)

        plot_lsa_pse(lsa_hypothesis, model_configuration, pse_results,
                     weighted_eigenvector_sum=lsa_service.weighted_eigenvector_sum,
                     n_eig=lsa_service.eigen_vectors_number,
                     figure_name=m + "_PSE_LSA_overview_" + lsa_hypothesis.name)
        # , show_flag=True, save_flag=False

        write_h5_model(prepare_for_h5(pse_results), FOLDER_RES, m + "_PSE_LSA_results_" +
                       lsa_hypothesis.name + ".h5")

        write_h5_model(prepare_for_h5(sa_results), FOLDER_RES, m + "_SA_LSA_results_" +
                       lsa_hypothesis.name + ".h5")
        # except:
        #     warnings.warn("Method " + m + " failed!")


