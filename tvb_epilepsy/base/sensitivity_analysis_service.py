import warnings
import numpy as np

from SALib.analyze import sobol, delta, fast, morris, dgsm,  ff

from tvb_epilepsy.base.utils import list_of_dicts_to_dicts_of_ndarrays


METHODS = ["sobol", "latin", "delta", "dgsm", "fast", "fast_sampler", "morris", "ff", "fractional_factorial"]

# TODO: incorporate a sampling function

# TODO: make sensitivity_analysis_from_hypothesis() helper function

# TODO: make example-testing function __main__

# TODO: make __repr__ and prepare h5_model functions

class SensitivityAnalysisService(object):

    def __init__(self, outputs, inputs={}, method="delta", calc_second_order=True, conf_level=0.95):

        if np.in1d(method.lower(), METHODS):
            self.method = method.lower()
        else:
            raise ValueError(
                "Method " + str(method.lower()) + " is not one of the available methods " + str(METHODS) + " !")

        if isinstance(calc_second_order, bool):
            self.calc_second_order = calc_second_order
        else:
            raise ValueError("calc_second_order = " + str(calc_second_order) + "is not a boolean as it should!")

        if isinstance(conf_level, float) and conf_level > 0.0 and conf_level < 1.0:
            self.conf_level = conf_level
        else:
            raise ValueError("conf_level = " + str(conf_level) +
                             "is not a float in the (0.0, 1.0) interval as it should!")

        self.n_samples = []
        self.input_names = []
        self.input_bounds = []
        self.input_samples = []
        self.n_inputs = 0

        for key, val in inputs.iteritems():

            self.n_inputs += 1

            self.input_names.append(key)

            samples = np.array(val["samples"]).flatten()
            self.n_samples.append(samples.size())
            self.input_samples.append(samples)

            self.input_bounds.append(val.get("bounds", [samples.min(), samples.max()]))

        self.n_samples = np.array(self.n_samples)
        if np.all(self.n_samples == self.n_samples[0]):
            self.n_samples = self.n_samples[0]
        else:
            raise ValueError("Not all input parameters have equal number of samples!: " + str(self.n_samples))

        self.input_samples = np.array(self.input_samples).T

        self.n_outputs = 0

        if len(outputs) == 1:

            self.output_names = outputs.keys()
            samples = np.array(outputs["samples"])
            if samples.size == self.n_samples:
                self.n_outputs = 1
                self.output_samples = samples.flatten()
            else:
                if samples.shape[0] == self.n_samples:
                    self.output_samples = samples.T
                elif samples.shape[1] == self.n_samples:
                    self.output_samples = samples
                else:
                    raise ValueError("Non of the dimensions of output samples: " +str(self.output_samples.shape) +
                                     " matches n_samples = " + str(self.n_samples) + " !")
                self.n_outputs = self.output_samples.shape[0]

            if self.n_outputs > 1 and len(self.output_names) == 1:
                self.output_names = np.array(["%d. %s" % l for l in zip(range(self.n_outputs),
                                                                        np.repeat(self.output_names[0], 3))])

        else:

            for key, val in outputs.iteritems():

                self.n_outputs += 1

                self.output_names.append(key)

                samples = np.array(val["samples"]).flatten()
                if samples.size() != self.n_samples:
                    ValueError("Output " + key + " has " +str(samples.size()) + "samples instead of n_samples = "
                               + str(self.n_samples) + " !")

                self.output_samples.append(samples)

            self.output_samples = np.array(self.output_samples).T

        self.problem = {}
        self.other_parameters = {}

    def update_parameters(self, method=None, calc_second_order=None, conf_level=None):

        if method is not None:
            if np.in1d(method.lower(), METHODS):
                self.method = method.lower()
            else:
                raise ValueError(
                    "Method " + str(method.lower()) + " is not one of the available methods " + str(METHODS) + " !")

        if calc_second_order is not None:
            if isinstance(calc_second_order, bool):
                self.calc_second_order = calc_second_order
            else:
                raise ValueError("calc_second_order = " + str(calc_second_order) + "is not a boolean as it should!")

        if conf_level is not None:
            if isinstance(conf_level, float) and conf_level > 0.0 and conf_level < 1.0:
                self.conf_level = conf_level
            else:
                raise ValueError("conf_level = " + str(conf_level) +
                                 "is not a float in the (0.0, 1.0) interval as it should!")

    def run(self, input_ids=None, output_ids=None, method=None, calc_second_order=None, conf_level=None, **kwargs):

        self.update_parameters(method, calc_second_order, conf_level)

        self.other_parameters = kwargs

        if input_ids is None:
            input_ids = range(self.n_inputs)

        self.problem = {"num_vars": len(input_ids),
                        "names": np.array(self.input_names[input_ids]).tolist(),
                        "bounds": np.array(self.input_bounds[input_ids]).tolist()}

        if output_ids is None:
            output_ids = range(self.n_outputs)

        n_outputs = len(output_ids)

        if self.method.lower() is "sobol":
            warnings.warn("'sobol' method requires 'saltelli' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # calc_second_order (bool) – Calculate second-order sensitivities (default True)
            # num_resamples (int) – The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float) – The confidence interval level (default 0.95)
            # print_to_console (bool) – Print results directly to console (default False)
            self.analyzer = lambda output: sobol.analyze(self.problem, output, **kwargs)

        elif np.in1d(self.method.lower(), ["latin", "delta"]):
            warnings.warn("'latin' sampling scheme is recommended for 'delta' method!")
            # Additional keyword parameters and their defaults:
            # num_resamples (int) – The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float) – The confidence interval level (default 0.95)
            # print_to_console (bool) – Print results directly to console (default False)
            self.analyzer = lambda output: delta.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                         **kwargs)

        elif np.in1d(self.method.lower(), ["fast", "fast_sampler"]):
            warnings.warn("'fast' method requires 'fast_sampler' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # M (int) – The interference parameter,
            #           i.e., the number of harmonics to sum in the Fourier series decomposition (default 4)
            # print_to_console (bool) – Print results directly to console (default False)
            # grid_jump (int) – The grid jump size, must be identical to the value passed to
            #                   SALib.sample.morris.sample() (default 2)
            # num_levels (int) – The number of grid levels, must be identical to the value passed to
            #                   SALib.sample.morris (default 4)
            self.analyzer = lambda output: fast.analyze(self.problem, output, **kwargs)

        elif np.in1d(self.method.lower(), ["ff", "fractional_factorial"]):
            warnings.warn("'fractional_factorial' method requires 'fractional_factorial' sampling scheme!")
            self.analyzer = lambda output: ff.analyze(self.problem, self.input_samples[:, input_ids], output, **kwargs)
            # Additional keyword parameters and their defaults:
            # second_order (bool, default=False) – Include interaction effects
            # print_to_console (bool, default=False) – Print results directly to console

        elif self.method.lower().lower() is "morris":
            warnings.warn("'morris' method requires 'morris' sampling scheme!")
            # Additional keyword parameters and their defaults:
            # num_resamples (int) – The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float) – The confidence interval level (default 0.95)
            # print_to_console (bool) – Print results directly to console (default False)
            self.analyzer = lambda output: morris.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                          **kwargs)

        elif self.method.lower() is "dgsm":
            # num_resamples (int) – The number of resamples used to compute the confidence intervals (default 1000)
            # conf_level (float) – The confidence interval level (default 0.95)
            # print_to_console (bool) – Print results directly to console (default False)
            self.analyzer = lambda output: dgsm.analyze(self.problem, self.input_samples[:, input_ids], output,
                                                        **kwargs)
        else:
            raise ValueError(
                "Method " + str(self.method) + " is not one of the available methods " + str(METHODS) + " !")

        output_names = []
        results = []
        for io in output_ids:
            output_names.append(self.output_names[io])
            results.append(self.analyzer(self.output_samples[:, io]))

         # TODO: Adjust list_of_dicts_to_dicts_of_ndarrays to handle ndarray concatenation
        results = list_of_dicts_to_dicts_of_ndarrays(results)

        results = results.update({"output_names": output_names})

        return results
