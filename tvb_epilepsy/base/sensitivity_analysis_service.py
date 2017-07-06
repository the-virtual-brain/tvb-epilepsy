import numpy as np

class SensitivityAnalysisService(object):

    def __init__(self, method="latin", inputs={}, outputs={}):

        if np.in1d(method, ["sobol", "latin", "delta", "dgsm", "fast", "fast_sampler", "morris", "ff",
                            "fractional_factorial"]):
            self.method = method
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


    def run(self, method=None, input_ids=None, output_ids=None):

        if np.in1d(method, ["sobol", "latin", "delta", "dgsm", "fast", "fast_sampler", "morris", "ff",
                            "fractional_factorial"]):
            self.method = method

        if input_ids==None:
            input_ids = range(self.n_inputs)

        if output_ids==None:
            output_ids = range(self.n_outputs)