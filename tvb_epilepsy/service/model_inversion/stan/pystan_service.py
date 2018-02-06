import os
import pickle
import time
import numpy as np
import pystan as ps
from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.utils.log_error_utils import raise_not_implemented_error, raise_value_error
from tvb_epilepsy.service.model_inversion.stan.stan_service import StanService
from tvb_epilepsy.service.model_inversion.stan.stan_factory import STAN_OUTPUT_OPTIONS


class PyStanService(StanService):

    def __init__(self, model_name=None, model=None, model_dir=FOLDER_RES, model_code=None, model_code_path="",
                 model_data_path="", fitmethod="sampling", random_seed=12345, init="random", **options):
        super(PyStanService, self).__init__(model_name, model, model_dir, model_code, model_code_path, model_data_path,
                                            fitmethod)
        self.assert_fitmethod()
        self.options = {"init": init, "seed": random_seed, "verbose": True}
        self.options.update(options)
        # TODO: check if this is used for pickle
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def assert_fitmethod(self):
        if self.fitmethod.lower().find("sampl") >= 0:  # for sample or sampling
            self.fitmethod = "sampling"
        elif self.fitmethod.lower().find("v") >= 0:  # for variational or vb or advi
            self.fitmethod = "vb"
        elif self.fitmethod.lower().find("optimiz") >= 0:  # for optimization or optimizing or optimize
            self.fitmethod = "optimizing"
        else:
            raise_value_error(self.fitmethod + " does not correspond to one of the input methods:\n" +
                              "sampling, vb, optimizing")

    def compile_stan_model(self, save_model=True, **kwargs):
        self.model_code_path = kwargs.get("model_code_path", self.model_code_path)
        tic = time.time()
        self.logger.info("Compiling model...")
        self.model = ps.StanModel(file=self.model_code_path, model_name=self.model_name)
        self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if save_model:
            self.write_model_to_file(**kwargs)

    def write_model_to_file(self, **kwargs):
        self.model_path = kwargs.get("model_path", self.model_path)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def set_model_from_file(self, **kwargs):
        self.model_path = kwargs.get("model_path", self.model_path)
        self.model = pickle.load(open(self.model_path, 'rb'))

    def fit(self, output_filepath=os.path.join(FOLDER_RES, STAN_OUTPUT_OPTIONS["file"]), diagnostic_filepath="",
            debug=0, simulate=0, read_output=True, **kwargs):
        if diagnostic_filepath == "":
            diagnostic_filepath = os.path.join(os.path.dirname(output_filepath), STAN_OUTPUT_OPTIONS["diagnostic_file"])
        self.fitmethod = kwargs.pop("fitmethod", self.fitmethod)
        self.fitmethod = kwargs.pop("method", self.fitmethod)
        model_data = self.set_model_data(debug, simulate, **kwargs)
        self.assert_fitmethod()
        self.options.update(kwargs)
        self.logger.info("Model fitting with " + self.fitmethod + "...")
        tic = time.time()
        fit = getattr(self.model, self.fitmethod)(data=model_data, sample_file=output_filepath,
                                                  diagnostic_file=diagnostic_filepath, **self.options)
        self.fitting_time = time.time() - tic
        self.logger.info(str(self.fitting_time) + ' sec required to fit')
        if self.fitmethod is "optimizing":
            return fit,
        else:
            if read_output:
                self.logger.info("Extracting estimates...")
                if self.fitmethod is "sampling":
                    samples = fit.extract(permuted=True)
                elif self.fitmethod is "vb":
                    samples, _ = self.read_vb_results(fit)
                est = self.compute_estimates_from_samples(samples)
                return est, samples, fit
            else:
                return fit,

    def read_vb_results(self, fit):
        est = {}
        samples = {}
        for ip, p in enumerate(fit['sampler_param_names']):
            p_split = p.split('.')
            p_name = p_split.pop(0)
            if est.get(p_name) is None:
                samples.update({p_name: []})
                est.update({p_name: []})
            if len(p_split) == 0:
                # scalar parameters
                samples[p_name] = fit["sampler_params"][ip]
                est[p_name] = fit["mean_pars"][ip]
            else:
                if len(p_split) == 1:
                    # vector parameters
                    samples[p_name].append(fit["sampler_params"][ip])
                    est[p_name].append(fit["mean_pars"][ip])
                else:
                    ii = int(p_split.pop(0)) - 1
                    if len(p_split) == 0:
                        # 2D matrix parameters
                        if len(est[p_name]) < ii + 1:
                            samples[p_name].append([fit["sampler_params"][ip]])
                            est[p_name].append([fit["mean_pars"][ip]])
                        else:
                            samples[p_name][ii].append(fit["sampler_params"][ip])
                            est[p_name][ii].append(fit["mean_pars"][ip])
                    else:
                        if len(est[p_name]) < ii + 1:
                            samples[p_name].append([])
                            est[p_name].append([])
                        jj = int(p_split.pop(0)) - 1
                        if len(p_split) == 0:
                            # 3D matrix parameters
                            if len(est[p_name][ii]) < jj + 1:
                                samples[p_name][ii].append([fit["sampler_params"][ip]])
                                est[p_name][ii].append([fit["mean_pars"][ip]])
                            else:
                                if len(est[p_name][ii]) < jj + 1:
                                    samples[p_name][ii].append([])
                                    est[p_name][ii].append([])
                                    samples[p_name][ii][jj].append(fit["sampler_params"][ip])
                                est[p_name][ii][jj].append(fit["mean_pars"][ip])
                        else:
                            raise_not_implemented_error("Extracting of parameters of more than 3 dimensions is not " +
                                                        "implemented yet for vb!", self.logger)
        for key in est.keys():
            if isinstance(est[key], list):
                est[key] = np.squeeze(np.array(est[key]))
            if isinstance(samples[key], list):
                samples[key] = np.squeeze(np.array(samples[key]))
        return samples, est
