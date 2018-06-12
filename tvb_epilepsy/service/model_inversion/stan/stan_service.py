import os
import pickle
from shutil import copyfile
from abc import ABCMeta, abstractmethod
from scipy.io import savemat, loadmat
from scipy.stats import describe
import numpy as np
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, \
                                                    raise_value_error, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, sort_dict,  \
                                                    list_of_dicts_to_dicts_of_ndarrays, switch_levels_of_dicts_of_dicts
from tvb_epilepsy.io.r_file_io import rdump, rload
from tvb_epilepsy.io.csv import parse_csv
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer


class StanService(object):
    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    def __init__(self, model_name="", model=None, model_code=None, model_code_path="",
                 model_data_path="", fitmethod="sampling", config=None):
        self.fitmethod = fitmethod
        self.model_name = model_name
        self.model = model
        self.config = config or Config()
        model_dir = config.out.FOLDER_RES
        if not (os.path.isdir(model_dir)):
            os.mkdir(model_dir)
        self.model_path = os.path.join(model_dir, self.model_name)
        self.model_code = model_code
        if os.path.isfile(model_code_path):
            self.model_code_path = model_code_path
        else:
            self.model_code_path = self.model_path + ".stan"
        if model_data_path == "":
            self.model_data_path = os.path.join(model_dir, "ModelData.h5")
        self.compilation_time = 0.0

    @abstractmethod
    def compile_stan_model(self, save_model=True, **kwargs):
        pass

    @abstractmethod
    def set_model_from_file(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, model_data, **kwargs):
        pass

    def write_model_data_to_file(self, model_data, reset_path=False, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        if reset_path:
            self.model_data_path = model_data_path
        extension = model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "npy"):
            np.save(model_data_path, model_data)
        elif isequal_string(extension, "mat"):
            savemat(model_data_path, model_data)
        elif isequal_string(extension, "pkl"):
            with open(model_data_path, 'wb') as f:
                pickle.dump(model_data, f)
        elif isequal_string(extension, "R"):
            rdump(model_data_path, model_data)
        else:
            H5Writer().write_dictionary(model_data, os.path.join(os.path.dirname(model_data_path),
                                                                 os.path.basename(model_data_path)))

    def load_model_data_from_file(self, reset_path=False, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        if reset_path:
            self.model_data_path = model_data_path
        extension = model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "R"):
            model_data = rload(model_data_path)
        elif isequal_string(extension, "npy"):
            model_data = np.load(model_data_path).item()
        elif isequal_string(extension, "mat"):
            model_data = loadmat(model_data_path)
        elif isequal_string(extension, "pkl"):
            with open(model_data_path, 'wb') as f:
                model_data = pickle.load(f)
        elif isequal_string(extension, "h5"):
            model_data = H5Reader().read_dictionary(model_data_path)
        else:
            raise_not_implemented_error("model_data file (" + model_data_path +
                                        ") that are not one of (.R, .npy, .mat, .pkl) cannot be read!")
        for key in model_data.keys():
            if key[:3] == "EPI":
                del model_data[key]
        return model_data

    def set_model_data(self, debug=0, simulate=0, **kwargs):
        self.model_data_path = kwargs.get("model_data_path", self.model_data_path)
        model_data = kwargs.pop("model_data", None)
        if not(isinstance(model_data, dict)):
            model_data = self.load_model_data_from_file(self.model_data_path)
        # -1 for no debugging at all
        # 0 for printing only scalar parameters
        # 1 for printing scalar and vector parameters
        # 2 for printing all (scalar, vector and matrix) parameters
        model_data["DEBUG"] = debug
        # > 0 for simulating without using the input observation data:
        model_data["SIMULATE"] = simulate
        model_data = sort_dict(model_data)
        return model_data

    def set_or_compile_model(self, **kwargs):
        try:
            self.set_model_from_file(**kwargs)
        except:
            self.logger.info("Trying to compile model from file: " + str(self.model_code_path) + str("!"))
            self.compile_stan_model(save_model=kwargs.get("save_model", True), **kwargs)
        copyfile(self.model_code_path, os.path.join(os.path.dirname(self.model_path),
                                                    os.path.basename(self.model_code_path)))

    def read_output_samples(self, output_filepath, **kwargs):
        samples = ensure_list(parse_csv(output_filepath.replace(".csv", "*"), merge=kwargs.pop("merge_outputs", False)))
        if len(samples) == 1:
            return samples[0]
        return samples

    def compute_estimates_from_samples(self, samples):
        ests = []
        for chain_or_run_samples in ensure_list(samples):
            est = {}
            for pkey, pval in chain_or_run_samples.items():
                try:
                    est[pkey + "_low"], est[pkey], est[pkey + "_std"] = describe(chain_or_run_samples[pkey])[1:4]
                    est[pkey + "_high"] = est[pkey + "_low"][1]
                    est[pkey + "_low"] = est[pkey + "_low"][0]
                    est[pkey + "_std"] = np.sqrt(est[pkey + "_std"])
                    for skey in [pkey, pkey + "_low", pkey + "_high", pkey + "_std"]:
                        est[skey] = np.squeeze(est[skey])
                except:
                    est[pkey] = chain_or_run_samples[pkey]
            ests.append(sort_dict(est))
        if len(ests) == 1:
            return ests[0]
        else:
            return ests


    def compute_information_criteria(self, samples, nparams=None, nsamples=None, ndata=None, parameters=[],
                                     skip_samples=0, merge_chains_or_runs_flag=False, log_like_str='log_likelihood'):

        """

        :param samples: a dictionary of stan outputs or a list of dictionaries for multiple runs/chains
        :param nparams: number of model parameters, it can be inferred from parameters if None
        :param nsamples: number of samples, it can be inferred from loglikelihood if None
        :param ndata: number of data points, it can be inferred from loglikelihood if None
        :param parameters: a list of parameter names, necessary for dic metric computations and in case nparams is None,
                           as well as for aicc, aic and bic computation
        :param merge_chains_or_runs_flag: logical flag for merging seperate chains/runs, default is True
        :param log_like_str: the name of the log likelihood output of stan, default ''log_likelihood
        :return:
        """

        import sys
        sys.path.insert(0, self.config.generic.MODEL_COMPARISON_PATH)
        from information_criteria.ComputeIC import maxlike, aicc, aic, bic, dic, waic
        from information_criteria.ComputePSIS import psisloo


        # if self.fitmethod.find("opt") >= 0:
        #     warning("No model comparison can be computed for optimization method!")
        #     return None

        samples = ensure_list(samples)
        if merge_chains_or_runs_flag and len(samples) > 1:
            samples = ensure_list(merge_samples(samples, skip_samples, flatten=True))
            skip_samples = 0

        results = []
        for sample in samples:
            log_likelihood = -1 * sample[log_like_str][skip_samples:]
            log_lik_shape = log_likelihood.shape
            if len(log_lik_shape) > 1:
                target_shape = log_lik_shape[1:]
            else:
                target_shape = (1,)
            if nsamples is None:
                nsamples = log_lik_shape[0]
            elif nsamples != log_likelihood.shape[0]:
                warning("nsamples (" + str(nsamples) +
                        ") is not equal to likelihood.shape[0] (" + str(log_lik_shape[0]) + ")!")

            log_likelihood = np.reshape(log_likelihood, (log_lik_shape[0], -1))
            if log_likelihood.shape > 1:
                ndata_real = np.maximum(log_likelihood.shape[1], 1)
            else:
                ndata_real = 1
            if ndata is None:
                ndata = ndata_real
            elif ndata != ndata_real:
                warning("ndata (" + str(ndata) + ") is not equal to likelihood.shape[1] (" + str(ndata_real) + ")!")

            result = maxlike(log_likelihood)

            if len(parameters) == 0:
                parameters = [param for param in sample.keys() if param.find("_star") >= 0]
            if len(parameters) > 0:
                nparams_real = 0
                zscore_params = []
                for p in parameters:
                    pval = sample[p][skip_samples:]
                    pzscore = np.array((pval - np.mean(pval, axis=0)) / np.std(pval, axis=0))
                    if len(pzscore.shape) > 2:
                        pzscore = np.reshape(pzscore, (pzscore.shape[0], -1))
                    zscore_params.append(pzscore)
                    if len(pzscore.shape) > 1:
                        nparams_real += np.maximum(pzscore.shape[1], 1)
                    else:
                        nparams_real += 1
                if nparams is None:
                    nparams = nparams_real
                elif nparams != nparams_real:
                    warning("nparams (" + str(nparams) +
                            ") is not equal to number of parameters included in the dic computation (" +
                            str(nparams_real) + ")!")
                # TODO: find out how to reduce dic to 1 value, from 1 value per parameter. mean(.) for the moment:
                result['dic'] = np.mean(dic(log_likelihood, zscore_params))
            else:
                warning("Parameters' names' list is empty and we found no _star parameters! No computation of dic!")

            if nparams is not None:
                result['aicc'] = aicc(log_likelihood, nparams, ndata)
                result['aic'] = aic(log_likelihood, nparams)
                result['bic'] = bic(log_likelihood, nparams, ndata)
            else:
                warning("Unknown number of parameters! No computation of aic, aaic, bic!")

            result.update(waic(log_likelihood))

            if nsamples > 1:
                result.update(psisloo(log_likelihood))
                result["loos"] = np.reshape(result["loos"], target_shape)
                result["ks"] = np.reshape(result["ks"], target_shape)
            else:
                result.pop('p_waic', None)

            for metric, value in result.items():
                result[metric] = value * np.ones(1,)

            results.append(result)

        if len(results) == 1:
            return results[0]
        else:
            return list_of_dicts_to_dicts_of_ndarrays(results)

    def compare_models(self, samples, nparams=None, nsamples=None, ndata=None, parameters=[],
                       skip_samples=0, merge_chains_or_runs_flag=False, log_like_str='log_likelihood'):

        """

        :param samples: a dictionary of model's names and samples
        :param nparams: a number or list of numbers of parameters,
                       it can be inferred from parameters list or from _star parameters
        :param nsamples: a number or lists of numbers of samples, it can be inferred from loglikelihood if None
        :param ndata: a number or lists of numbers of data point, it can be inferred from loglikelihood if None
        :param parameters: a list (or list of lists) of parameter names,
                          it can be inferred from parameters list or from _star parameters
        :param merge_chains_or_runs_flag: logical flag for merging seperate chains/runs, default is True
        :param log_like_str: the name of the log likelihood output of stan, default ''log_likelihood
        :return:
        """

        def check_number_of_inputs(nmodels, input, input_str):
            input = ensure_list(input)
            ninput = len(input)
            if ninput != nmodels:
                if ninput == 1:
                    input *= nmodels
                else:
                    raise_value_error("The size of input " + input_str + " (" + str(ninput) +
                                      ") is neither equal to the number of models (" + str(nmodels) +
                                      ") nor equal to 1!")
            return input

        nmodels = len(samples)

        n_parameters = parameters
        if n_parameters > 0:
            if isinstance(parameters[0], (list, tuple)):
                parameters = check_number_of_inputs(nmodels, parameters, "number of parameters")
            else:
                parameters = nmodels*[parameters]
        nparams = check_number_of_inputs(nmodels, nparams, "number of parameters")
        nsamples = check_number_of_inputs(nmodels, nsamples, "number of samples")
        ndata = check_number_of_inputs(nmodels, ndata, "number of data points")
        skip_samples = check_number_of_inputs(nmodels, skip_samples, "skip_samples")
        log_like_str = check_number_of_inputs(nmodels, log_like_str, "log_like_str")

        results = {}

        for i_model, (model_name, model_samples) in enumerate(samples.items()):
            results[model_name] = \
               self.compute_information_criteria(model_samples, nparams[i_model], nsamples[i_model], ndata[i_model],
                                                 parameters[i_model], skip_samples[i_model], merge_chains_or_runs_flag,
                                                 log_like_str[i_model])

        # Return result into a dictionary with metrics at the upper level and models at the lower one
        return switch_levels_of_dicts_of_dicts(results)


def merge_samples(samples, skip_samples=0, flatten=False):
    samples = ensure_list(samples)
    if len(samples) > 1:
        samples = list_of_dicts_to_dicts_of_ndarrays(samples)
        for skey in samples.keys():
            if len(samples[skey].shape) == 1:
                samples[skey] = (samples[skey]*np.ones((1, 1))).T
            if flatten:
                sshape = samples[skey].shape
                if len(sshape) > 2:
                    samples[skey] = np.reshape(samples[skey][:, skip_samples:], tuple((-1,) + sshape[2:]))
                else:
                    samples[skey] = samples[skey][:, skip_samples:].flatten()

        return samples
    else:
        return samples[0]

