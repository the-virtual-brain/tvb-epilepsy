import os
import pickle
from abc import ABCMeta, abstractmethod
from scipy.io import savemat, loadmat
from scipy.stats import describe
import numpy as np
from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, sort_dict
from tvb_epilepsy.io.rdump import rdump, rload
from tvb_epilepsy.io.csv import parse_csv
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer


class StanService(object):
    __metaclass__ = ABCMeta

    def __init__(self, model_name="", model=None, model_dir=FOLDER_RES, model_code=None, model_code_path="",
                 model_data_path="", fitmethod="sampling", logger=None):
        if logger is None:
            self.logger = initialize_logger(__name__)
        else:
            self.logger = logger
        self.fitmethod = fitmethod
        self.model_name = model_name
        self.model = model
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
        extension = self.model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "R"):
            model_data = rload(self.model_data_path)
        elif isequal_string(extension, "npy"):
            model_data = np.load(self.model_data_path).item()
        elif isequal_string(extension, "mat"):
            model_data = loadmat(self.model_data_path)
        elif isequal_string(extension, "pkl"):
            with open(self.model_data_path, 'wb') as f:
                model_data = pickle.load(f)
        elif isequal_string(extension, "h5"):
            model_data = H5Reader().read_dictionary(self.model_data_path)
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

    def read_output_samples(self, output_filepath, **kwargs):
        samples = ensure_list(parse_csv(output_filepath.replace(".csv", "*"), merge=kwargs.pop("merge_outputs", False)))
        if len(samples) == 1:
            return samples[0]
        return samples

    def compute_estimates_from_samples(self, samples):
        ests = []
        for chain_samples in ensure_list(samples):
            est = {}
            for pkey, pval in chain_samples.iteritems():
                try:
                    est[pkey + "_low"], est[pkey], est[pkey + "_std"] = describe(chain_samples[pkey])[1:4]
                    est[pkey + "_high"] = est[pkey + "_low"][1]
                    est[pkey + "_low"] = est[pkey + "_low"][0]
                    est[pkey + "_std"] = np.sqrt(est[pkey + "_std"])
                    for skey in [pkey, pkey + "_low", pkey + "_high", pkey + "_std"]:
                        est[skey] = np.squeeze(est[skey])
                except:
                    est[pkey] = chain_samples[pkey]
            ests.append(sort_dict(est))
        if len(ests) == 1:
            return ests[0]
        else:
            return ests

