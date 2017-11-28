import os
import pickle
from abc import ABCMeta, abstractmethod

from scipy.io import savemat, loadmat
import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path, isequal_string
from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
from tvb_epilepsy.service.rdump_factory import rdump


LOG = initialize_logger(__name__)


class StanService(object):
    __metaclass__ = ABCMeta

    def __init__(self, model_name="", model=None, model_dir=FOLDER_RES, model_code=None, model_code_path="",
                 model_data_path="", fitmethod="sampling", logger=LOG):
        self.logger = logger
        self.fitmethod = fitmethod
        self.model_name = model_name
        self.model = model
        if not(os.path.isdir(model_dir)):
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
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "ProbabilityDistributionModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.type + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    @abstractmethod
    def compile_stan_model(self, store_model=True, **kwargs):
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
            convert_to_h5_model(model_data).write_to_h5(os.path.dirname(model_data_path),
                                                        os.path.basename(model_data_path))

    def load_model_data_from_file(self, reset_path=False, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        if reset_path:
            self.model_data_path = model_data_path
        extension = self.model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "npy"):
            return np.load(self.model_data_path).item()
        elif isequal_string(extension, "mat"):
            return loadmat(self.model_data_path)
        elif isequal_string(extension, "pkl"):
            with open(self.model_data_path, 'wb') as f:
                return pickle.load(f)
        elif isequal_string(extension, "h5"):
            return read_h5_model(self.model_data_path).convert_from_h5_model()
        else:
            raise_not_implemented_error("model_data file (" + model_data_path +
                                        ") that are not one of (.npy, .mat, .pkl) cannot be read!")

    def set_or_compile_model(self, **kwargs):
        try:
            self.set_model_from_file(**kwargs)
        except:
            self.logger.info("Failed to load the model from file: " + str(self.model_path) + " !" +
                             "\nTrying to compile model from file: " + str(self.model_code_path) + str("!"))
            self.compile_stan_model(store_model=kwargs.get("store_model", True), **kwargs)
