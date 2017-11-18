import os
import pickle
import time
from copy import deepcopy

import numpy as np
import pystan as ps

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error, warning

LOG = initialize_logger(__name__)


class StanService(object):

    def __init__(self, model_name=None, model=None, model_dir=os.path.join(FOLDER_VEP_HOME, "stan_models"),
                 model_code=None, model_code_path="", fitmode="sampling", logger=LOG):
        self.logger = logger
        self.fitmode = fitmode
        self.model_name = model_name
        self.model = model
        if not(os.path.isdir(model_dir)):
            os.mkdir(model_dir)
        self.model_path = os.path.join(model_dir, self.model_name + "_stanmodel.pkl")
        self.model_code = model_code
        self.model_code_path = model_code_path
        self.compilation_time = 0.0
        self.fitting_time = 0.0

    def compile_stan_model(self, write_model=True):
        tic = time.time()
        self.logger.info("Compiling model...")
        self.model = ps.StanModel(file=self.model_code_path, model_name=self.model_name)
        self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if write_model:
            self.write_model_to_file()

    def write_model_to_file(self):
        with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def load_model_from_file(self):
        self.model = pickle.load(open(self.model_path, 'rb'))

    def load_or_compile_model(self):
        if os.path.isfile(self.model_path):
            try:
                self.load_model_from_file()
            except:
                warning("Failed to load the model from file: " + str(self.model_path) + " !" +
                        "\nTrying to compile model from file: " + str(self.model_code_path) + str("!"))
                self.compile_stan_model()
        else:
            self.compile_stan_model()
