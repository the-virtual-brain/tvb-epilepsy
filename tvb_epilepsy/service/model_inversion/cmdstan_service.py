import os
import time
import subprocess
from shutil import copyfile

import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME, CMDSTAN_PATH
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, raise_not_implemented_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.service.model_inversion.stan_service import StanService


LOG = initialize_logger(__name__)


class CmdStanService(StanService):

    def __init__(self, model_name=None, model=None, model_dir=os.path.join(FOLDER_VEP_HOME, "stan_models"),
                 model_code=None, model_code_path="", fitmethod="sampling", cmdstanpath=CMDSTAN_PATH, logger=LOG,
                 **options):
        super(CmdStanService, self).__init__(model_name, model, model_dir, model_code, model_code_path, fitmethod, logger)
        if not os.path.exists(os.path.join(cmdstanpath, 'runCmdStanTests.py')):
            raise_value_error('Please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")!')
        self.path = cmdstanpath
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def set_model_from_file(self, **kwargs):
        self.model_path = kwargs.get("model_path", self.model_path)
        if not(os.path.exists(self.model_path)):
            raise

    def compile_stan_model(self, store_model=True, **kwargs):
        self.model_code_path = kwargs.get("model_code_path", self.model_code_path)
        self.logger.info("Compiling model...")
        tic = time.time()
        proc = subprocess.Popen(['make', self.model_code_path], cwd=self.path,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = proc.stdout.read().decode('ascii').strip()
        if stdout:
            print(stdout)
        stderr = proc.stderr.read().decode('ascii').strip()
        if stderr:
            print(stderr)
        self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if store_model:
            self.model_path = kwargs.get("model_path", self.model_path)
            if self.model_path != self.model_code_path[:-5]:
                copyfile(self.model_code_path[:-5], self.model_path)

    def fit(self, model_data, **kwargs):
        self.model_path = kwargs.get("model_path", self.model_path)
        self.fitmode = kwargs.get("fitmethod", self.fitmethod)
        self.chains = kwargs.get("chains", self.chains)

        self.logger.info("Model fitting with " + self.fitmethod +
                         "\nof model: " + self.model_path + "...")
        tic = time.time()
        fit = getattr(self.model, self.fitmethod)(data=model_data, **kwargs)
        self.fitting_time = time.time() - tic
        self.logger.info(str(self.fitting_time) + ' sec required to fit')
        if self.fitmethod is "optimize":
            return fit,
        else:
            self.logger.info("Extracting estimates...")
            if self.fitmethod is "sample":
                est = fit.extract(permuted=True)
            elif self.fitmethod is "variational":
                est = self.read_vb_results(fit)
            return est, fit

