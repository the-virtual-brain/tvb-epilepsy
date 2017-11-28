import os
import subprocess
import time
from shutil import copyfile
from copy import deepcopy

from tvb_epilepsy.base.constants.configurations import FOLDER_RES, CMDSTAN_PATH
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path, isequal_string
from tvb_epilepsy.service.model_inversion.stan.stan_service import StanService
from tvb_epilepsy.service.csv_factory import parse_csv
from tvb_epilepsy.service.model_inversion.stan.stan_options import *


LOG = initialize_logger(__name__)


class CmdStanService(StanService):

    def __init__(self, model_name=None, model=None, model_dir=FOLDER_RES,
                 model_code=None, model_code_path="", model_data_path="", cmdstanpath=CMDSTAN_PATH,
                 fitmethod="sample", random_seed=12345, init="random", logger=LOG, **options):
        super(CmdStanService, self).__init__(model_name, model, model_dir, model_code, model_code_path, model_data_path,
                                             fitmethod, logger)
        self.assert_fitmethod()
        if not os.path.isfile(os.path.join(cmdstanpath, 'runCmdStanTests.py')):
            raise_value_error('Please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")!')
        self.path = cmdstanpath
        self.command_path = ""
        self.options = {"init": init, "random_seed": random_seed}
        self.options = self.set_options(**options)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def assert_fitmethod(self):
        if self.fitmethod.lower().find("sampl") >= 0:  # for sample or sampling
            self.fitmethod = "sample"
        elif self.fitmethod.lower().find("v") >= 0:  # for variational or vb or advi
            self.fitmethod = "variational"
        elif self.fitmethod.lower().find("optimiz") >= 0:  # for optimization or optimizing or optimize
            self.fitmethod = "optimize"
        elif self.fitmethod.lower().find("diagnos") >= 0:  # for diagnose or diagnosing
            self.fitmethod = "diagnose"
        else:
            raise_value_error(self.fitmethod + " does not correspond to one of the input methods:\n" +
                              "sample, variational, optimize, diagnose")

    def assert_model_data_path(self, reset_path=False):
        model_data_path, extension = self.model_data_path.split(".", -1)
        if extension != "R":
            model_data_path = model_data_path + ".R"
            if not(os.path.isfile(model_data_path)):
                self.write_model_data_to_file(self.load_model_data_from_file(self.model_data_path), reset_path=False,
                                              model_data_path=model_data_path)
        else:
            model_data_path = self.model_data_path
        return model_data_path

    def set_options(self, **options):
        self.fitmethod = options.get("method", self.fitmethod)
        self.assert_fitmethod()
        self.options = generate_cmdstan_options(self.fitmethod, **options)

    def set_model_from_file(self, **kwargs):
        self.model_path = kwargs.pop("model_path", self.model_path)
        if not(os.path.exists(self.model_path)):
            raise

    def compile_stan_model(self, store_model=True, **kwargs):
        self.model_code_path = kwargs.pop("model_code_path", self.model_code_path)
        self.logger.info("Compiling model...")
        tic = time.time()
        proc = subprocess.Popen(['make', self.model_code_path.split(".stan", 1)[0]], cwd=self.path,
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
            self.model_path = kwargs.pop("model_path", self.model_path)
            if self.model_path != self.model_code_path.split(".stan", 1)[0]:
                copyfile(self.model_code_path.split(".stan", 1)[0], self.model_path)

    def generate_fit_command(self, output_filepath, diagnostic_filepath):
        command = self.model_path
        if isequal_string(self.fitmethod, "sample"):
            command += " method=sample"' \\ ' + "\n"
            command += "\t\talgorithm=" + self.options["algorithm"] + ' \\ ' + "\n"
            if isequal_string(self.options["algorithm"], "hmc"):
                command += "\t\t\t\tengine=" + self.options["engine"] + ' \\ ' + "\n"
                if isequal_string(self.options["engine"], "nuts"):
                    command += "\t\t\t\t\t\tmax_depth=" + str(self.options["max_depth"]) + ' \\ ' + "\n"
                elif isequal_string(self.options["engine"], "static"):
                    command += "\t\t\t\t\t\tint_time=" + str(self.options["int_time"]) + ' \\ ' + "\n"
                hmc_options = dict(STAN_HMC_OPTIONS)
                del hmc_options["engine"]
                for option in STAN_HMC_OPTIONS.keys():
                    command += "\t\t\t\t" + option + "=" + str(self.options[option]) + ' \\ ' + "\n"
        elif isequal_string(self.fitmethod, "variational"):
            command += " method=variational"' \\ ' + "\n"
            for option in STAN_VARIATIONAL_OPTIONS.keys():
                # due to sort_dict, we know that algorithm is the first option
                command += "\t\t\t\t" + option + "=" + str(self.options[option]) + ' \\ ' + "\n"
        elif isequal_string(self.fitmethod, "optimize"):
            command += " method=optimize"' \\ ' + "\n"
            command += "\t\talgorithm=" + self.options["algorithm"] + ' \\ ' + "\n"
            if (self.options["algorithm"].find("bfgs") >= 0):
                for option in STAN_BFGS_OPTIONS.keys():
                    command += "\t\t\t\t" + option + "=" + str(self.options[option]) + ' \\ ' + "\n"
        # + " data file=" + model_data_path
        elif isequal_string(self.fitmethod, "diagnose"):
            command += " method=diagnose"' \\ ' + "\n" + "\t\ttest=gradient "
            for option in STAN_DIAGNOSE_TEST_GRADIENT_OPTIONS.keys():
                command += "\t\t\t\t" + + option + "=" + str(self.options[option]) + ' \\ ' + "\n"
        if isequal_string(self.fitmethod, "sample") or isequal_string(self.fitmethod, "variational"):
            command += "\t\tadapt"' \\ ' + "\n"
            if isequal_string(self.fitmethod, "sample"):
                adapt_options = STAN_SAMPLE_ADAPT_OPTIONS
            else:
                adapt_options = STAN_VARIATIONAL_ADAPT_OPTIONS
            for option in adapt_options.keys():
                command += "\t\t\t\t" + option + "=" + str(self.options[option]) + ' \\ ' + "\n"
        command += "\t\tdata file="+ self.assert_model_data_path() + ' \\ ' + "\n"
        command += "\t\tinit=" + str(self.options["init"]) + ' \\ ' + "\n"
        command += "\t\trandom seed=" + str(self.options["random_seed"]) + ' \\ ' + "\n"
        if diagnostic_filepath == "":
            diagnostic_filepath = os.path.join(os.path.dirname(output_filepath), STAN_OUTPUT_OPTIONS["diagnostic_file"])
        if self.options["chains"] > 1:
            command = ("for i in {1.." + str(self.options["chains"]) + "} \ndo\n" +
                       "\t" + command +
                       "\t\tid=$i" + ' \\ ' + "\n" +
                       "\t\toutput file=" + output_filepath[:-4] + "$i.csv"' \\ ' + "\n" +
                       "\t\tdiagnostic_file=" + diagnostic_filepath[:-4] + "$i.csv"' \\ ' + "\n" +
                       "\t\trefresh=" + str(self.options["refresh"]) + " &" + "\n" +
                       "done")
        else:
            command += "\t\toutput file=" + output_filepath + ' \\ ' + "\n"
            command += "\t\tdiagnostic_file=" + diagnostic_filepath + ' \\ ' + "\n"
            command += "\t\trefresh=" + str(self.options["refresh"])
        command = "#!/bin/bash\n" + command
        command = ''.join(command)
        self.command_path = os.path.join(os.path.dirname(output_filepath), "command.sh")
        command_file = open(self.command_path, "w")
        command_file.write(command)
        command_file.close()
        return output_filepath, diagnostic_filepath

    def fit(self, output_filepath=os.path.join(FOLDER_RES, STAN_OUTPUT_OPTIONS["file"]), diagnostic_filepath="",
            read_output=True, **kwargs):
        self.model_path = kwargs.pop("model_path", self.model_path)
        self.fitmethod = kwargs.pop("fitmethod", self.fitmethod)
        self.fitmethod = kwargs.pop("method", self.fitmethod)
        self.set_options(**kwargs)
        output_filepath, diagnostic_filepath = \
                                                    self.generate_fit_command(output_filepath, diagnostic_filepath)
        self.logger.info("Model fitting with " + self.fitmethod +
                         "\nof model: " + self.model_path + "...")
        tic = time.time()
        proc = subprocess.Popen(['sh', self.command_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = proc.stdout.read().decode('ascii').strip()
        if stdout:
            print(stdout)
        stderr = proc.stderr.read().decode('ascii').strip()
        if stderr:
            print(stderr)
        self.fitting_time = time.time() - tic
        self.logger.info(str(self.fitting_time) + ' sec required to fit')
        if read_output:
            return parse_csv(output_filepath.replace(".csv", "*"), merge=True)
        else:
            return

