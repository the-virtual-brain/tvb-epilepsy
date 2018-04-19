from shutil import copyfile
from glob import glob
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.utils.command_line_utils import execute_command
from tvb_epilepsy.base.utils.file_utils import change_filename_or_overwrite_with_wildcard
from tvb_epilepsy.io.csv import parse_csv_in_cols
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.model_inversion.stan.stan_service import StanService
from tvb_epilepsy.service.model_inversion.stan.stan_factory import *


class CmdStanService(StanService):

    def __init__(self, model_name=None, model=None, model_code=None, model_code_path="", model_data_path="",
                 output_filepath=None, diagnostic_filepath=None, summary_filepath=None,
                 fitmethod="sample", random_seed=12345, init="random", config=None, **options):
        super(CmdStanService, self).__init__(model_name, model, model_code, model_code_path, model_data_path,
                                             fitmethod, config)
        if not os.path.isfile(os.path.join(self.config.generic.CMDSTAN_PATH, 'runCmdStanTests.py')):
            raise_value_error('Please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")!')
        self.path = self.config.generic.CMDSTAN_PATH
        self.output_filepath, self.diagnostic_filepath, self.summary_filepath = \
            self.set_output_files(output_filepath, diagnostic_filepath, summary_filepath, False)
        self.assert_fitmethod()
        self.command = ""
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

    def set_output_files(self, output_filepath=None, diagnostic_filepath=None, summary_filepath=None, check_files=False,
                         overwrite_output_files=False):
        if output_filepath is None:
            output_filepath = os.path.join(self.config.out.FOLDER_RES, STAN_OUTPUT_OPTIONS["file"])
        if diagnostic_filepath is None:
            diagnostic_filepath = os.path.join(self.config.out.FOLDER_RES, STAN_OUTPUT_OPTIONS["diagnostic_file"])
        if summary_filepath is None:
            summary_filepath = os.path.join(self.config.out.FOLDER_RES, "stan_summary.csv")
        if check_files:
            return change_filename_or_overwrite_with_wildcard(output_filepath.split(".csv")[0],
                                                              overwrite_output_files) + ".csv", \
                   change_filename_or_overwrite_with_wildcard(diagnostic_filepath.split(".csv")[0],
                                                              overwrite_output_files) + ".csv", \
                   change_filename_or_overwrite_with_wildcard(summary_filepath.split(".csv")[0],
                                                              overwrite_output_files) + ".csv"
        else:
            return output_filepath, diagnostic_filepath, summary_filepath

    def set_model_data(self, debug=0, simulate=0, **kwargs):
        model_data = super(CmdStanService, self).set_model_data(debug, simulate, **kwargs)
        model_data_path = self.model_data_path.split(".", -1)[0] + ".R"
        self.write_model_data_to_file(model_data, model_data_path=model_data_path)
        return model_data_path

    def set_options(self, **options):
        self.fitmethod = options.get("method", self.fitmethod)
        self.assert_fitmethod()
        self.options = generate_cmdstan_options(self.fitmethod, **options)

    def set_model_from_file(self, **kwargs):
        self.model_path = kwargs.pop("model_path", self.model_path)
        if not (os.path.exists(self.model_path)):
            raise_value_error("Failed to load the model from file: " + str(self.model_path) + " !")

    def compile_stan_model(self, save_model=True, **kwargs):
        self.model_code_path = kwargs.pop("model_code_path", self.model_code_path)
        self.logger.info("Compiling model...")
        mode_code_path = self.model_code_path.split(".stan", 1)[0]
        command = "make CC=" + self.config.generic.C_COMPILER + " " + mode_code_path + \
                  " && " + "chmod +x " + mode_code_path
        self.compilation_time = execute_command(command, cwd=self.path, shell=True)[1]
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if save_model:
            self.model_path = kwargs.pop("model_path", self.model_path)
            if self.model_path != self.model_code_path.split(".stan", 1)[0]:
                copyfile(self.model_code_path.split(".stan", 1)[0], self.model_path)

    def read_output(self):
        samples = self.read_output_samples(self.output_filepath)
        est = self.compute_estimates_from_samples(samples)
        if os.path.isfile(self.summary_filepath):
            summary = parse_csv_in_cols(self.summary_filepath)
        else:
            summary = None
        return est, samples, summary

    def stan_summary(self):
        command = "bin/stansummary " + self.output_filepath.split(".csv")[0] + "*.csv" + " --csv_file=" \
                  + self.summary_filepath
        execute_command(command, cwd=self.path, shell=True)

    def get_Rhat(self, summary):
        if isinstance(summary, dict):
            return summary.get("R_hat", None)
        #     if Rhat is not None:
        #         Rhat = {"R_hat": Rhat}
        # return Rhat
        else:
            return None

    def fit(self,debug=0, simulate=0, return_output=True, plot_HMC=True, overwrite_output_files=False, plot_warmup=1,
            **kwargs):
        num_warmup = kwargs.get("num_warmup", 0)
        # Confirm output files and check if overwriting is necessary
        self.output_filepath, self.diagnostic_filepath, self.summary_filepath = \
            self.set_output_files(kwargs.pop("output_filepath", self.output_filepath),
                                  kwargs.pop("diagnostic_filepath", self.diagnostic_filepath),
                                  kwargs.pop("summary_filepath", self.summary_filepath),
                                  True, overwrite_output_files)
        self.model_path = kwargs.pop("model_path", self.model_path)
        self.fitmethod = kwargs.pop("fitmethod", self.fitmethod)
        self.fitmethod = kwargs.pop("method", self.fitmethod)
        self.set_options(**kwargs)
        self.command, self.output_filepath, self.diagnostic_filepath = \
            generate_cmdstan_fit_command(self.fitmethod, self.options, self.model_path,
                                         self.set_model_data(debug, simulate, **kwargs),
                                         self.output_filepath, self.diagnostic_filepath)
        self.logger.info("Model fitting with " + self.fitmethod +
                         " method of model: " + self.model_path + "...")
        self.fitting_time = execute_command(self.command.replace("\t", ""), shell=True)[1]
        self.logger.info(str(self.fitting_time) + ' sec required to ' + self.fitmethod + "!")
        self.logger.info("Computing stan summary...")
        self.stan_summary()
        if return_output:
            est, samples, summary = self.read_output()
            if plot_HMC and self.fitmethod.find("sampl") >= 0 and \
                isequal_string(self.options.get("algorithm", "None"), "HMC"):
                Plotter(self.config).plot_HMC(samples,
                                              kwargs.pop("skip_samples", (1-kwargs.get("plot_warmup", 0)) * num_warmup))
            return est, samples, summary
        else:
            return None, None, None
