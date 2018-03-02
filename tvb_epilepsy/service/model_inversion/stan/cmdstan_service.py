from shutil import copyfile
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.utils.command_line_utils import execute_command
from tvb_epilepsy.io.csv import parse_csv_in_cols
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.model_inversion.stan.stan_service import StanService
from tvb_epilepsy.service.model_inversion.stan.stan_factory import *


class CmdStanService(StanService):

    def __init__(self, model_name=None, model=None, model_code=None, model_code_path="", model_data_path="",
                 fitmethod="sample", random_seed=12345, init="random", config=None, **options):
        super(CmdStanService, self).__init__(model_name, model, model_code, model_code_path, model_data_path,
                                             fitmethod, config)
        if not os.path.isfile(os.path.join(self.config.generic.CMDSTAN_PATH, 'runCmdStanTests.py')):
            raise_value_error('Please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")!')
        self.path = self.config.generic.CMDSTAN_PATH
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
        command = "make CC=" + self.config.generic.C_COMPILER + " " + self.model_code_path.split(".stan", 1)[0] + \
                  " && " + "chmod +x " + self.model_code_path.split(".stan", 1)[0]
        self.compilation_time = execute_command(command, cwd=self.path, shell=True)[1]
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if save_model:
            self.model_path = kwargs.pop("model_path", self.model_path)
            if self.model_path != self.model_code_path.split(".stan", 1)[0]:
                copyfile(self.model_code_path.split(".stan", 1)[0], self.model_path)

    def read_output(self, output_filepath=None, **kwargs):
        if output_filepath is None:
            output_filepath = os.path.join(self.config.out.FOLDER_RES, STAN_OUTPUT_OPTIONS["file"])
        samples = self.read_output_samples(output_filepath, **kwargs)

        est = self.compute_estimates_from_samples(samples)
        summary_filepath = kwargs.pop("summary_filepath",
                                      os.path.join(self.config.out.FOLDER_RES, "stan_summary.csv"))
        if os.path.isfile(summary_filepath):
            summary = parse_csv_in_cols(summary_filepath)
        else:
            summary = None
        return est, samples, summary

    def stan_summary(self, output_filepath=None, summary_filepath=None):
        if output_filepath is None:
            output_filepath = os.path.join(self.config.out.FOLDER_RES, STAN_OUTPUT_OPTIONS["file"])
        if summary_filepath is None:
            summary_filepath = os.path.join(self.config.out.FOLDER_RES, "stan_summary.csv")

        command = "bin/stansummary " + output_filepath[:-4] + "*.csv" + " --csv_file=" + summary_filepath
        execute_command(command, cwd=self.path, shell=True)
        return summary_filepath

    def fit(self, output_filepath=None, diagnostic_filepath="", summary_filepath=None, debug=0, simulate=0,
            return_output=True, plot_HMC=True, **kwargs):
        if output_filepath is None:
            output_filepath = os.path.join(self.config.out.FOLDER_RES, STAN_OUTPUT_OPTIONS["file"])
        if summary_filepath is None:
            summary_filepath = os.path.join(self.config.out.FOLDER_RES, "stan_summary.csv")

        self.model_path = kwargs.pop("model_path", self.model_path)
        self.fitmethod = kwargs.pop("fitmethod", self.fitmethod)
        self.fitmethod = kwargs.pop("method", self.fitmethod)
        self.set_options(**kwargs)
        self.command, output_filepath, diagnostic_filepath = \
            generate_cmdstan_fit_command(self.fitmethod, self.options, self.model_path,
                                         self.set_model_data(debug, simulate, **kwargs),
                                         output_filepath, diagnostic_filepath)
        self.logger.info("Model fitting with " + self.fitmethod +
                         " method of model: " + self.model_path + "...")
        self.fitting_time = execute_command(self.command.replace("\t", ""), shell=True)[1]
        self.logger.info(str(self.fitting_time) + ' sec required to ' + self.fitmethod + "!")
        self.logger.info("Computing stan summary...")
        summary_filepath = self.stan_summary(output_filepath, summary_filepath)
        if return_output:
            est, samples, summary = self.read_output(output_filepath, summary_filepath=summary_filepath, **kwargs)
            if plot_HMC and self.fitmethod.find("sampl") >= 0 and \
                isequal_string(self.options.get("algorithm", "None"), "HMC"):
                Plotter(self.config).plot_HMC(samples, kwargs.pop("skip_samples", 0))
            return est, samples, summary
        else:
            return None, None, None
