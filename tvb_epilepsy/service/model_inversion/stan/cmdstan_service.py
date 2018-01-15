import time
from shutil import copyfile
from tvb_epilepsy.base.constants.configurations import FOLDER_RES, CMDSTAN_PATH
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.base.utils.command_line_utils import execute_command
from tvb_epilepsy.service.model_inversion.stan.stan_service import StanService
from tvb_epilepsy.service.model_inversion.stan.stan_factory import *

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

    def assert_model_data_path(self, debug=0, simulate=0, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        model_data = self.load_model_data_from_file(self.model_data_path)
        # -1 for no debugging at all
        # 0 for printing only scalar parameters
        # 1 for printing scalar and vector parameters
        # 2 for printing all (scalar, vector and matrix) parameters
        model_data["DEBUG"] = debug
        # > 0 for simulating without using the input observation data:
        model_data["SIMULATE"] = simulate
        model_data = sort_dict(model_data)
        model_data_path = model_data_path.split(".", -1)[0] + ".R"
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
        tic = time.time()
        command = "make " + self.model_code_path.split(".stan", 1)[0] + " && " + \
                  "chmod +x " + self.model_code_path.split(".stan", 1)[0]
        self.compilation_time = execute_command(command, cwd=self.path, shell=True)[1]
        # proc = subprocess.Popen(command, cwd=self.path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # # use this to track the ongoing process:
        # # tail -n 1 vep-fe-rev-05.sample.*.out
        # stdout = proc.stdout.read().decode('ascii').strip()
        # if stdout:
        #     print(stdout)
        # stderr = proc.stderr.read().decode('ascii').strip()
        # if stderr:
        #     print(stderr)
        # self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if save_model:
            self.model_path = kwargs.pop("model_path", self.model_path)
            if self.model_path != self.model_code_path.split(".stan", 1)[0]:
                copyfile(self.model_code_path.split(".stan", 1)[0], self.model_path)

    def fit(self, output_filepath=os.path.join(FOLDER_RES, STAN_OUTPUT_OPTIONS["file"]), diagnostic_filepath="",
            debug=0, simulate=0, read_output=True, **kwargs):
        self.model_path = kwargs.pop("model_path", self.model_path)
        self.fitmethod = kwargs.pop("fitmethod", self.fitmethod)
        self.fitmethod = kwargs.pop("method", self.fitmethod)
        self.set_options(**kwargs)
        self.command, output_filepath, diagnostic_filepath = \
            generate_cmdstan_fit_command(self.fitmethod, self.options, self.model_path,
                                         self.assert_model_data_path(debug, simulate, **kwargs),
                                         output_filepath, diagnostic_filepath)
        self.logger.info("Model fitting with " + self.fitmethod +
                         " method of model: " + self.model_path + "...")
        self.fitting_time = execute_command(self.command.replace("\t", ""), shell=True)[1]
        # tic = time.time()
        # print(self.command.replace("\t", ""))
        # proc = subprocess.Popen(self.command.replace("\t", ""), shell=True,
        #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # # tail -n 1 vep-fe-rev-05.sample.*.out
        # stdout = proc.stdout.read().decode('ascii').strip()
        # if stdout:
        #     print(stdout)
        # stderr = proc.stderr.read().decode('ascii').strip()
        # if stderr:
        #     print(stderr)
        # self.fitting_time = time.time() - tic
        self.logger.info(str(self.fitting_time) + ' sec required to ' + self.fitmethod + "!")
        if read_output:
            est, csv = self.read_output_csv(output_filepath, **kwargs)
            # self.plot_HMCstats(output_filepath, self.model_name)
            return est, None
        else:
            return None, None
