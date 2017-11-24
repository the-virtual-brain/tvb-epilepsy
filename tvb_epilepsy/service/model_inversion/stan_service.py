import os

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
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
        self.model_path = os.path.join(model_dir, self.model_name)
        self.model_code = model_code
        self.model_code_path = model_code_path
        self.compilation_time = 0.0
        self.fitting_time = 0.0
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

