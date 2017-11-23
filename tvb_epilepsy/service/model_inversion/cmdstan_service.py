import os
# import pickle
import time

# import numpy as np
import pystan as ps

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.service.model_inversion.stan_service import StanService


LOG = initialize_logger(__name__)


class CmdstanService(StanService):

    def __init__(self, model_name=None, model=None, model_dir=os.path.join(FOLDER_VEP_HOME, "stan_models"),
                 model_code=None, model_code_path="", fitmode="sampling", logger=LOG):
        super(CmdstanService, self).__init__(model_name, model, model_dir, model_code, model_code_path, fitmode, logger)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"


    def compile_stan_model(self, write_model=True):
        tic = time.time()
        self.logger.info("Compiling model...")
        self.model = ps.StanModel(file=self.model_code_path, model_name=self.model_name)
        self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')
        if write_model:
            self.write_model_to_file()

            # def fit_stan_model(self, model_data=None, **kwargs):
    #     self.logger.info("Model fitting with " + self.fitmode + "...")
    #     tic = time.time()
    #     fit = getattr(self.model, self.fitmode)(data=model_data, **kwargs)
    #     self.fitting_time = time.time() - tic
    #     self.logger.info(str(self.fitting_time) + ' sec required to fit')
    #     if self.fitmode is "optimizing":
    #         return fit,
    #     else:
    #         self.logger.info("Extracting estimates...")
    #         if self.fitmode is "sampling":
    #             est = fit.extract(permuted=True)
    #         elif self.fitmode is "vb":
    #             est = self.read_vb_results(fit)
    #         return est, fit
    #
    # def read_vb_results(self, fit):
    #     est = {}
    #     for ip, p in enumerate(fit['sampler_param_names']):
    #         p_split = p.split('.')
    #         p_name = p_split.pop(0)
    #         p_name_samples = p_name + "_s"
    #         if est.get(p_name) is None:
    #             est.update({p_name_samples: []})
    #             est.update({p_name: []})
    #         if len(p_split) == 0:
    #             # scalar parameters
    #             est[p_name_samples] = fit["sampler_params"][ip]
    #             est[p_name] = fit["mean_pars"][ip]
    #         else:
    #             if len(p_split) == 1:
    #                 # vector parameters
    #                 est[p_name_samples].append(fit["sampler_params"][ip])
    #                 est[p_name].append(fit["mean_pars"][ip])
    #             else:
    #                 ii = int(p_split.pop(0)) - 1
    #                 if len(p_split) == 0:
    #                     # 2D matrix parameters
    #                     if len(est[p_name]) < ii + 1:
    #                         est[p_name_samples].append([fit["sampler_params"][ip]])
    #                         est[p_name].append([fit["mean_pars"][ip]])
    #                     else:
    #                         est[p_name_samples][ii].append(fit["sampler_params"][ip])
    #                         est[p_name][ii].append(fit["mean_pars"][ip])
    #                 else:
    #                     if len(est[p_name]) < ii + 1:
    #                         est[p_name_samples].append([])
    #                         est[p_name].append([])
    #                     jj = int(p_split.pop(0)) - 1
    #                     if len(p_split) == 0:
    #                         # 3D matrix parameters
    #                         if len(est[p_name][ii]) < jj + 1:
    #                             est[p_name_samples][ii].append([fit["sampler_params"][ip]])
    #                             est[p_name][ii].append([fit["mean_pars"][ip]])
    #                         else:
    #                             if len(est[p_name][ii]) < jj + 1:
    #                                 est[p_name_samples][ii].append([])
    #                                 est[p_name][ii].append([])
    #                             est[p_name_samples][ii][jj].append(fit["sampler_params"][ip])
    #                             est[p_name][ii][jj].append(fit["mean_pars"][ip])
    #                     else:
    #                         raise_not_implemented_error("Extracting of parameters of more than 3 dimensions is not " +
    #                                                     "implemented yet for vb!", self.logger)
    #     for key in est.keys():
    #         if isinstance(est[key], list):
    #             est[key] = np.squeeze(np.array(est[key]))
    #     return est
