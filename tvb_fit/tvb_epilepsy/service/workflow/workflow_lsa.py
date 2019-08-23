
import os

import numpy as np

from tvb_fit.samplers.probabilistic_sampler import ProbabilisticSampler

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import MAX_DISEASE_VALUE
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_fit.tvb_epilepsy.service.workflow.workflow_configure_model import WorkflowConfigureModel

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard
from tvb_scripts.utils.data_structures_utils import \
    list_of_dicts_to_dicts_of_ndarrays, dicts_of_lists_to_lists_of_dicts, linear_index_to_coordinate_tuples


class WorkflowLSA(WorkflowConfigureModel):

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        super(WorkflowLSA, self).__init__(config, reader, writer, plotter)
        self._lsa_params = {"lsa_method": self._config.calcul.LSA_METHOD,
                            "eigen_vectors_number_selection": self._config.calcul.EIGENVECTORS_NUMBER_SELECTION,
                            "eigen_vectors_number": None,
                            "weighted_eigenvector_sum": self._config.calcul.WEIGHTED_EIGENVECTOR_SUM,
                            "normalize_propagation_strength": False}
       
        self._lsa_service = None
        self._lsa_pse_params = {"n_samples": 100, "param_range": 0.1}
        self._lsa_pse_sampler = None
        self._lsa_pse_service = None
        self._lsa_pse_results = {}
        self._lsa_pse_path = ""

    @property
    def _lsa_done(self):
        return lsa_done(self.hypothesis)

    @property
    def lsa_pse_path(self):
        if len(self._lsa_pse_path) == 0:
            self._lsa_pse_path = os.path.join(self.hypo_folder, "LSA_PSE.h5")
        return self._lsa_pse_path

    @property
    def disease_propagation(self):
        lsa_propagation_indices, lsa_propagation_strengths = self.hypothesis.disease_propagation
        if not (len(lsa_propagation_indices) == len(lsa_propagation_strengths) > 0):
            self.run_lsa()
        return self.hypothesis.disease_propagation

    @property
    def disease_propagation_strengths(self):
        lsa_propagation_indices, lsa_propagation_strengths = self.disease_propagation()
        return lsa_propagation_strengths[lsa_propagation_indices]

    def run_lsa(self, write_lsa_hypo=True, plot_lsa=True):
        if self._write_flag(write_lsa_hypo):
            writer = self._writer
            self._ensure_folder(self.hypo_folder)
        else:
            writer = None
        if self._plot_flag(plot_lsa):
            plotter = self._plotter
            self._ensure_folder(self.hypo_figsfolder)
        else:
            plotter = None
        self._hypothesis, self._lsa_service = \
            run_lsa(self.hypothesis, self.modelconfig,  None, self.region_labels,
                    self.hypo_path, self._add_prefix("LSA", self.hypo_prefix), self.hypo_figsfolder,
                    writer, plotter, self._config, **self._lsa_params)

    def run_lsa_pse(self, write_pse_results=True, plot_lsa_pse=True):
        if not self._lsa_done:
            self.run_lsa()
        if self._write_flag(write_pse_results):
            self._ensure_folder(self._get_foldername(self.lsa_pse_path))
            writer = self._writer
        else:
            writer = None
        if self._plot_flag(plot_lsa_pse):
            plotter = self._plotter
            self._ensure_folder(self.hypo_figsfolder)
        else:
            plotter = None
        self._lsa_pse_results, self._lsa_pse_service, self._lsa_pse_sampler = \
            run_lsa_pse(self.hypothesis, self.modelconfig, self._modelconfig_builder,
                        self.lsa_service, self.region_labels, self._random_seed,
                        self.lsa_pse_path, self._add_prefix("LSA_PSE", self.hypo_prefix), self.hypo_figsfolder,
                        writer, plotter, self._logger, self._config, **self._lsa_pse_params)

    @property
    def lsa_service(self):
        if isinstance(self._lsa_service, LSAService):
            return self._lsa_service
        else:
            self._lsa_service = LSAService(**self._lsa_params)
            return self._lsa_service

    @property
    def lsa_pse_results(self):
        if len(self._lsa_pse_results) == 0:
            if os.path.isfile(self.lsa_pse_path):
                self._lsa_pse_results = self._reader.read_dictionary(self.lsa_pse_path)
            else:
                self.run_lsa_pse()
        return self._lsa_pse_results


def lsa_done(hypothesis=None):
    lsa_done_flag = False
    if isinstance(hypothesis, DiseaseHypothesis):
        lsa_propagation_indices, lsa_propagation_strengths = hypothesis.disease_propagation
        lsa_done_flag = (len(lsa_propagation_indices) == len(lsa_propagation_strengths) > 0)
    return lsa_done_flag


def run_lsa(hypothesis, modelconfig, lsa_service=None, region_labels=[],
            hypo_path="", figname="", hypo_figsfolder="",
            writer=True, plotter=True, config=Config(), **lsa_params):
    if not isinstance(lsa_service, LSAService):
        lsa_service = LSAService(**lsa_params)
    hypothesis = lsa_service.run_lsa(hypothesis, modelconfig)
    if plotter:
        plotter.plot_lsa(hypothesis, modelconfig,
                         lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, region_labels, None,
                         title=figname, lsa_service=lsa_service)
        if os.path.isdir(hypo_figsfolder) and (hypo_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(hypo_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("StateSpace")))
            move_overwrite_files_to_folder_with_wildcard(hypo_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("LSA")))
    if writer:
        writer.write_hypothesis(hypothesis, hypo_path)

    return hypothesis, lsa_service


def run_lsa_pse(hypothesis, modelconfig, modelconfig_builder, lsa_service, region_labels, random_seed=0,
                lsa_pse_path="", figname="", hypo_figsfolder="",
                writer=None, plotter=None, logger=None, config=Config(), **lsa_pse_params):
    if not lsa_done(hypothesis):
        from tvb_fit.tvb_epilepsy.service.workflow.workflow_configure_model import hypo_prefix
        hypo_prfx = hypo_prefix(hypothesis.name, hypothesis.type) + "_"
        hypo_path = os.path.join(os.path.dirname(lsa_pse_path), hypo_prfx + ".h5")
        hypothesis, lsa_service = run_lsa(hypothesis, modelconfig, lsa_service, region_labels,
                                          hypo_path, hypo_prfx+"LSA", hypo_figsfolder,
                                          writer, plotter, config)
    n_samples = lsa_pse_params.get("n_samples", 100)
    param_range = lsa_pse_params.get("param_range", 0.1)
    all_regions_indices = np.arange(hypothesis.number_of_regions)
    healthy_regions_indices = np.delete(all_regions_indices, hypothesis.regions_disease_indices).tolist()
    # This a specific example function. Overload for different applications
    pse_params = {"path": [], "indices": [], "name": [], "samples": []}
    lsa_pse_sampler = \
        ProbabilisticSampler(n_samples=n_samples, random_seed=random_seed)

    # First build from the hypothesis the input parameters of the parameter search exploration.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)

    # x0 parameters'sampling
    for ii in range(len(hypothesis.x0_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(region_labels[hypothesis.x0_indices[ii]]) + " Excitability")

        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.x0_values[ii],  # loc
                                                        param_range / 3.0),  # scale
                                                        probability_distribution="norm",
                                                        high=MAX_DISEASE_VALUE, shape=(1,)))

    # e parameters'sampling
    for ii in range(len(hypothesis.e_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(region_labels[hypothesis.e_indices[ii]]) + " Epileptogenicity")
        # Now generate samples using a truncated uniform distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.e_values[ii],  # loc
                                                        param_range / 3.0),  # scale
                                                       probability_distribution="norm",
                                                       high=MAX_DISEASE_VALUE, shape=(1,)))

    # w parameters'sampling
    for ii in range(len(hypothesis.w_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(hypothesis.w_indices[ii], modelconfig.connectivity.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(region_labels[inds[0][0]]) + "-" +
                                      str(region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds) + "]")
        # Now generate samples using a truncated normal distribution
        pse_params["samples"].append(
            lsa_pse_sampler.generate_samples(parameter=(hypothesis.w_values[ii],  # loc
                                                        param_range * hypothesis.w_values[ii]),  # scale
                                                        probability_distribution="norm", low=0.0, shape=(1,)))

    # Global coupling jitter
    kloc = modelconfig_builder.K_unscaled[0]
    pse_params["path"].append("model_configuration_builder.K_unscaled")
    pse_params["indices"].append(all_regions_indices)
    # Now generate samples using a truncated normal distribution
    pse_params["samples"].append(
        lsa_pse_sampler.generate_samples(parameter=(0.1 * kloc,  # loc
                                                    2 * kloc),  # scale
                                                    probability_distribution="uniform", low=1.0, shape=(1,)))
    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)

    # Add a random jitter to the healthy regions...:
    n_params = len(healthy_regions_indices)
    samples = lsa_pse_sampler.generate_samples(parameter=(0.0,  # loc
                                                          param_range / 10),  # scale
                                                          probability_distribution="norm", shape=(n_params,))
    for ii in range(n_params):
        pse_params_list.append({"path": "model_configuration_builder.e_values", "samples": samples[ii],
                                "indices": [healthy_regions_indices[ii]], "name": "e_values"})

    # Now run pse service to generate output samples:
    lsa_pse_service = LSAPSEService(hypothesis=hypothesis, params_pse=pse_params_list)
    lsa_pse_results, execution_status = \
        lsa_pse_service.run_pse(modelconfig.connectivity, False, modelconfig_builder, lsa_service)
    logger.info(lsa_pse_service.__repr__())
    lsa_pse_results = list_of_dicts_to_dicts_of_ndarrays(lsa_pse_results)

    # Compute statistical estimates across samples:
    for key in lsa_pse_results.keys():
        lsa_pse_results[key + "_mean"] = np.mean(lsa_pse_results[key], axis=0)
        lsa_pse_results[key + "_std"] = np.std(lsa_pse_results[key], axis=0)

    # Plot samples
    if plotter:
        plotter.plot_lsa(hypothesis, modelconfig,
                         lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, region_labels, lsa_pse_results,
                         title=figname, lsa_service=lsa_service)
        if os.path.isdir(hypo_figsfolder) and (hypo_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(hypo_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("LSA_PSE")))

    if writer:
        writer.write_dictionary(lsa_pse_results, lsa_pse_path)

    return lsa_pse_results, lsa_pse_service, lsa_pse_sampler