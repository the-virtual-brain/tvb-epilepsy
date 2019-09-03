
import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.top.scripts.lsa_scripts import run_lsa, lsa_done
from tvb_fit.tvb_epilepsy.top.scripts.pse_scripts import run_lsa_pse
from tvb_fit.tvb_epilepsy.top.workflow.workflow_configure_model import WorkflowConfigureModel


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
            run_lsa(self.hypothesis, self.modelconfig, None, self.region_labels,
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
