
import os

import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.top.scripts.model_config_scripts import hypo_prefix, set_hypothesis, configure_model
from tvb_fit.tvb_epilepsy.service.workflow_epilepsy import WorkflowEpilepsy


class WorkflowConfigureModel(WorkflowEpilepsy):

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        super(WorkflowConfigureModel, self).__init__(config, reader, writer, plotter)
        self._epi_name = ""
        self._hypo_name = ""
        self._epi_path = ""
        self._hypo_folder = ""
        self._hypo_path = ""
        self._hypo_params = {"normalize_disease_values": [], "scale_x0": 1.0}
        self._hypo_builder = None
        self._hypothesis = None
        self._hypo_manual = \
            {"x0_indices": [], "x0_values": [], "e_indices": [], "e_values": [], "w_indices": [], "w_values": []}
        self._modelconfig = None
        self._model_name = "EpileptorDP"
        self._modelconfig_builder = None
        self._modelconfig_path = ""
        self._model_params = {}

    @property
    def epi_path(self):
        if len(self._epi_path) > 0:
            return self._epi_path
        else:
            return os.path.join(self.head_folder, self._epi_name, self._epi_name + ".h5")

    @property
    def hypo_name(self):
        if isinstance(self._hypothesis, DiseaseHypothesis):
            self._hypo_name = self._hypothesis.name
            return self._hypo_name
        else:
            return self._hypo_name

    @property
    def hypo_type(self):
        if isinstance(self._hypothesis, DiseaseHypothesis):
            return self._hypothesis.type
        return ""

    @property
    def hypo_prefix(self):
        return hypo_prefix(self.hypo_name, self.hypo_type)

    @property
    def hypo_folder(self):
        if os.path.isdir(os.path.dirname(self._hypo_path)):
            self._hypo_folder = os.path.dirname(self._hypo_path)
        if os.path.isdir(self._hypo_folder) > 0:
            return self._hypo_folder
        else:
            return os.path.join(self.res_folder, self._epi_name)

    @property
    def hypo_path(self):
        if len(self._hypo_path) > 0:
            return self._hypo_path
        else:
            return os.path.join(self.hypo_folder, self.hypo_prefix + ".h5")

    @property
    def hypo_foldername(self):
        hypo_foldername =  self.hypo_folder.split(os.sep)[-1]
        if hypo_foldername != self.res_folder.split(os.sep)[-1]:
            return hypo_foldername
        else:
            return ""

    @property
    def modelconfig_path(self):
        if len(self._modelconfig_path) == 0:
            self._modelconfig_path = os.path.join(self.hypo_folder, "ModelConfig.h5")
        return self._modelconfig_path

    @property
    def hypo_figsfolder(self):
        hypo_figsfolder = os.path.join(self.figs_folder, self.hypo_foldername)
        return hypo_figsfolder

    def _assert_number_of_regions(self):
        super(WorkflowConfigureModel, self)._assert_number_of_regions()
        if isinstance(self._hypothesis, DiseaseHypothesis):
            assert self._number_of_regions == self._hypothesis.number_of_regions
        if isinstance(self._modelconfig, ModelConfiguration):
            assert self._number_of_regions == self._modelconfig.number_of_regions

    @property
    def number_of_regions(self):
        self._assert_number_of_regions()
        return self._number_of_regions

    def set_hypothesis(self, write_hypo=True):
        if self._write_flag(write_hypo):
            self._ensure_folder(self.hypo_folder)
            writer = self._writer
        else:
            writer = None
        self._hypothesis, self._hypo_builder, self._hypo_name, self._hypo_path = \
           set_hypothesis(self.number_of_regions, self._hypothesis, self._hypo_manual, self._hypo_name, self.hypo_folder,
                          self._epi_name, self.epi_path, writer, self._config, **self._hypo_params)

    def configure_model(self, write_modelconfig=True, plot_state_space=True, **model_params):
        if self._write_flag(write_modelconfig):
            writer = self._writer
            self._ensure_folder(self._get_foldername(self.modelconfig_path))
        else:
            writer = None
        if self._plot_flag(plot_state_space):
            plotter = self._plotter
            self._ensure_folder(self.hypo_figsfolder)
        else:
            plotter = None
        self._model_params.update(model_params)
        self._modelconfig, self._modelconfig_builder = \
            configure_model(self.hypothesis, self.normalized_weights, self._model_name,
                            self._modelconfig_builder, self._modelconfig,  self.region_labels,
                            self.modelconfig_path,  "StateSpace", self.hypo_figsfolder,
                            writer,plotter, self._config, **self._model_params)

    @property
    def hypothesis(self):
        if not isinstance(self._hypothesis, DiseaseHypothesis):
            try:
                # Read hypothesis from file
                self._hypothesis = self._reader.read_hypothesis(self.hypo_path)
            except:
                self.set_hypothesis()
        return self._hypothesis

    @property
    def regions_disease_indices(self):
        return self.hypothesis.regions_disease_indices

    @property
    def healthy_regions_indices(self):
        return np.delete(self.all_regions_indices, self.regions_disease_indices).tolist()

    @property
    def modelconfig(self):
        if not isinstance(self._modelconfig, ModelConfiguration):
            try:
                self._modelconfig = \
                    self._reader.read_modelconfig(self.modelconfig_path, self._model_name)
            except:
                self.configure_model()
        return self._modelconfig

    @property
    def modelconfig_builder(self):
        if isinstance(self._modelconfig_builder, ModelConfigurationBuilder):
            return self._modelconfig_builder
        else:
            return ModelConfigurationBuilder(self.modelconfig)

    @property
    def model_connectivity(self):
        return self.modelconfig.connectivity
