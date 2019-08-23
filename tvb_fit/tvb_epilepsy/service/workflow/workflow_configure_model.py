
import os

import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.service.workflow.workflow_epilepsy_base import WorkflowEpilepsyBase


from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard


class WorkflowConfigureModel(WorkflowEpilepsyBase):

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
        self._model_type = "EpileptorDP"
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
           set_hypothesis(self.number_of_regions, self._hypo_manual, self._hypo_name, self.hypo_folder,
                          self._hypothesis, self._epi_name, self.epi_path, self._config, writer, **self._hypo_params)

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
            configure_model(self.hypothesis, self.normalized_weights, self._model_type,
                            self._modelconfig_builder, self._modelconfig, writer, self.modelconfig_path,
                            plotter, "StateSpace", self.hypo_figsfolder, self.region_labels, **self._model_params)

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
                    self._reader.read_modelconfig(self.modelconfig_path, self._model_type)
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


def hypo_prefix(hypo_name="", hypo_type=""):
    prefix = []
    for p in [hypo_type, hypo_name]:
        if len(p) > 0:
            prefix.append(p)
    return "_".join(prefix)


def set_hypothesis(number_of_regions, hypo_manual={}, hypo_name="", hypo_folder="", hypothesis=None,
                   epi_name="", epi_path="", config=Config(), writer=None, **hypo_params):
    _hypo_params = {"normalize_disease_values": [], "scale_x0": 1.0}
    _hypo_params.update(hypo_params)
    _hypo_manual = \
        {"x0_indices": [], "x0_values": [], "e_indices": [], "e_values": [], "w_indices": [], "w_values": []}
    _hypo_manual.update(hypo_manual)
    if os.path.isfile(epi_path):
        # Build hypothesis from epileptogenicity (disease values) file
        hypo_builder = HypothesisBuilder(number_of_regions, config). \
            set_normalize(_hypo_params["normalize_disease_values"])
        hypothesis = \
            hypo_builder.build_hypothesis_from_file(epi_name, _hypo_manual["e_indices"])
    else:
        if isinstance(hypothesis, DiseaseHypothesis):
            # Build hypothesis from another hypothesis
            hypo_builder = HypothesisBuilder(number_of_regions, config). \
                set_attributes_based_on_hypothesis(hypothesis)
            scale_x0 = 1.0
        else:
            hypo_builder = HypothesisBuilder(number_of_regions, config). \
                set_normalize(_hypo_params["normalize_disease_values"])
            scale_x0 = _hypo_params["scale_x0"]
        # Further setting/modification of hypothesis
        if len(_hypo_manual["e_indices"]) > 0 and len(_hypo_manual["e_values"]) > 0:
            hypo_builder.set_e_hypothesis(_hypo_manual["e_indices"],
                                          _hypo_manual["e_values"])
        if len(_hypo_manual["x0_indices"]) > 0 and len(_hypo_manual["x0_values"]) > 0:
            _hypo_manual["x0_values"] = \
                (scale_x0 * np.array(_hypo_manual["x0_values"])).tolist()
            hypo_builder.set_x0_hypothesis(_hypo_manual["x0_indices"],
                                           _hypo_manual["x0_values"])
        if len(_hypo_manual["w_indices"]) > 0 and len(_hypo_manual["w_values"]) > 0:
            hypo_builder.set_w_hypothesis(_hypo_manual["w_indices"], _hypo_manual["w_values"])

        # Now build the hypothesis
        hypothesis = hypo_builder.build_hypothesis()

        # Optional hypothesis name setting
    if len(hypo_name) > 0:
        hypothesis.name = hypo_name
    else:
        hypo_name = hypothesis.name

    hypo_path = os.path.join(hypo_folder, "_".join([hypothesis.type, hypothesis.name]) + ".h5")
    if writer:
        if not os.path.isdir(hypo_folder):
            os.makedirs(os.path.dirname(hypo_folder))
        writer.write_hypothesis(hypothesis, hypo_path)

    return hypothesis, hypo_builder, hypo_name, hypo_path


def configure_model(hypothesis, normalized_weights, model_type, modelconfig_builder=None, modelconfig=None,
                    config=Config(), writer=None, modelconfig_path="",
                    plotter=None, figure_name="", hypo_figsfolder="", region_labels=[], **model_params):
    if isinstance(modelconfig, ModelConfiguration):
        # Build modelconfig from another model configuration
        modelconfig = ModelConfigurationBuilder().build_model_config_from_model_config(modelconfig)
    else:
        # Build model configuration from hypothesis
        if not(isinstance(modelconfig_builder, ModelConfigurationBuilder)):
            modelconfig_builder = ModelConfigurationBuilder(model_type, normalized_weights, **model_params)
        if hypothesis.type == "e":
            modelconfig = modelconfig_builder.build_model_from_E_hypothesis(hypothesis)
        else:
            modelconfig = modelconfig_builder.build_model_from_hypothesis(hypothesis)

        if plotter:
            plotter.plot_state_space(modelconfig, region_labels,
                                     special_idx=hypothesis.regions_disease_indices, figure_name=figure_name)
            if os.path.isdir(hypo_figsfolder) and (hypo_figsfolder != config.out.FOLDER_FIGURES):
                move_overwrite_files_to_folder_with_wildcard(hypo_figsfolder,
                                                             os.path.join(config.out.FOLDER_FIGURES,
                                                                          wildcardit("StateSpace")))

        if writer:
            writer.write_model_configuration(modelconfig, modelconfig_path)

    return modelconfig, modelconfig_builder
