import os

import numpy as np
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard


LOG = initialize_logger(__name__)


def hypo_prefix(hypo_name="", hypo_type=""):
    prefix = []
    for p in [hypo_type, hypo_name]:
        if len(p) > 0:
            prefix.append(p)
    return "_".join(prefix)


def set_hypothesis(number_of_regions, hypothesis=None, hypo_manual={}, hypo_name="", hypo_folder="",
                   epi_name="", epi_path="", writer=None, config=Config(), **hypo_params):
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


def configure_model(hypothesis, normalized_weights, model_name, modelconfig_builder=None, modelconfig=None,
                    region_labels=[], modelconfig_path="", figure_name="", hypo_figsfolder="",
                    writer=None, plotter=None, config=Config(), **model_params):
    if isinstance(modelconfig, ModelConfiguration):
        # Build modelconfig from another model configuration
        modelconfig = ModelConfigurationBuilder().build_model_config_from_model_config(modelconfig)
    else:
        # Build model configuration from hypothesis
        if not(isinstance(modelconfig_builder, ModelConfigurationBuilder)):
            modelconfig_builder = ModelConfigurationBuilder(model_name, normalized_weights, **model_params)
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
