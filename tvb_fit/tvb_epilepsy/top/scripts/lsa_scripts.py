import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.top.scripts.model_config_scripts import configure_model

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard


LOG = initialize_logger(__name__)


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


def from_hypothesis_to_model_config_lsa(hyp, head, model_params={}, lsa_params={},
                                        writer=None, plotter=None, logger=LOG, config=Config()):
    logger.info("\n\nRunning hypothesis: " + hyp.name)
    logger.info("\n\nCreating model configuration...")

    logger.info("\n\nConfigureing model...")
    model_configuration, builder = \
        configure_model(hyp, head.connectivity.normalized_weights, "EpileptorDP2D",
                        region_labels=head.connectivity.region_labels,
                        modelconfig_path=os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelConfig.h5"),
                        writer=writer, plotter=plotter, config=config, **model_params)

    logger.info("\n\nRunning LSA...")
    lsa_hypothesis, lsa_service = \
        run_lsa(hyp, model_configuration, lsa_service=None, region_labels=head.connectivity.region_labels,
                hypo_path=os.path.join(config.out.FOLDER_RES, hyp.name + ".h5"),
                writer=writer, plotter=plotter, config=Config(), **lsa_params)

    return model_configuration, lsa_hypothesis, builder, lsa_service