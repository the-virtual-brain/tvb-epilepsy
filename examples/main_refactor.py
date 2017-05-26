"""
Entry point for working with refactored code
"""
import numpy

from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.equilibrum_service import EquilibrumComputationService
from tvb_epilepsy.base.utils import initialize_logger
from tvb_epilepsy.custom.read_write import write_h5_model
from tvb_epilepsy.custom.readers_custom import CustomReader
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP

if __name__ == "__main__":
    logger = initialize_logger(__name__)
    data_folder = '/WORK/Episense/trunk/demo-data/Head_TREC'

    reader = CustomReader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    x0_indices = [20]
    x0_values = numpy.random.normal(0.85, 0.02, (len(x0_indices),))

    hypothesis = DiseaseHypothesis("x0", head.connectivity, x0_indices, x0_values, [], [], "x0_Hypothesis")

    all_regions_one = numpy.ones((hypothesis.get_number_of_regions(),), dtype=numpy.float32)

    epileptor_model = EpileptorDP(Iext1=3.1 * all_regions_one, yc=all_regions_one,
                                  K=10 * all_regions_one / hypothesis.get_number_of_regions())

    equilibrum_service = EquilibrumComputationService(hypothesis, epileptor_model)

    model_configuration, lsa_hypothesis = equilibrum_service.configure_model_from_x0_hypothesis()

    write_h5_model(hypothesis.prepare_for_h5(), folder_name=data_folder, file_name=hypothesis.get_name() + ".h5")
    write_h5_model(lsa_hypothesis.prepare_for_h5(), folder_name=data_folder,
                   file_name=lsa_hypothesis.get_name() + ".h5")

    # plot_hypothesis_equilibrium_and_lsa(lsa_hypothesis, model_configuration, figure_dir=data_folder)
