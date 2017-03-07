import os
import numpy
from tvb_epilepsy.base.utils import get_logger
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.custom.read_write import write_hypothesis
from tvb_epilepsy.custom.readers_custom import CustomReader

if __name__ == "__main__":
    logger = get_logger(__name__)

    logger.info("Reading from custom")
    data_folder = os.path.join("/WORK/Episense/trunk/demo-data", 'Head_JUNCH')
    reader = CustomReader()

    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    logger.info("Loaded Head " + str(head))
    logger.info("Loaded Connectivity " + str(head.connectivity))

    hypothesis = Hypothesis(head.number_of_regions, head.connectivity.weights, "EP Hypothesis")

    epi_name = "ep"
    epi_complete_path = os.path.join(data_folder, epi_name)

    epileptogenicity = reader.read_epileptogenicity(data_folder, epi_name)
    logger.info("Loaded epileptogenicity from " + epi_complete_path)

    epi_indices = numpy.arange(0, 88, 1)
    hypothesis.configure_e_hypothesis(epi_indices, epileptogenicity, epi_indices)

    hypothesis_name = "hypo.h5"

    write_hypothesis(hypothesis, epi_complete_path, hypothesis_name)
