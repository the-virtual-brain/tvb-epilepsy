import os
from collections import OrderedDict
import numpy as np

from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer


NUMBER_OF_REGIONS = 87

if __name__ == "__main__":

    User = os.path.expanduser("~")

    base_folder = os.path.join(User, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC")
    output_base = os.path.join(base_folder, "testing_heads")

    config = Config(output_base=output_base, separate_by_run=True)

    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)

    patients_included = []

    attributes = ["areas", "normalized_weights", "weights", "tract_lengths"]
    stats = dict((attribute, []) for attribute in attributes)

    head = None
    for ii in range(1, 31):

        patient = "TVB" + str(ii)
        logger.info("Patient TVB " + str(ii) + ":")

        head_folder = os.path.join(base_folder, patient, "Head")

        try:
            if head is not None:
                del head
            head = H5Reader().read_head(head_folder)
        except:
            head = None
            logger.warning("Failed reading head of patient TVB " + str(ii) + "!")
            if not os.path.isdir(head_folder):
                logger.warning("Head folder of patient TVB " + str(ii) + " is not an existing directory!")
            continue

        if head.number_of_regions == NUMBER_OF_REGIONS:
            logger.info("Number of regions: " + str(head.number_of_regions))
        else:
            logger.warning("Excluding patient TVB " + str(ii) + " because of a head of "
                           + str(head.number_of_regions) + " regions!")
            continue

        connections_sum = head.connectivity.normalized_weights.sum(axis=1)
        if np.all(connections_sum > 0):
            for attribute in stats.keys():
                stats[attribute].append(getattr(head.connectivity, attribute))
        else:
            zero_conn_regions = np.where(connections_sum == 0)[0]
            logger.warning("Excluding patient TVB " + str(ii)
                           + " because of the following regions with zero connectivity!:\n"
                           + str(zero_conn_regions))
            continue

        patients_included.append(ii)

    patients_included = np.array(patients_included)
    n_patients_included = patients_included.size

    results = OrderedDict()
    results["number_of_patients"] = n_patients_included
    results["patients_included"] = np.array(["TVB" + str(p) for p in patients_included])
    results["patients_ids_included"] = np.array(patients_included)
    for attribute in stats.keys():
        stats[attribute] = np.stack(stats[attribute], axis=-1)
        results[attribute + "_mean"] = stats[attribute].mean(axis=-1).squeeze()

    H5Writer().write_dictionary(results, os.path.join(output_base, "heads_stats.h5"))







