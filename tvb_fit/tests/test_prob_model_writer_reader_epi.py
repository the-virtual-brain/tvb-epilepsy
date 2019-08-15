import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.service.probabilistic_models_builders import SDEProbabilisticModelBuilder

from tvb_io.h5_writer import H5Writer
from tvb_io.h5_reader import H5Reader


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    head_folder = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")

    if user_home == "/home/denis":
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "INScluster")
        config = Config(head_folder=head_folder, output_base=output, separate_by_run=False)
    elif user_home == "/Users/lia.domide":
        config = Config(head_folder="/WORK/episense/tvb-epilepsy/data/TVB3/Head",
                        raw_data_folder="/WORK/episense/tvb-epilepsy/data/TVB3/ts_seizure")
    else:
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "fit")
        config = Config(head_folder=head_folder, output_base=output, separate_by_run=False)

    # Read head
    reader = H5Reader()

    head = reader.read_head(config.input.HEAD)

    hyp_builder = HypothesisBuilder(head.connectivity.number_of_regions, config).set_normalize(0.99)
    e_indices = [1, 26]  # [1, 2, 25, 26]
    hypothesis = hyp_builder.build_hypothesis_from_file("clinical_hypothesis_postseeg", e_indices)

    model_configuration = \
        ModelConfigurationBuilder("Epileptor", head.connectivity.normalized_weights).\
            build_model_from_E_hypothesis(hypothesis)

    probabilistic_model = SDEProbabilisticModelBuilder(model_config=model_configuration).generate_model()
    H5Writer().write_probabilistic_model(probabilistic_model, config.out.FOLDER_RES, "TestProbModelorig.h5")

    probabilistic_model2 = reader.read_probabilistic_model(os.path.join(config.out.FOLDER_RES, "TestProbModelorig.h5"))
    H5Writer().write_probabilistic_model(probabilistic_model2, config.out.FOLDER_RES, "TestProbModelread.h5")

