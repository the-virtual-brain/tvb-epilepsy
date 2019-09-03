from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter

from tvb_fit.tvb_epilepsy.top.scripts.model_config_scripts import set_hypothesis
from tvb_fit.tvb_epilepsy.top.scripts.pse_scripts import LOG, pse_from_hypothesis

from tvb_scripts.io.tvb_data_reader import TVBReader


def main_pse(config=Config()):

    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    head = reader.read_head(config.input.HEAD)
    plotter = Plotter(config)

    # --------------------------Manual Hypothesis definition-----------------------------------
    # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis:
    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]
    hypo_manual = {"x0_indices": x0_indices, "x0_values": x0_values, "e_indices": e_indices, "e_values": e_values}
    hyp_x0_E = set_hypothesis(head.connectivity.number_of_regions, hypo_manual=hypo_manual,
                              hypo_folder=os.path.join(config.out.FOLDER_RES),
                              writer=writer, logger=LOG, config=config)[0]

    # Now running the parameter search analysis:
    LOG.info("running PSE LSA...")
    model_configuration_builder, model_configuration, lsa_service, lsa_hypothesis, sa_results, pse_results = \
        pse_from_hypothesis(hyp_x0_E, head, n_samples=100, param_range=0.1,
                            writer=writer, plotter=plotter, logger=LOG, config=config)


if __name__ == "__main__":
    import os
    head_folder = os.path.join(__file__.split("tvb_fit")[0][:-1], "data", "head")
    output = os.path.join(os.path.dirname(__file__), "outputs")
    # head_folder = os.path.join(os.path.expanduser("~"),
    #                            'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "HeadD")
    # output = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "tests")
    config = Config(head_folder=head_folder, output_base=output, separate_by_run=False)
    main_pse(config)
