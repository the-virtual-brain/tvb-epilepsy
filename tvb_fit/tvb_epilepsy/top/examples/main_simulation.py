import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter
from tvb_fit.tvb_epilepsy.top.scripts.model_config_scripts import set_hypothesis, configure_model
from tvb_fit.tvb_epilepsy.top.scripts.simulation_scripts import LOG, from_model_configuration_to_simulation

from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb_scripts.io.tvb_data_reader import TVBReader


def main_simulation(config=Config(), sim_types=["fitting", "reduced", "paper", "default", "realistic"]):

    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)
    LOG.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    plotter.plot_head(head)
    # --------------------------Hypothesis definition-----------------------------------

    hypotheses = []
    # Reading a h5 file:

    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]
    hypo_manual = {"x0_indices": x0_indices, "x0_values": x0_values, "e_indices": e_indices, "e_values": e_values}
    hypothesis = set_hypothesis(head.connectivity.number_of_regions, hypo_manual=hypo_manual,
                                hypo_folder=os.path.join(config.out.FOLDER_RES),
                                writer=writer, logger=LOG, config=config)[0]

    # --------------------------Configure model-----------------------------------
    model_configuration = configure_model(hypothesis, head.connectivity.normalized_weights, "EpileptorDP",
                                          region_labels=head.connectivity.region_labels,
                                          modelconfig_path=os.path.join(config.out.FOLDER_RES, "ModelConfig.h5"),
                                          writer=writer, plotter=plotter, config=config)[0]

    # --------------------------Simulate-----------------------------------
    # We don't want any time delays for the moment
    head.connectivity.tract_lengths *= config.simulator.USE_TIME_DELAYS_FLAG
    outputs = []
    for sim_type in ensure_list(sim_types):
        outputs.append(
            from_model_configuration_to_simulation(
                model_configuration, head, hypothesis,
                x1eq_rescale=1.3, sim_type=sim_type, compute_seeg=True,
                seeg_gain_mode="lin", hpf_flag=False, hpf_low=10.0, hpf_high=512.0, plot_spectral_raster=False,
                reader=reader, writer=writer, plotter=plotter, logger=LOG, config=config))


if __name__ == "__main__":
    head_folder = os.path.join(__file__.split("tvb_fit")[0][:-1], "data", "head")
    output = os.path.join(os.path.dirname(__file__), "outputs")
    # head_folder = os.path.join(os.path.expanduser("~"),
    #                            'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "HeadD")
    # output = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "tests")
    config = Config(head_folder=head_folder, output_base=output, separate_by_run=False)
    main_simulation(config, sim_types=["fitting", "reduced", "paper", "default", "realistic"])
