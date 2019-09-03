# -*- coding: utf-8 -*-
import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.top.workflow.workflow import Workflow


if __name__ == "__main__":

    head_folder = os.path.join(__file__.split("tvb_fit")[0][:-1], "data", "head")
    output = os.path.join(os.path.dirname(__file__), "outputs")

    # user_home = os.path.expanduser("~")
    # SUBJECT = "TVB3"
    # DROPBOX_HOME = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP')
    # head_folder = os.path.join(DROPBOX_HOME, "results", "CC", SUBJECT, "HeadD")
    # SEEG_data = os.path.join(DROPBOX_HOME, "data/CC", "seeg", SUBJECT)
    # output = os.path.join(user_home, 'Dropbox/Work/VBtech/VEP/results/tests/workflow')
    config = Config(head_folder=head_folder,  output_base=output, separate_by_run=False)  # raw_data_folder=SEEG_data,

    wf = Workflow(config)
    wf.read_head()
    # wf.set_attr("epi_name",  "preseeg")
    wf._hypo_manual["e_indices"] = [52, 53]
    wf._hypo_manual["e_values"] = [0.9, 0.75]
    wf.set_hypothesis()
    wf.run_lsa()
    wf.run_lsa_pse()
    wf._sim_params["x1eq_rescale"] = -1.3
    simTS = wf.simulate()
