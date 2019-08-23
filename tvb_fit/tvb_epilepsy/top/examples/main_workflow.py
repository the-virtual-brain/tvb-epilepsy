# -*- coding: utf-8 -*-
import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.workflow.workflow import Workflow


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    SUBJECT = "TVB3"
    head_folder = os.path.join(user_home, 'Dropbox/Work/VBtech/VEP/results/tests/workflow', SUBJECT, "HeadD")
    SEEG_data = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "data/CC", "seeg", SUBJECT)
    output = os.path.join(user_home, 'Dropbox/Work/VBtech/VEP/results/tests/workflow', SUBJECT, "HeadD")
    config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)
    config.generic.CMDSTAN_PATH = config.generic.CMDSTAN_PATH + "_precompiled"

    wf = Workflow(config)
    wf.read_head()
    wf.set_attr("epi_name",  "preseeg")
    wf._hypo_manual["e_indices"] = [52, 53]
    wf.set_hypothesis()
    wf.run_lsa()
    wf.run_lsa_pse()
    wf._sim_params["rescale_x1eq"] = -1.3
    simTS = wf.simulate()
