# -*- coding: utf-8 -*-
import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.service.workflow.workflow import Workflow


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    SUBJECT = "TVB3"
    DROPBOX_HOME = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP')
    head_folder = os.path.join(DROPBOX_HOME, "results", "CC", SUBJECT, "HeadD")
    SEEG_data = os.path.join(DROPBOX_HOME, "data/CC", "seeg", SUBJECT)
    output = os.path.join(user_home, 'Dropbox/Work/VBtech/VEP/results/tests/workflow')
    config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)

    wf = Workflow(config)
    wf.read_head()
    wf.set_attr("epi_name",  "preseeg")
    wf._hypo_manual["e_indices"] = [52, 53]
    wf.set_hypothesis()
    wf.run_lsa()
    wf.run_lsa_pse()
    wf._sim_params["x1eq_rescale"] = -1.3
    simTS = wf.simulate()
