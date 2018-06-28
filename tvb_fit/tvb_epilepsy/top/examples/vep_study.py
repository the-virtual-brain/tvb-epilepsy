"""
Entry point for working with VEP
"""

import os

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_fit.tvb_epilepsy.top.examples.main_vep import main_vep


def vep_study():
    subjects_top_folder = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', 'results', "CC")
    subject_base_name = "TVB"
    subj_ids = [3] #[1, 2, 3, 4, 4]

    e_indices = [[]]# ([40, 42], [], [1, 26], [], [])
    ep_names = "clinical_hypothesis_preseeg" # (3 * ["clinical_hypothesis_preseeg"] + ["clinical_hypothesis_preseeg_right"]
                                             # + ["clinical_hypothesis_preseeg_bilateral"])

    for subj_id in range(4, len(subj_ids)):

        subject_name = subject_base_name + str(subj_ids[subj_id])
        head_path = os.path.join(subjects_top_folder, subject_name, "Head")
        e_inds = e_indices[subj_id]
        folder_results = os.path.join(head_path, ep_names[subj_id], "res")

        config = Config(head_path, Config.generic.MODE_JAVA, folder_results, True)

        main_vep(config, ep_name=ep_names[subj_id], K_unscaled=K_DEF, ep_indices=e_inds, hyp_norm=0.99, manual_hypos=[],
                 sim_type=["default", "realistic"], pse_flag=True, sa_pse_flag=False, sim_flag=True, n_samples=100)


if __name__ == "__main__":
    vep_study()