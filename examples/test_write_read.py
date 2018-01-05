#!/usr/bin/env python

from main_vep import main_vep
from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects


if __name__ == "__main__":
    main_vep(test_write_read=True, pse_flag=True, sa_pse_flag=True, sim_flag=True)

