import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list


def str_slice_to_ind_slice(old_slice, labels):
    slice_dict = {}
    for s in ["start", "stop", "step"]:
        if isinstance(getattr(old_slice, s), basestring):
            try:
                index = labels.index(getattr(old_slice, s))
            except:
                raise_value_error("Failed to find label slice " + s  + " of slice " + str(old_slice) +
                                  " within labels \n" + str(labels) + "!")
            slice_dict[s] = index
        else:
            slice_dict[s] = getattr(old_slice, s)

    return slice(slice_dict["start"], slice_dict["stop"], slice_dict["step"])


def verify_inds(inds, labels):

    inds = ensure_list(inds)

    # For every index...
    for iind, ind in enumerate(inds):
        # ...if it is a string, find its integer index
        if isinstance(ind, basestring):
            try:
                index = labels.index(ind)
            except:
                raise_value_error("Failed to find label indice " + ind + " within labels \n" + str(labels) + "!")
            inds[iind] = index

        # ...if it is a similar container object, continue recursively
        elif isinstance(inds[iind], (list, tuple, np.ndarray)):
            inds[iind] = verify_inds(inds[iind], labels)

        else:
            # ...do nothing and pray...
            pass

    return inds


def verify_index(index, labels):
    index = list(index)
    # For every index...
    for iind, (ind, label) in enumerate(zip(index, labels)):
        # ...if it is a slice object, call the corresponding function to treat it separetely...
        if isinstance(ind, slice):
            index[iind] = str_slice_to_ind_slice(ind, label)
        # ...if it is a Ellipsis...
        elif isinstance(ind, Ellipsis.__class__):
            pass
        # ...or a numpy.newaxis... do nothing!
        elif ind is np.newaxis:
            pass
        else:
            index[iind] = np.array(verify_inds(ind, label))
    return tuple(index)