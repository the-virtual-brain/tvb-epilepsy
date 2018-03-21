from collections import OrderedDict

import numpy as np
from numpy.core.multiarray import ndarray

from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, formal_repr, sort_dict
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error


def str_slice_to_ind_slice(old_slice, labels):
    slice_dict = {}
    labels = ensure_list(labels)
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
    labels = ensure_list(labels)

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
    index = ensure_list(index)
    # For every index...
    for iind, (ind, label) in enumerate(zip(index, ensure_list(labels))):
        # ...if it is a slice object, call the corresponding function to treat it separetely...
        if isinstance(ind, slice):
            index[iind] = str_slice_to_ind_slice(ind, label)
        # ...if it is a Ellipsis...
        elif isinstance(ind, Ellipsis.__class__):
            pass
        # ...or a numpy.newaxis... do nothing!
        elif ind is np.newaxis:
            pass
        elif isinstance(index[iind], (list, tuple, np.ndarray)):
            index[iind] = np.array(verify_inds(ind, label))
    return tuple(index)


def unravel_index(index, shape):
    dummy = np.zeros(shape)
    dummy[index] = 1
    return  np.nonzero(dummy)


def marginal_index(index, shape):
    index = unravel_index(index, shape)
    index = [np.unique(ind) for ind in index]
    return tuple(index)


class LabelledArray(ndarray):

    _labels = []

    def __new__(cls, input_array, labels=[]):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj._labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, '_labels', [])

    def _slice_labels(self, index):
        # TODO: make this work!
        index = marginal_index(index, self.shape)
        labels = []
        for label, ind in zip(self._labels, index):
            if len(label) > 0:
                labels.append(label[ind])
            else:
                labels.append(np.array([]))
        return labels

    def slice(self, index):
        index = verify_index(index, self._labels)
        return LabelledArray(self.__getitem__(index), self._slice_labels(index))

    def __getitem__(self, index):
        index = verify_index(index, self._labels)
        return np.array(super(LabelledArray, self).__getitem__(index))

    def __setitem__(self, index, data):
        super(LabelledArray, self).__setitem__(verify_index(index, self._labels), data)

    def __repr__(self):
        d = OrderedDict({"array": super(LabelledArray, self).__repr__(),
                         "labels": str(self._labels)})
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()