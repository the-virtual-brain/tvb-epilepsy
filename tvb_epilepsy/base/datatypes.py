from collections import OrderedDict

from numpy import ndarray
import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, ensure_list


logger = initialize_logger(__name__)


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


class LabelledArray(ndarray):

    _labels = np.array([])

    def __new__(cls, input_array, labels=np.array([])):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj._labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, '_labels', np.array([]))

    def __getitem__(self, index):
        index = verify_index(index, self._labels)
        try:
            labels = []
            for ind in index:
                labels.append(self._labels[ind])
        except:
            warning("Unable to slice labels!")
            labels = np.array([])
        return LabelledArray(super(LabelledArray, self).__getitem__(index), labels)

    def __setitem__(self, index, data):
        super(LabelledArray, self).__setitem__(verify_index(index, self._labels), data)


    def __repr__(self):
        d = OrderedDict({"array": super(LabelledArray, self).__repr__(),
                         "labels": str(self._labels)})
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()


# TODO: on medium term, we should remove this module.
class DictDot(object):

    def __init__(self, d):
        self.dict = dict(d)
        for key in self.dict.keys():
            if isinstance(self.dict[key], OrderedDict):
                self.dict[key] = OrderedDictDot(self.dict[key])
            elif isinstance(self.dict[key], dict):
                self.dict[key] = DictDot(self.dict[key])

    def __getattr__(self, item):
        try:
            return dict.__getattr__(self.dict, item)
        except:
            try:
                return self.dict[item]
            except:
                try:
                    for key in self.dict.keys():
                        if isinstance(key, basestring) and key.find(item) >= 0:
                            logger.warning("Item with key " + item + " not found!" +
                                    "\nReturning item with key " + key + " instead!")
                            return self.dict[key]
                except KeyError as e:
                    raise AttributeError(e)

    def __getitem__(self, item):
        return self.dict[item]


class OrderedDictDot(object):

    def __init__(self, d):
        self.dict = dict(d)
        for key in self.dict.keys():
            if isinstance(self.dict[key], OrderedDict):
                self.dict[key] = OrderedDictDot(self.dict[key])
            elif isinstance(self.dict[key], dict):
                self.dict[key] = DictDot(self.dict[key])

    def __getattr__(self, item):
        try:
            return OrderedDict.__getattr__(self.dict, item)
        except:
            try:
                return self.dict[item]
            except:
                try:
                    for key in self.dict.keys():
                        if isinstance(key, basestring) and key.find(item) >= 0:
                            logger.warning("Item with key " + item + " not found!" +
                                    "\nReturning item with key " + key + " instead!")
                            return self.dict[key]
                except KeyError as e:
                    raise AttributeError(e)

    def __getitem__(self, item):
        return self.dict[item]