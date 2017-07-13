import os
import warnings

import h5py
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils import initialize_logger, ensure_unique_file, change_filename_or_overwrite, \
                                    set_list_item_by_reference_safely, get_list_or_tuple_item_safely, \
                                    list_or_tuple_to_dict, dict_to_list_or_tuple, sort_dict

logger = initialize_logger(__name__)

bool_inf_nan_none_empty = OrderedDict()
bool_inf_nan_none_empty.update({"True": True})
bool_inf_nan_none_empty.update({"False": False})
bool_inf_nan_none_empty.update({"inf": np.inf})
bool_inf_nan_none_empty.update({"nan": np.nan})
bool_inf_nan_none_empty.update({"None": None})
bool_inf_nan_none_empty.update({"''": ""})
bool_inf_nan_none_empty.update({"[]": []})
bool_inf_nan_none_empty.update({"{}": {}})
bool_inf_nan_none_empty.update({"()": ()})


class H5Model(object):

    def __init__(self, datasets_dict, metadata_dict):
        self.datasets_dict = datasets_dict
        self.metadata_dict = metadata_dict

    def add_or_update_metadata_attribute(self, key, value):
        self.metadata_dict.update({key: value})

    def add_or_update_datasets_attribute(self, key, value):
        self.datasets_dict.update({key: value})

    def append(self, h5_model):
        for key, value in h5_model.datasets_dict.iteritems():
            self.add_or_update_datasets_attribute(key, value)

        for key, value in h5_model.metadata_dict.iteritems():
            self.add_or_update_metadata_attribute(key, value)

    def write_to_h5(self, folder_name, file_name):
        """
        Store H5Model object to a hdf5 file
        """
        final_path, overwrite = change_filename_or_overwrite(folder_name, file_name)
        # final_path = ensure_unique_file(folder_name, file_name)

        if overwrite:
            try:
                os.remove(final_path)
            except:
                warnings.warn("\nFile to overwrite not found!")

        logger.info("Writing %s at: %s" % (self, final_path))

        h5_file = h5py.File(final_path, 'a', libver='latest')

        for attribute, field in self.datasets_dict.iteritems():
            h5_file.create_dataset("/" + attribute, data=field)

        for meta, val in self.metadata_dict.iteritems():
            h5_file.attrs.create(meta, val)

        h5_file.close()

    def convert_from_h5_model(self, obj=dict()):

        output = obj.__class__.__name__
        if np.in1d(output, ["tuple", "list"]):
            obj = list_or_tuple_to_dict(obj)

        data = dict()
        data.update(self.datasets_dict)
        data.update(self.metadata_dict)

        data = sort_dict(data)

        for key, value in data.iteritems():

            if key[0] == "/":
                key = key.split('/', 1)[1]

            build_hierarchical_object_recursively(obj, key, value)

        if np.in1d(output, ["tuple", "list"]):
            obj = dict_to_list_or_tuple(obj, output)

        return obj


def convert_to_h5_model(obj, name=""):
    h5_model = H5Model(OrderedDict(), OrderedDict())
    object_to_h5_model_recursively(h5_model, obj, name="")
    return h5_model


def object_to_h5_model_recursively(h5_model, obj, name=""):

    # Use in some cases the name of the class as key, when name is empty string. Otherwise, class+name = name.
    name_empty = (len(name) == 0)
    class_name = name_empty * obj.__class__.__name__ + name

    for key, val in bool_inf_nan_none_empty.iteritems():
        # Bool, inf, nan, None, or empty list/tuple/dict/str
        try:
            if obj is val or all(obj == val):
                h5_model.add_or_update_metadata_attribute(class_name, key)
                return
        except:
            continue

    if isinstance(obj, (float, int, long, complex, str, np.ndarray)):

        if isinstance(obj, (float, int, long, complex, str)):
            h5_model.add_or_update_metadata_attribute(class_name, obj)

        elif isinstance(obj, np.ndarray):
            h5_model.add_or_update_datasets_attribute(class_name, obj)

    else:

        # In any other case, make sure object is/becomes an alphabetically ordered dictionary:

        if isinstance(obj, (list, tuple)):
            obj = list_or_tuple_to_dict(obj)

        elif not(isinstance(obj, dict)):
            try:
                obj = vars(obj)
            except:
                logger.info("Object " + name + (len(name) > 0) * "/" + key + "cannot be assigned to h5_model because it"
                                                                             "is has no __dict__ property")
                return

        obj = sort_dict(obj)

        for key, value in obj.iteritems():

            key = name + (len(name) > 0) * "/" + key

            # call recursively...
            object_to_h5_model_recursively(h5_model, value, key)


def build_hierarchical_object_recursively(obj, key, value):

    if isinstance(obj, dict):
        set_field = lambda obj, key, value: obj.update({key: value})
        get_field = lambda obj, key: obj.get(key, None)

    elif isinstance(obj, list):
        set_field = lambda obj, key, value: set_list_item_by_reference_safely(int(key), value, obj)
        get_field = lambda obj, key: get_list_or_tuple_item_safely(obj, key)

    else:
        set_field = lambda obj, attribute, value: setattr(obj, attribute, value)
        get_field = lambda obj, attribute: getattr(obj, attribute, None)

    # Check whether value is an inf, nan, None, bool, or empty list/dict/tuple/str value
    try:
        bool_inf_nan_none_empty_value = bool_inf_nan_none_empty.get(value, "skip_bool_inf_nan_none_empty_value")
        if bool_inf_nan_none_empty_value != "skip_bool_inf_nan_none_empty_value":
            set_field(obj, key, bool_inf_nan_none_empty_value)
        return
    except:
        pass

    child_object = get_field(obj, key)
    if child_object is not None:
        set_field(obj, key, value)

    else:
        name = key.split('/', 1)[0]
        try:
            if name == key:
                set_field(obj, key, value)
                return 1
            else:
                child_key = key.split('/', 1)[1]
                child_object = get_field(obj, name)
                if child_object is None:
                    grandchild_name = child_key.split('/', 1)[0]
                    if grandchild_name.isdigit():
                        child_object = list()
                    else:
                        child_object = dict()
                    set_field(obj, name, child_object)
                build_hierarchical_object_recursively(child_object, child_key, value)

        except:
            warnings.warn("Failed to set attribute " + str(key) + "of object " + obj.__class__.__name__ + "!")


def read_h5_model(path):

    h5_file = h5py.File(path, 'r', libver='latest')

    datasets_dict = dict()
    metadata_dict = dict()

    for key, value in h5_file.attrs.iteritems():
        metadata_dict.update({key: value})

    datasets_keys = return_h5_dataset_paths_recursively(h5_file)

    for key in datasets_keys:
        datasets_dict.update({key: h5_file[key][()]})

    datasets_dict = sort_dict(datasets_dict)
    metadata_dict = sort_dict(metadata_dict)

    return H5Model(datasets_dict, metadata_dict)


def return_h5_dataset_paths_recursively(group):
    try:
        paths = []
        for key in group.keys():
            paths += return_h5_dataset_paths_recursively(group[key])

        return paths
    except:
        return [group.name]


if __name__ == "__main__":

    from tvb_epilepsy.base.constants import FOLDER_RES
    from tvb_epilepsy.base.utils import assert_equal_objects

    from copy import deepcopy

    obj = {"h5_model": H5Model({"a/b": np.array([1,2,3]), "a/c": np.array([1,2,3])},
                                  {"list0": ["l00", 1, {"d020": "a", "d021": []}]}),
              "dict": {"list0": ["l00", 1, {"d020": "a", "d021": [True, False, np.inf, np.nan, None, [], (), {}, ""]}]}}
    logger.info("\n\nOriginal object:\n" + str(obj))

    logger.info("\n\nWriting object to h5 file...")
    convert_to_h5_model(obj).write_to_h5(FOLDER_RES, "test_h5_model.h5")

    obj1 = read_h5_model(FOLDER_RES + "/test_h5_model.h5").convert_from_h5_model(deepcopy(obj))
    if assert_equal_objects(obj, obj1):
        print "\n\nRead identical object:\n" + str(obj1)
        logger.info("\n\nRead identical object:\n" + str(obj1))

    obj2 = read_h5_model(FOLDER_RES + "/test_h5_model.h5").convert_from_h5_model()
    if assert_equal_objects(obj, obj2):
        print "\n\nRead object as dictionary:\n" + str(obj2)
        logger.info("\n\nRead object as dictionary:\n" + str(obj2))



