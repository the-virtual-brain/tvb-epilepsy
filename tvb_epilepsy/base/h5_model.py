import os
import warnings

import h5py
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils import initialize_logger, ensure_unique_file, change_filename_or_overwrite, \
                                    set_list_item_by_reference_safely, list_or_tuple_to_dict, sort_dict

logger = initialize_logger(__name__)


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

    def convert_to_object(self, object=dict()):

        data = dict()
        data.update(self.datasets_dict)
        data.update(self.metadata_dict)

        data = sort_dict(data)

        for key, value in data.iteritems():

            if key[0] == "/":
                key = key.split('/', 1)[1]

            build_hierarchical_object_recursively(object, key, value)

        return object


def flatten_hierarchical_object_recursively(object, name, datasets_dict, metadata_dict):

    if isinstance(object, (list, tuple)):
        object = list_or_tuple_to_dict(object)

    elif not(isinstance(object, dict)):
        object = vars(object)

    object = sort_dict(object)

    for key, value in object.iteritems():

        key = name + "/" + key

        if (isinstance(value, np.ndarray)):
            datasets_dict.update({key: value})

        elif value == [] or value == () or value == "":
            datasets_dict.update({key: np.array(value)})

        elif value == {} or value is None:
            pass

        elif isinstance(value, (float, int, long, complex, str)):
            metadata_dict.update({key: value})

        else: # if isinstance(value, dict):
            flatten_hierarchical_object_recursively(value, key, datasets_dict, metadata_dict)



def object_to_h5_model(object):

    datasets_dict = OrderedDict()

    metadata_dict = OrderedDict()

    if isinstance(object, (list, tuple)):
        object = list_or_tuple_to_dict(object)

    elif not(isinstance(object, dict)):
        object = vars(object)

    object = sort_dict(object)

    for key, value in object.iteritems():

        if (isinstance(value, np.ndarray)):
            datasets_dict.update({key: value})

        elif value == [] or value == () or value == "":
            datasets_dict.update({key: np.array(value)})

        elif value == {} or value is None:
            pass

        elif isinstance(value, (float, int, long, complex, str)):
            metadata_dict.update({key: value})

        else: # isinstance(value, dict):
            try:
                flatten_hierarchical_object_recursively(value, key, datasets_dict, metadata_dict)
            except:
                warnings.warn("Not able to include attribute " + key + " to the h5_model!")
                continue

    h5_model = H5Model(datasets_dict, metadata_dict)

    return h5_model


def get_list_or_tuple_item_safely(object, key):
    try:
        return object[int(key)]
    except:
        return None


def build_hierarchical_object_recursively(object, key, value):

    if isinstance(object, dict):
        set_field = lambda object, key, value: object.update({key: value})
        get_field = lambda object, key: object.get(key, None)

    elif isinstance(object, list):
        set_field = lambda object, key, value: set_list_item_by_reference_safely(int(key), value, object)
        get_field = lambda object, key: get_list_or_tuple_item_safely(object, key)

    else:
        set_field = lambda object, attribute, value: setattr(object, attribute, value)
        get_field = lambda object, attribute: getattr(object, attribute, None)

    child_object = get_field(object, key)
    if child_object is not None:
        set_field(object, key, value)

    else:
        name = key.split('/', 1)[0]
        try:
            if name == key:
                set_field(object, key, value)
                return 1
            else:
                child_key = key.split('/', 1)[1]
                child_object = get_field(object, name)
                if child_object is None:
                    grandchild_name = child_key.split('/', 1)[0]
                    if grandchild_name.isdigit():
                        child_object = list()
                    else:
                        child_object = dict()
                    set_field(object, name, child_object)
                build_hierarchical_object_recursively(child_object, child_key, value)

        except:
            warnings.warn("Failed to set attribute " + str(key) + "to object " + object.get("__name__", "") + "!")


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

    object = {"h5_model": H5Model({"a/b": np.array([1,2,3]), "a/c": np.array([1,2,3])},
                                  {"list0": ["l00", 1, {"d020": "a", "d021": [1,2,3]}]}),
              "dict": {"list0": ["l00", 1, {"d020": "a", "d021": [1,2,3]}]}}
    logger.info("\n\nOriginal object:\n" + str(object))

    logger.info("\n\nWritine object to h5 file...")
    object_to_h5_model(object).write_to_h5(FOLDER_RES,"test_h5_model.h5")

    object1 = read_h5_model(FOLDER_RES + "/test_h5_model.h5").convert_to_object(deepcopy(object))
    assert_equal_objects(object, object1)
    logger.info("\n\nRead identical object:\n" + str(object1))

    object2 = read_h5_model(FOLDER_RES + "/test_h5_model.h5").convert_to_object()
    assert_equal_objects(object, object2)
    logger.info("\n\nRead object as dictionary:\n" + str(object2))



