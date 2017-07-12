import os
import warnings

import h5py

import numpy as np

from tvb_epilepsy.base.utils import initialize_logger, ensure_unique_file, change_filename_or_overwrite

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

        data = {}
        data.update(self.metadata_dict)
        data.update(self.datasets_dict)

        for key, value in data.iteritems():
            if key[0] == "/":
                key.split('/', 1)[1]
            object = build_hierarchical_object_recursively(object, key, value)

        return object


def flatten_hierarchical_object_recursively(object, name, datasets_dict, metadata_dict):

    if isinstance(object, (list, tuple)):

        for ii, item in enumerate(object):
            key = name + "#" + str(ii)
            datasets_dict, metadata_dict = flatten_hierarchical_object_recursively(item, key, datasets_dict,
                                                                                   metadata_dict)
    else:

        if not(isinstance(object, dict)):
            object_dict = vars(object)

        for key, value in object_dict.iteritems():

            key = name + "/" + key

            if (isinstance(value, np.ndarray)):
                datasets_dict.update({key: value})

            elif value == {} or value == [] or value == () or value == "":
                datasets_dict.update({key: np.array(value)})

            elif isinstance(value, (float, int, long, complex, str)):
                metadata_dict.update({key: value})

            else: # if isinstance(value, dict):
                datasets_dict, metadata_dict = flatten_hierarchical_object_recursively(value, key, datasets_dict,
                                                                                       metadata_dict)

    return datasets_dict, metadata_dict


def object_to_h5_model(obj):

    datasets_dict = {}

    metadata_dict = {}

    if not(isinstance(obj, dict)):
        obj = vars(obj)

    for key, value in obj.iteritems():

        if (isinstance(value, np.ndarray)):
            datasets_dict.update({key: value})

        elif value == {} or value == [] or value == () or value == "":
            continue

        elif isinstance(value, (float, int, long, complex, str)):
            metadata_dict.update({key: value})

        else: # isinstance(value, dict):
            try:
                datasets_dict, metadata_dict = flatten_hierarchical_object_recursively(value, key, datasets_dict,
                                                                                       metadata_dict)
            except:
                warnings.warn("Not able to include attribute " + key + " to the h5_model!")
                continue
    h5_model = H5Model(datasets_dict, metadata_dict)

    return h5_model


def build_hierarchical_object_recursively(object, key, value):

    if isinstance(object, dict):
        set_field = lambda object, key, data: object.update({key: data})
        get_field = lambda object, key: object[key]
    else:
        set_field = lambda object, attribute, data: setattr(object, attribute, data)
        get_field = lambda object, attribute: getattr(object, attribute)

    name = key.split('/', 1)[0]
    if name == key:
        name_list_tuple = name.split('#', 1)[0]
        if name_list_tuple == name: # if it is NOT a list nor a tuple
            try:
                set_field(object, name, value)
            except:
                warnings.warn("Failed to set attribute " + str(name) + "to object " + object.get("__name__", "") + "!")
                return object
        else: # if it IS a list or tuple
            ind = name.split('#', 1)[1]
            try:
                get_field(object, name_list_tuple)[ind] = value
            except:
                warnings.warn("Failed to set attribute " + str(name) + "to object " + object.get("__name__", "") + "!")
                return object
    else:
        child_key = key.split('/', 1)[1]
        name_list_tuple = name.split('#', 1)[0]
        if name_list_tuple == name:  # if it is NOT a list nor a tuple
            child_object = get_field(object, name)
            set_field(child_object, build_hierarchical_object_recursively(child_object, child_key, value))

        else:
            ind = name.split('#', 1)[1]
            child_object = get_field(object, name_list_tuple) # this a list or a tuple of object
            temp = list(child_object)
            temp[ind] = build_hierarchical_object_recursively(child_object[ind], child_key, value)
            if child_object.__class__.__name__ == "tuple":
                child_object = tuple(temp)
            else:
                child_object = temp
            set_field(object, name_list_tuple, child_object)


    return object


def read_h5_model(path):

    h5_file = h5py.File(path, 'r', libver='latest')

    datasets_dict = {}
    metadata_dict = {}

    for key, value in h5_file.attrs.iteritems():
        metadata_dict.update({key: value})

    datasets_keys = return_h5_dataset_paths_recursively(h5_file)

    for key in datasets_keys:
        datasets_dict.update({key: h5_file[key]})

    return H5Model(datasets_dict, metadata_dict)


def return_h5_dataset_paths_recursively(group):
    try:
        paths = []
        for key in group.keys():
            paths += return_h5_dataset_paths_recursively(group[key])

        return paths
    except:
        return [group.name]


