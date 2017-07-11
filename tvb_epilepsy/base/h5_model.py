import os
import warnings

import h5py

import numpy

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

    def convert_to_object(self, object=dict(), children_objects={}):

        data = {}
        data.update(self.metadata_dict)
        data.update(self.datasets_dict)

        object = build_hierarchical_object_recursively(data, object, children_objects)

        return object


def object_to_h5_model(obj):

    datasets_dict = {}

    metadata_dict = {}

    if not(isinstance(obj, dict)):
        obj = vars(obj)

    for key, value in obj.iteritems():

        if (isinstance(value, numpy.ndarray)):
            datasets_dict.update({key: value})

        elif isinstance(value, list):
            datasets_dict.update({key: numpy.array(value)})

        elif isinstance(value, dict):
            datasets_dict, metadata_dict = flatten_hierarchical_object_recursively(value, key, datasets_dict,
                                                                                   metadata_dict)
        else:
            if isinstance(value, (float, int, long, complex, str)):
                metadata_dict.update({key: value})
    #
    # else:
    #
    #     for key, value in vars(obj).iteritems():
    #         if (isinstance(value, numpy.ndarray)):
    #             datasets_dict.update({key: value})
    #         elif isinstance(value, list):
    #             datasets_dict.update({key: numpy.array(value)})
    #         elif isinstance(value, dict):
    #             datasets_dict, metadata_dict = flatten_dict_recursively(value, key, datasets_dict, metadata_dict)
    #         else:
    #             if isinstance(value, (float, int, long, complex, str)):
    #                 metadata_dict.update({key: value})

    h5_model = H5Model(datasets_dict, metadata_dict)

    return h5_model


def build_hierarchical_object_recursively(in_object, add_object, children_objects={}):

    if isinstance(in_object, dict):
        set_field = lambda object, key, data: object.update({key: data})
    else:
        set_field = lambda object, attribute, data: setattr(object, attribute, data)

    if isinstance(add_object, dict):
        add_object_dict = add_object
    else:
        add_object_dict = vars(add_object)

    for key, value in add_object_dict.iteritems():
        name = key.split("/")[0]
        if name == key:
            try:
                set_field(in_object, name, value)
            except:
                warnings.warn("Failed to set attribute " + str(name) + "to object " + in_object.get("__name__") + "!")
        else:
            set_field(in_object, name,
                      build_hierarchical_object_recursively(value, children_objects.get(name.split("/")[0], dict()),
                                                            children_objects=children_objects))

    return object


def flatten_hierarchical_object_recursively(object, name, datasets_dict, metadata_dict):

    if not(isinstance(object, dict)):
        object_dict = vars(object)

    for key, value in object_dict.iteritems():

        key = name + "/" + key

        if (isinstance(value, numpy.ndarray)):
            datasets_dict.update({key: value})

        elif isinstance(value, list):
            datasets_dict.update({key: numpy.array(value)})

        elif isinstance(value, dict):
            datasets_dict, metadata_dict = flatten_hierarchical_object_recursively(value, key, datasets_dict,
                                                                                   metadata_dict)

        else:
            if isinstance(value, (float, int, long, complex, str)):
                metadata_dict.update({key: value})

    return datasets_dict, metadata_dict


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


