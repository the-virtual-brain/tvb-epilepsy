import os
import warnings

import h5py
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils import initialize_logger, ensure_unique_file, change_filename_or_overwrite, \
                                    set_list_item_by_reference_safely, get_list_or_tuple_item_safely, \
                                    list_or_tuple_to_dict, dict_to_list_or_tuple, sort_dict

logger = initialize_logger(__name__)

bool_inf_nan_empty = OrderedDict()
bool_inf_nan_empty.update({"True": True})
bool_inf_nan_empty.update({"False": False})
bool_inf_nan_empty.update({"inf": np.inf})
bool_inf_nan_empty.update({"nan": np.nan})
# bool_inf_nan_none_empty.update({"None": None})
bool_inf_nan_empty.update({"''": ""})

class_dict = {"list": list(), "tuple": tuple(), "dict": OrderedDict()}
class_list = ["list", "tuple", "dict",  "Connectivity", "Surface", "Sensors", "Head", "DiseaseHypothesis",
                 "ModelConfigurationService", "ModelConfiguration", "LSAService", "SamplingService", 
                 "DeterministicSamplingService", "StochasticSamplingService", "PSEService", 
                 "SensitivityAnalysisService"]

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

    def convert_from_h5_model(self, obj=dict(), children_dict=class_dict):

        children_dict.update(getattr(obj, "children_dict", {}))

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

            build_hierarchical_object_recursively(obj, key, value, children_dict)

        if np.in1d(output, ["tuple", "list"]):
            obj = dict_to_list_or_tuple(obj, output)

        return obj


def convert_to_h5_model(obj):
    h5_model = H5Model(OrderedDict(), OrderedDict())
    object_to_h5_model_recursively(h5_model, obj, "")
    return h5_model


def object_to_h5_model_recursively(h5_model, obj, name=""):

    # Use in some cases the name of the class as key, when name is empty string. Otherwise, class_name = name.
    name_empty = (len(name) == 0)
    class_name = name_empty * obj.__class__.__name__ + name

    if obj is None:
        h5_model.add_or_update_metadata_attribute(class_name, "None")



    if isinstance(obj, (float, int, long, complex, str, np.ndarray)):

        if isinstance(obj, (float, int, long, complex, str)):
            h5_model.add_or_update_metadata_attribute(class_name, obj)

        elif isinstance(obj, np.ndarray):
            h5_model.add_or_update_datasets_attribute(class_name, obj)

    else:

        for key, val in bool_inf_nan_empty.iteritems():
            # Bool, inf, nan, or empty list/tuple/dict/str
            try:
                if all(obj == val):
                    h5_model.add_or_update_metadata_attribute(class_name, key)
                    return
            except:
                continue

        # In any other case, make sure object is/becomes an alphabetically ordered dictionary:
        if not(isinstance(obj, dict)):

            try:

                for class_type in class_list:
                    if obj.__class__.__name__ == class_type:
                        name = name + ":" + obj.__class__.__name__
                        break

                if isinstance(obj, (list, tuple)):
                    try:
                        # empty list or tuple get into metadata
                        if len(obj) == 0:
                            h5_model.add_or_update_metadata_attribute(name, key)
                            return
                        temp = np.array(obj)
                        # those that can be converted to np arrays get in datasets
                        if temp.dtype != "O":
                            h5_model.add_or_update_datasets_attribute(name, temp)
                            return
                    except:
                        pass
                    # the rest are converted to dict
                    obj = list_or_tuple_to_dict(obj)

                elif isinstance(obj, dict):
                    obj = sort_dict(obj)

                else:
                    obj = sort_dict(vars(obj))

            except:
                logger.info("Object " + name + (len(name) > 0) * "/" + key + "cannot be assigned to h5_model because it"
                                                                             "is has no __dict__ property")
                return

        for key, value in obj.iteritems():

            key = name + (len(name) > 0) * "/" + key

            if key.find("children_dict") < 0 :
                # call recursively...
                object_to_h5_model_recursively(h5_model, value, key)


def build_hierarchical_object_recursively(obj, key, value, children_dict=class_dict):

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
        bool_inf_nan_empty_value = bool_inf_nan_empty.get(value, "skip_bool_inf_nan_empty_value")
        if bool_inf_nan_empty_value != "skip_bool_inf_nan_empty_value":
            set_field(obj, key, bool_inf_nan_empty_value)
            return
    except:
        pass

    child_object = get_field(obj, key)
    if child_object is not None:
        set_field(obj, key, value)

    else:

        this_name = key.split('/', 1)[0]
        split_name = this_name.split(':')
        if len(split_name) == 2:
            name = split_name[0]
            class_name = split_name[1]
        else:
            class_name = ""
            name = this_name

        try:

           # value is not a container object, or it is an empty container object
            if this_name == key:
                # just assign it:
                if np.in1d(class_name, ["tuple", "list"]) and isinstance(value, np.ndarray):
                    value = value.tolist()
                    if class_name == "tuple":
                        value = tuple(value)
                set_field(obj, name, value)
                return 1

            else:
                child_key = key.split('/', 1)[1]
                child_object = deepcopy(get_field(obj, name))
                # Check if it exists already:
                if child_object is None:
                    # and create it if not:
                    for class_type, class_instance in children_dict.iteritems():
                        if class_name == class_type:
                            child_object = class_instance
                    if isinstance(child_object, (list, tuple)):
                        # if it is a list or tuple...
                        grandchild_name = child_key.split('/', 1)[0]
                        # but its own children names are not strings of integers:
                        if not(grandchild_name.isdigit()):
                            # convert to a dict
                            child_object = list_or_tuple_to_dict(child_object)
                        # if it is a tuple...
                        if isinstance(child_object, tuple):
                            # ...convert to list that is mutable
                            child_object = list(child_object)
                # If still not created, make a dict() by default:
                if child_object is None:
                    logger.warning("\n Child object " + str(name) +
                                   " still not created! Creating an Ordereddict() by default!")
                    child_object = OrderedDict()
                # ...and continue to further specify it...
                children_dict.update(getattr(child_object, "children_dict", {}))
                build_hierarchical_object_recursively(child_object, child_key, value, children_dict)
                if class_name == "tuple":
                    child_object = tuple(child_object)
                set_field(obj, name, child_object)
        except:
            warnings.warn("Failed to set attribute " + str(key) + " of object " + obj.__class__.__name__ + "!")


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

    from copy import deepcopy

    from tvb_epilepsy.base.constants import FOLDER_RES, DATA_MODE, DATA_CUSTOM, TVB
    from tvb_epilepsy.base.utils import assert_equal_objects
    from tvb_epilepsy.base.model_vep import Connectivity
    from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis

    if DATA_MODE is TVB:
        from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
    else:
        from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

    # -------------------------------Reading data-----------------------------------

    empty_connectivity = Connectivity("", np.array([]), np.array([]))
    empty_hypothesis = DiseaseHypothesis(empty_connectivity)

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    # # Manual definition of hypothesis...:
    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]
    disease_values = x0_values + e_values
    disease_indices = x0_indices + e_indices

    # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
    hyp_x0_E = DiseaseHypothesis(head.connectivity, excitability_hypothesis={tuple(x0_indices): x0_values},
                                 epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                 connectivity_hypothesis={})

    obj = {"hyp_x0_E": hyp_x0_E,
            "test_dict":
                {"list0": ["l00", 1, {"d020": "a", "d021": [True, False, np.inf, np.nan, None, [], (), {}, ""]}]}}
    logger.info("\n\nOriginal object:\n" + str(obj))

    logger.info("\n\nWriting object to h5 file...")
    h5_model = convert_to_h5_model(obj)

    h5_model.write_to_h5(FOLDER_RES, "test_h5_model.h5")

    h5_model1 = read_h5_model(FOLDER_RES + "/test_h5_model.h5")
    obj1 = h5_model1.convert_from_h5_model(deepcopy(obj))

    if assert_equal_objects(obj, obj1):
        logger.info("\n\nRead identical object:\n" + str(obj1))
    else:
        logger.info("\n\nComparison failed!:\n" + str(obj1))

    h5_model2 = read_h5_model(FOLDER_RES + "/test_h5_model.h5")
    obj2 = h5_model2.convert_from_h5_model(children_dict={"DiseaseHypothesis": empty_hypothesis})

    if assert_equal_objects(obj, obj2):
        logger.info("\n\nRead object as dictionary:\n" + str(obj2))
    else:
        logger.info("\n\nComparison failed!:\n" + str(obj2))



