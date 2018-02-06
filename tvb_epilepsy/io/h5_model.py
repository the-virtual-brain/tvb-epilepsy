import os
from collections import OrderedDict
# from copy import deepcopy
# import inspect

import h5py

import numpy as np

from tvb_epilepsy.base.types import OrderedDictDot
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import sort_dict, iterable_to_dict, dict_to_list_or_tuple, \
    set_list_item_by_reference_safely, get_list_or_tuple_item_safely, isequal_string
from tvb_epilepsy.base.utils.file_utils import change_filename_or_overwrite


bool_inf_nan_none_empty = OrderedDict()
bool_inf_nan_none_empty.update({"True": True})
bool_inf_nan_none_empty.update({"False": False})
bool_inf_nan_none_empty.update({"inf": np.inf})
bool_inf_nan_none_empty.update({"nan": np.nan})
bool_inf_nan_none_empty.update({"None": None})
bool_inf_nan_none_empty.update({"[]": list()})
bool_inf_nan_none_empty.update({"()": tuple()})
bool_inf_nan_none_empty.update({"{}": OrderedDict()})


#TODO: the generic read/write methods should be adjusted to the new models
#TODO: also, the purpose of writting in h5 is inspection, so we should write only in the format that can be visualized
#TODO: the h5_model should not be necessary after the changes
#Observation: we should solve these TODOs after the next refactoring tasks are over, and the models are simpler
class H5Model(object):

    def __init__(self, datasets_dict, metadata_dict, logger=None):
        if logger is None:
            self.logger = initialize_logger(__name__)
        else:
            self.logger = logger
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
        final_path = change_filename_or_overwrite(os.path.join(folder_name, file_name))
        # final_path = ensure_unique_file(folder_name, file_name)
        self.logger.info("Writing %s at: %s" % (self, final_path))
        h5_file = h5py.File(final_path, 'a', libver='latest')
        for attribute, field in self.datasets_dict.iteritems():
            h5_file.create_dataset(attribute, data=field)
        for meta, val in self.metadata_dict.iteritems():
            dataset_path, attribute_name = os.path.split(meta)
            if dataset_path == "":
                h5_file.attrs.create(attribute_name, val)
            else:
                try:
                    h5_file[dataset_path].attrs.create(attribute_name, val)
                except:
                    print("WTF")
        h5_file.close()

    def convert_from_h5_model(self, obj=None, output_shape=None):
        output_type = obj.__class__.__name__
        if isinstance(obj, dict):
            obj = sort_dict(obj)
        elif np.in1d(output_type, ["tuple", "list"]):
            obj = iterable_to_dict(obj)
        elif isequal_string(output_type, "numpy.ndarray"):
            if isequal_string(obj.dtype, "numpy.ndarray"):
                obj = iterable_to_dict(obj.tolist())
        else:
            obj, output_type = create_object("/", self.metadata_dict)[:2]
        if obj is None:
            obj = OrderedDict()
        if output_type is None:
            output_type = obj.__class__.__name__
        for abs_path in self.datasets_dict.keys():
            child_obj = self.datasets_dict.pop(abs_path)
            rel_path = abs_path.split("/", 1)[1]
            build_hierarchical_object_recursively(obj, rel_path, child_obj, "/", abs_path, self.metadata_dict)
        if np.in1d(output_type, ["tuple", "list"]):
            obj = dict_to_list_or_tuple(obj, output_type)
        elif isequal_string(output_type, "numpy.ndarray"):
            obj = np.array(dict.values())
            if isinstance(output_shape, tuple):
                try:
                    obj = np.reshape(obj, output_shape)
                except:
                    warning("Failed to reshape read object to target shape " + str(output_shape) + "!" +
                            "\nReturning array of shape " + str(obj.shape) + "!")
        else:
            obj = update_object(obj, "/", self.metadata_dict, getORpop="pop")[0]
        if isinstance(obj, dict) and output_type.lower().find("dict") < 0:
            return OrderedDictDot(obj)
        else:
            return obj


def dict_to_h5_model(h5_model, obj, path, container_path):
    if len(obj) == 0:
        h5_model.add_or_update_datasets_attribute(path, "{}")
        return h5_model, None
    else:
        h5_model.add_or_update_metadata_attribute(os.path.join(container_path, "create_str"), "OrderedDict()")
        return h5_model, sort_dict(obj)


def list_or_tuple_to_h5_model(h5_model, obj, path, container_path, obj_type):
    # empty list or tuple get into metadata
    if len(obj) == 0:
        h5_model.add_or_update_metadata_attribute(path + "/type_str", obj_type)
        if isinstance(obj, list):
            h5_model.add_or_update_datasets_attribute(path, "[]")
        else:
            h5_model.add_or_update_datasets_attribute(path, "()")
        return h5_model, None
    # Try to store it as a ndarray of numbers or strings but not objects...
    temp = np.array(obj)
    if not(isequal_string(str(temp.dtype)[0], "O")):
        h5_model.add_or_update_metadata_attribute(path + "/type_str", obj.__class__.__name__)
        if isinstance(obj, tuple):
            h5_model.add_or_update_metadata_attribute(path + "/transform_str", "tuple(obj)")
        else:
            h5_model.add_or_update_metadata_attribute(path + "/transform_str", "obj.tolist()")
        h5_model.add_or_update_datasets_attribute(path, temp)
        return h5_model, None
    else:
        h5_model.add_or_update_metadata_attribute(os.path.join(container_path[1:], "create_str"), "list()")
        if isinstance(obj, tuple):
            h5_model.add_or_update_metadata_attribute(os.path.join(container_path[1:], "transform_str"), "tuple(obj))")
        return h5_model, iterable_to_dict(obj)


def array_to_h5_model(h5_model, obj, path, container_path, obj_type):
    if isequal_string(str(obj.dtype)[0], "O"):
        h5_model.add_or_update_metadata_attribute(os.path.join(container_path, "create_str"), "list()")
        h5_model.add_or_update_metadata_attribute(os.path.join(container_path, "transform_str"),
                                                  "np.reshape(obj, " + str(obj.shape) + ")")
        return h5_model, iterable_to_dict(obj)
    else:
        h5_model.add_or_update_metadata_attribute(path + "/type_str", obj_type)
        h5_model.add_or_update_metadata_attribute(path + "/transform_str",
                                                  "np.reshape(obj, " + str(obj.shape) + ")")
        h5_model.add_or_update_datasets_attribute(path, obj)
        return h5_model, None


def object_to_h5_model(h5_model, obj, container_path):
    obj_dict = sort_dict(vars(obj))
    for this_gen_str in ["context_str", "create_str", "transform_str", "update_str"]:
        gen_str = obj_dict.get(this_gen_str, None)
        if isinstance(gen_str, basestring):
            h5_model.add_or_update_metadata_attribute(os.path.join(container_path[1:], this_gen_str), gen_str)
    return h5_model, obj_dict


def object_to_h5_model_recursively(h5_model, obj, path="/"):
    # Use in some cases the name of the class as key, when name is empty string. Otherwise, class_name = name.
    obj_type = obj.__class__.__name__
    container_path = path
    if path == "/":
        path += obj_type
    if callable(obj):
        h5_model.add_or_update_metadata_attribute(path, str(obj))
        return
    if isinstance(obj, dict):
        h5_model, obj_dict = dict_to_h5_model(h5_model, obj, path, container_path)
        if obj_dict is None:
            return
    if isinstance(obj, (list, tuple)):
        h5_model, obj_dict = list_or_tuple_to_h5_model(h5_model, obj, path, container_path, obj_type)
        if obj_dict is None:
            return
    if isinstance(obj, np.ndarray):
        h5_model, obj_dict = array_to_h5_model(h5_model, obj, path, container_path, obj_type)
        if obj_dict is None:
            return
    if obj is None:
        h5_model.add_or_update_datasets_attribute(path, "None")
        return
    # This has to be before checking for float, to avoid np.inf and np.nan getting in the h5 model
    for key, val in bool_inf_nan_none_empty.iteritems():
        # Bool, inf, nan, or empty list/tuple/dict/str
        try:
            if obj is val:
                h5_model.add_or_update_metadata_attribute(path + "/type_str", obj_type)
                h5_model.add_or_update_datasets_attribute(path, key)
                return
        except:
            continue
    if isinstance(obj, (float, np.float, np.float16, np.float32, np.float64,
                        int, long, np.int, np.int8, np.int16, np.int32, np.int64, np.long,
                        complex, basestring)):
        h5_model.add_or_update_metadata_attribute(path + "/type_str", obj_type)
        h5_model.add_or_update_datasets_attribute(path, obj)
        return
    if not('obj_dict' in locals()):
        try:
            h5_model, obj_dict = object_to_h5_model(h5_model, obj, container_path)
        except:
            warning("Object " + path[1:] + " cannot be assigned to h5_model because it has no __dict__ property!")
            return
    h5_model.add_or_update_metadata_attribute(os.path.join(container_path[1:], "type_str"), obj_type)
    for key, this_obj in obj_dict.iteritems():
        this_path = os.path.join(container_path, key)
        # Ignore logger!
        if key.find("logger") < 0 and key.find("LOG") < 0: # and not(inspect.ismethod(getattr(obj, key))):
            # call recursively...
            object_to_h5_model_recursively(h5_model, this_obj, this_path)


def convert_to_h5_model(obj):
    h5_model = H5Model(OrderedDict(), OrderedDict())
    object_to_h5_model_recursively(h5_model, obj)
    h5_model.datasets_dict = sort_dict(h5_model.datasets_dict)
    h5_model.metadata_dict = sort_dict(h5_model.metadata_dict)
    return h5_model


def getORpop_object_strings(key, metadata, getORpop="get",
                            object_strings= ["type_str", "context_str", "create_str", "transform_str", "update_str"],
                            default_strings={}):
    defaults = {"type_str": "unknown", "context_str": "", "create_str": "", "transform_str": "", "update_str": ""}
    defaults.update(default_strings)
    output = []
    for s in object_strings:
        output.append(getattr(metadata, getORpop)(os.path.join(key, s), defaults.get(s, None)))
    return tuple(output)


def create_object(key, metadata, getORpop="get"):
    [type_str, context_str, create_str] = getORpop_object_strings(key, metadata, getORpop,
                                                                              ["type_str", "context_str", "create_str"])
    if len(context_str) > 0:
        exec context_str in globals(), locals()
    if len(create_str) > 0:
        obj = eval(create_str)
    else:
        obj = None
    return obj, type_str, context_str, create_str


def update_object(obj, key, metadata, getORpop="get"):
    [context_str, transform_str, update_str] = getORpop_object_strings(key, metadata, getORpop,
                                                                       ["context_str", "transform_str", "update_str"])
    if len(context_str) > 0:
        exec context_str in globals(), locals()
    if len(transform_str) > 0:
        obj = eval(transform_str)
    if len(update_str) > 0:
        exec update_str in globals(), locals()
    return obj, context_str, transform_str, update_str


def check_for_last_granchild(child_path, metadata, object_strings=
                                              ["type_str", "context_str", "create_str", "transform_str", "update_str"]):
    last_grandchild = True
    for key in metadata.keys():
        if key.find(child_path) >= 0:
            if key.split(child_path + "/", 1)[-1] not in object_strings:
                last_grandchild = False
                break
    return last_grandchild


def assert_obj(obj, obj_name, obj_type):
    if (isequal_string(obj_type, "list") or isequal_string(obj_type, "tuple")) and not (isinstance(obj, list)):
        return []
    if (isequal_string(obj_type, "dict") or isequal_string(obj_type, "OrderedDict")) and not (isinstance(obj, dict)):
        return OrderedDict()
    # If still not created, make an  OrderedDict() by default:
    if obj is None:
        warning("\n Child object " + str(obj_name) + " still not created!" +
                       "\nCreating an OrderedDict() by default!")
        return OrderedDict()
    return obj


def recurse_object(parent, obj, child_name, parent_path, rel_path, abs_path, metadata, get_field, set_field):
    try:
        # check if it already exists inside parent:
        child_obj = get_field(parent, child_name)
        child_path = os.path.join(parent_path, child_name)
        child_type = metadata.get(child_path + "/type_str", "unknown")
        if child_obj is None:
            child_obj = create_object(child_path, metadata)[0]
        child_obj = assert_obj(child_obj, child_name, child_type)
        # and set it to parent:
        set_field(parent, child_name, child_obj)
        # ...and continue to further specify it...
        build_hierarchical_object_recursively(child_obj, rel_path.split('/', 1)[1], obj, child_path, abs_path,
                                              metadata)
        # Check in remaining metadata, if this was the last grandchild of parent from this child,
        # in order to transform and/or update the child object, if required
        last_grandchild = check_for_last_granchild(child_path, metadata)
        if last_grandchild:
            child_obj = update_object(child_obj, child_path, metadata, getORpop="pop")[0]
            set_field(parent, child_name, child_obj)
            # Remove any object strings left...
            getORpop_object_strings(child_path, metadata, getORpop="pop")
    except:
        warning("Failed to set attribute " + str(abs_path) + " of object " + obj.__class__.__name__ + "!")


def set_object(parent, obj, child_name, abs_path, metadata, set_field):
    try:
        obj = update_object(obj, abs_path, metadata, getORpop="pop")[0]
        # Get rid of these keys if they still exist:
        # Remove any object strings left...
        getORpop_object_strings(abs_path, metadata, getORpop="pop")
        # Check whether obj is an inf, nan, None, bool, or empty list/dict/tuple/str value
        try:
            bool_inf_nan_empty_value = bool_inf_nan_none_empty.get(obj, "skip_bool_inf_nan_empty_value")
            if bool_inf_nan_empty_value != "skip_bool_inf_nan_empty_value":
                set_field(parent, child_name, bool_inf_nan_empty_value)
                return
        except:
            pass
        set_field(parent, child_name, obj)
        return
    except:
        warning("Failed to set attribute " + str(abs_path) + " of object " + obj.__class__.__name__ + "!")


def build_hierarchical_object_recursively(parent, rel_path, obj, parent_path, abs_path, metadata):
    if isinstance(parent, dict):
        set_field = lambda obj, key, value: obj.update({key: value})
        get_field = lambda obj, key: obj.get(key, None)
    elif isinstance(parent, list):
        set_field = lambda obj, key, value: set_list_item_by_reference_safely(int(key), value, obj)
        get_field = lambda obj, key: get_list_or_tuple_item_safely(obj, key)
    else:
        set_field = lambda obj, attribute, value: setattr(obj, attribute, value)
        get_field = lambda obj, attribute: getattr(obj, attribute, None)
    child_name = rel_path.split('/', 1)[0]
    if child_name != rel_path:
        # There is still tree below, this is a container object
        recurse_object(parent, obj, child_name, parent_path, rel_path, abs_path, metadata, get_field, set_field)
    else:
        # there is no tree below, obj is not a container, (transform it and) set it to parent
        set_object(parent, obj, child_name, abs_path, metadata, set_field)


def read_h5_model(path):
    h5_file = h5py.File(path, 'r', libver='latest')
    datasets_dict = dict()
    metadata_dict = dict()
    for attr_key, value in h5_file.attrs.iteritems():
        metadata_dict.update({"/" + attr_key: value})

    def add_dset_and_attr(name, obj):
        if isinstance(obj, h5py.Dataset):
            # node is a dataset
            datasets_dict.update({"/" + name: obj[()]})
        for key, val in obj.attrs.iteritems():
            metadata_dict.update({os.path.join("/" + name, key): val})

    h5_file.visititems(add_dset_and_attr)
    datasets_dict = sort_dict(datasets_dict)
    metadata_dict = sort_dict(metadata_dict)
    return H5Model(datasets_dict, metadata_dict)
