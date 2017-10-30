# File writing/reading and manipulations
import os
from datetime import datetime

import h5py
import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error


def ensure_unique_file(parent_folder, filename):
    final_path = os.path.join(parent_folder, filename)
    while os.path.exists(final_path):
        filename = raw_input("\n\nFile %s already exists. Enter a different name: " % final_path)
        final_path = os.path.join(parent_folder, filename)
    return final_path


def change_filename_or_overwrite(parent_folder, original_filename):
    final_path = os.path.join(parent_folder, original_filename)
    overwrite = False
    while os.path.exists(final_path) and not(overwrite):
        filename = raw_input("\n\nFile %s already exists. Enter a different name or press enter to overwrite file: "
                             % final_path)
        if filename == "":
            overwrite = True
            filename = original_filename
        final_path = os.path.join(parent_folder, filename)
    return final_path, overwrite


def print_metadata(h5_file, logger):
    logger.info("\n\nMetadata:")
    for key, val in h5_file["/"].attrs.iteritems():
        logger.info("\t" + str(key) + ", " + str(val))


def write_metadata(meta_dict, h5_file, key_date, key_version, path="/"):
    root = h5_file[path].attrs
    root[key_date] = str(datetime.now())
    root[key_version] = 2
    for key, val in meta_dict.iteritems():
        root[key] = val


# Depreciated since prepare_h5_model and write_h5_model can handle dictionaries (even recursively)...
def write_object_to_h5_file(obj, h5_file, attributes_dict=None,  add_overwrite_fields_dict=None, keys=None,
                            logger=None):
    if isinstance(h5_file, basestring):
        logger.info("\nWriting to: " + h5_file)
        h5_file = h5py.File(h5_file, 'a', libver='latest')
        if isinstance(keys, dict):
            write_metadata(keys, h5_file, keys["date"], keys["version"], path="/")
    if isinstance(obj, dict):
        get_field = lambda obj, key: obj[key]
        if not(isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj.keys():
                attributes_dict.update({key: key})
    else:
        get_field = lambda obj, attribute: getattr(obj, attribute)
        if not(isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj.__dict__.keys():
                attributes_dict.update({key: key})

    for attribute in attributes_dict:
        field = get_field(obj, attributes_dict[attribute])
        try:
            logger.info("\nWriting " + attributes_dict[attribute] + "...")
            if isinstance(field, basestring):
                logger.info("\nString length: " + str(len(field)))
                h5_file.create_dataset("/" + attribute, data=field)
                logger.info("\nString written length: " + str(len(h5_file['/' + attribute][()])))
            elif isinstance(field, np.ndarray):
                logger.info("\nNumpy array shape: " + str(field.shape))
                #TODO: deal with arrays of more than 2 dimensions
                if len(field.shape) > 2:
                    field = field.squeeze()
                    if len(field.shape) > 2:
                        field = field.flatten()
                h5_file.create_dataset("/" + attribute, data=field)
                logger.info("\nNumpy array written shape: " + str(h5_file['/' + attribute][()].shape))
            else:
                #try to write a scalar value
                try:
                    logger.info("\nWriting scalar value...")
                    h5_file.create_dataset("/" + attribute, data=field)
                except:
                    raise_value_error("\n\nValueError: Failed to write " + attribute + " as a scalar value!", logger)
        except:
            raise_value_error("ValueError: " + attribute + " not found in the object!", logger)
        #logger.info("dataset " + attribute +"value " + str(h5_file['/' + attribute][()]))

    if isinstance(add_overwrite_fields_dict, dict):
        for attribute in add_overwrite_fields_dict:
            logger.info("\nAdding or overwritting " + attribute + "... ")
            field = add_overwrite_fields_dict[attribute][0]
            mode = add_overwrite_fields_dict[attribute][1]
            if isinstance(field, basestring):
                logger.info("\nString length: " + str(len(field)))
                if mode == "overwrite":
                    del h5_file["/" + attribute]
                h5_file.create_dataset("/" + attribute, data=field)
                logger.info("\nString written length: " + str(len(h5_file['/' + attribute][()])))
            elif isinstance(field, np.ndarray):
                logger.info("\nNumpy array shape:" + str(field.shape))
                if mode == "overwrite":
                    del h5_file["/" + attribute]
                h5_file.create_dataset("/" + attribute, data=field)
                logger.info("\nlNumpy array written shape: " + str(h5_file['/' + attribute][()].shape))
            else:
                #try to write a scalar value
                try:
                    logger.info("\nWriting scalar value...")
                    if mode == "overwrite":
                        del h5_file["/" + attribute]
                    h5_file.create_dataset("/" + attribute, data=field)
                except:
                    raise_value_error("ValueError: Failed to write " + attribute + " as a scalar value!", logger)
            # logger.info("dataset " + attribute +"value " + str(h5_file['/' + attribute][()]))
    if isinstance(h5_file, basestring):
        h5_file.close()


# TODO: read dictionaries of dictionaries and objects of objects recursively
def read_object_from_h5_file(obj, h5_file, attributes_dict=None, add_overwrite_fields_dict=None, logger=None):
    if isinstance(h5_file, basestring):
        logger.info("\nReading from:" + h5_file)
        h5_file = h5py.File(h5_file, 'r', libver='latest')
        print_metadata(h5_file)
    if not(isinstance(attributes_dict, dict)):
        attributes_dict = dict()
        for key in h5_file.keys():
            attributes_dict.update({key: key})
    if isinstance(obj, dict):
        set_field = lambda obj, key, data: obj.update({key: data})
        get_field = lambda obj, key: obj[key]
    else:
        set_field = lambda obj, attribute, data: setattr(obj, attribute, data)
        get_field = lambda obj, attribute: getattr(obj, attribute)

    for attribute in attributes_dict:
        logger.info("\nReading " + attributes_dict[attribute] + "... ")
        try:
            set_field(obj, attributes_dict[attribute], h5_file['/' + attribute][()])
        except:
            raise_value_error("ValueError: Failed to read " + attribute + "!", logger)
        # logger.info("dataset " + attribute + "value " + str(get_field(obj, attributes_dict[attribute])))

    if isinstance(h5_file, basestring):
        h5_file.close()
    if isinstance(add_overwrite_fields_dict, dict):
        for attribute in add_overwrite_fields_dict:
            logger.info("\nSetting or overwritting " + attribute + "... ")
            try:
                set_field(obj, attribute, add_overwrite_fields_dict[attribute])
            except:
                raise_value_error("ValueError: Failed to set " + attribute + "!", logger)
            # logger.info("dataset " + attribute " value " + str(get_field(obj, attribute)))

    return obj