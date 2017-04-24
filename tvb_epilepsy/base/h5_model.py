import numpy


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


def prepare_for_h5(obj):
    datasets_dict = {}

    metadata_dict = {}

    for key, value in vars(obj).iteritems():
        if (isinstance(value, numpy.ndarray)):
            datasets_dict.update({key: value})
        else:
            if isinstance(value, (float, int, long, complex, str)):
                metadata_dict.update({key: value})

    h5_model = H5Model(datasets_dict, metadata_dict)

    return h5_model
