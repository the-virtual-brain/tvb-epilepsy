"""
Abstract Reading mechanism for VEP related entities
"""

from abc import ABCMeta, abstractmethod


class ABCReader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read_connectivity(self, path):
        pass

    @abstractmethod
    def read_cortical_surface(self, path):
        pass

    @abstractmethod
    def read_region_mapping(self, path):
        pass

    @abstractmethod
    def read_volume_mapping(self, path):
        pass

    @abstractmethod
    def read_t1(self, path):
        pass

    @abstractmethod
    def read_sensors(self, path, s_type):
        pass

    @abstractmethod
    def read_gain_matrix(self, path, s_type):
        pass

    @abstractmethod
    def read_head(self, root_folder):
        pass
