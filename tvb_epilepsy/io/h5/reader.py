from abc import abstractmethod


class ABCH5Reader(object):
    @abstractmethod
    def read_connectivity(self, path):
        pass

    @abstractmethod
    def read_surface(self, path):
        pass

    @abstractmethod
    def read_sensors(self, path):
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
    def read_head(self, path):
        pass

    @abstractmethod
    def read_hypothesis(self, path):
        pass

    @abstractmethod
    def read_model_configuration(self, path):
        pass

    @abstractmethod
    def read_parameter(self, path):
        pass

    @abstractmethod
    def read_lsa_service(self, path):
        pass

    @abstractmethod
    def read_model_configuration_service(self, path):
        pass

    @abstractmethod
    def read_head(self, path):
        pass

    @abstractmethod
    def read_stan_model_data_path(self, path):
        pass

    @abstractmethod
    def read_pse_params(self, path):
        pass

    @abstractmethod
    def read_generic(self, path):
        pass
