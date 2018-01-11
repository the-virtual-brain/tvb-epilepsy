from abc import abstractmethod


class ABCH5Writer(object):

    @abstractmethod
    def write_connectivity(self, connectivity, path):
        pass

    @abstractmethod
    def write_sensors(self, sensors, path):
        pass

    @abstractmethod
    def write_surface(self, surface, path):
        pass

    @abstractmethod
    def write_head(self, head, path):
        pass

    @abstractmethod
    def write_hypothesis(self, hypothesis, path):
        pass

    @abstractmethod
    def write_model_configuration(self, model_configuration, path):
        pass

    @abstractmethod
    def write_model_configuration_service(self, model_configuration_service, path):
        pass
