from abc import ABCMeta, abstractmethod

import numpy

from tvb_infer.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_equilibrium_point


class ABCSimulator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def launch_simulation(self, **kwargs):
        pass

    # @abstractmethod
    # def launch_pse(self, hypothesis, head):
    #     pass

    ###
    # Prepare for tvb-epilepsy epileptor_models initial conditions
    ###

    def prepare_initial_conditions(self, history_length=1):
        # Set default initial conditions right on the resting equilibrium point of the model...
        # ...after computing the equilibrium point (and correct it for zeql for a >=6D model
        initial_conditions = calc_equilibrium_point(self.model, self.model_configuration,
                                                    self.connectivity.normalized_weights)
        # -------------------The lines below are for a specific "realistic" demo simulation:---------------------------------
        if (self.model._nvar > 6):
            shape = initial_conditions[5].shape
            n_regions = max(shape)
            type = initial_conditions[5].dtype
            initial_conditions[6] = 0.0 ** numpy.ones(shape, dtype=type)  # hypothesis.x0_values.T
            initial_conditions[7] = 1.0 * numpy.ones(
                (1, n_regions))  # model.slope * numpy.ones((hypothesis.number_of_regions,1))
            initial_conditions[9] = 0.0 * numpy.ones(
                (1, n_regions))  # model.Iext2.T * numpy.ones((hypothesis.number_of_regions,1))
        # ------------------------------------------------------------------------------------------------------------------
        initial_conditions = numpy.expand_dims(initial_conditions, 2)
        initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
        return initial_conditions
