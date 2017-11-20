"""
Python Demo for configuring Custom Simulations from Python.

Classes Settings, EpileptorParams and FullConfiguration are synchronized with the Java code, and should not be changed!
"""

import json
import os
import subprocess
from copy import copy

import numpy

from tvb_epilepsy.base.constants.module_constants import LIB_PATH, HDF5_LIB, JAR_PATH, JAVA_MAIN_SIM
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import obj_to_dict, assert_arrays
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings
from tvb_epilepsy.base.computations.calculations_utils import calc_x0_val__to_model_x0


# TODO: It is imperative to allow for modification of the connectivity.normalized_weights of the Connecitivity.h5
# according to the model_configuration.connectivity


class SimulationSettings(object):

    def __init__(self, integration_step=0.01220703125, noise_seed=42, noise_intensity=10 ** -6, simulated_period=5000,
                 downsampling_period=0.9765625):
        self.integration_step = integration_step
        self.integration_noise_seed = noise_seed
        self.noise_intensity = noise_intensity
        self.chunk_length = 2048
        self.downsampling_period = numpy.round(downsampling_period / integration_step)
        self.simulated_period = simulated_period


class EpileptorParams(object):

    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=1.5, tau=10.0, iext=3.1,
                 iext2=0.45, slope=0.0, x0=-2.2, tt=1.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.aa = aa
        self.r = r
        self.Kvf = kvf
        self.Kf = kf
        self.Ks = ks
        self.tau = tau
        self.Iext = iext
        self.Iext2 = iext2
        self.slope = slope
        self.x0 = x0
        self.tt = tt


class EpileptorModel(object):

    _ui_name = "CustomEpileptor"
    _nvar = 2
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    aa = 6.0
    r = 0.00035
    Kvf = 0.0
    Kf = 0.0
    Ks = 1.0
    tau = 10.0
    Iext = 3.1
    Iext2 = 0.45
    slope = 0.0
    tt = 1.0

    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=1.0, tau=10.0, iext=3.1,
                 iext2=0.45, slope=0.0, x0=-2.1, tt=1.0):
        a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2, slope, x0, tt = \
            assert_arrays([a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2, slope, x0, tt])
        # TODO: add desired shape as argument in assert_arrays
        self.nvar = 6
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.aa = aa
        self.r = r
        self.Kvf = kvf
        self.Kf = kf
        self.Ks = ks
        self.tau = tau
        self.Iext = iext
        self.Iext2 = iext2
        self.slope = slope
        self.x0 = x0
        self.tt = tt


class FullConfiguration(object):

    def __init__(self, name="full-configuration", connectivity_path="Connectivity.h5", epileptor_paramses=[],
                 settings=SimulationSettings(), initial_states=None, initial_states_shape=None):
        self.configurationName = name
        self.connectivityPath = connectivity_path
        self.settings = settings
        self.variantName = None
        self.epileptorParamses = epileptor_paramses
        if initial_states is not None and initial_states_shape is not None:
            self.initialStates = initial_states
            self.initialStatesShape = initial_states_shape

    def set(self, at_indices, ep_param):
        for i in at_indices:
            self.epileptorParamses[i] = copy(ep_param)


class SimulatorCustom(ABCSimulator):
    """
    From a VEP Hypothesis, write a custom JSON simulation configuration.
    To run a simulation, we can also open a GUI and import the resulted JSON file.
    """

    json_custom_config_file = "SimulationConfiguration.json"

    def __init__(self, connectivity, model_configuration, model, simulation_settings):
        self.model = model
        self.simulation_settings = simulation_settings
        self.model_configuration = model_configuration
        self.connectivity = connectivity

    @staticmethod
    def _save_serialized(ep_full_config, result_path):
        json_text = json.dumps(obj_to_dict(ep_full_config), indent=2)
        result_file = open(result_path, 'w')
        result_file.write(json_text)
        result_file.close()

    def config_simulation(self):
        ep_settings = SimulationSettings(self.simulation_settings.integration_step, self.simulation_settings.noise_seed,
                                         self.simulation_settings.noise_intensity, self.simulation_settings.simulated_period,
                                         self.simulation_settings.monitor_sampling_period)
        json_model = self.prepare_epileptor_model_for_json(self.connectivity.number_of_regions)
        # TODO: history length has to be computed given the time delays (i.e., the tract lengts...)
        # history_length = ...
        initial_conditions = self.prepare_initial_conditions(history_length=1)
        self.custom_config = FullConfiguration(connectivity_path=os.path.abspath(self.connectivity.file_path),
                                               epileptor_paramses=json_model, settings=ep_settings,
                                               initial_states=initial_conditions.flatten(),
                                               initial_states_shape=numpy.array(initial_conditions.shape))
        self.head_path = os.path.dirname(self.connectivity.file_path)
        self.json_config_path = os.path.join(self.head_path, self.json_custom_config_file)
        self._save_serialized(self.custom_config, self.json_config_path)

    def launch_simulation(self, n_report_blocks=0):
        opts = "java -Dncsa.hdf.hdf5lib.H5.hdf5lib=" + os.path.join(LIB_PATH, HDF5_LIB) + " " + \
               "-Djava.library.path=" + LIB_PATH + " " + "-cp" + " " + JAR_PATH + " " + \
               JAVA_MAIN_SIM + " " + os.path.abspath(self.json_config_path) + " " + os.path.abspath(self.head_path)
        try:
            status = subprocess.call(opts, shell=True)
            print(status)
        except:
            status = False
            warning("Something went wrong with this simulation...")
        from tvb_epilepsy.custom.read_write import read_ts
        time, data = read_ts(os.path.join(self.head_path, "full-configuration", "ts.h5"), data="data")
        return time, data, status

    def prepare_epileptor_model_for_json(self, no_regions=88):
        epileptor_params_list = []
        warning("No of regions is " + str(no_regions))
        for idx in xrange(no_regions):
            epileptor_params_list.append(
                EpileptorParams(self.model.a[idx], self.model.b[idx], self.model.c[idx], self.model.d[idx],
                                self.model.aa[idx], self.model.r[idx], self.model.Kvf[idx], self.model.Kf[idx],
                                self.model.Ks[idx], self.model.tau[idx], self.model.Iext[idx], self.model.Iext2[idx],
                                self.model.slope[idx], self.model.x0[idx], self.model.tt[idx]))

        return epileptor_params_list

    def prepare_for_h5(self):
        settings_h5_model = convert_to_h5_model(self.simulation_settings)
        epileptor_model_h5_model = convert_to_h5_model(self.model)
        epileptor_model_h5_model.append(settings_h5_model)
        epileptor_model_h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        epileptor_model_h5_model.add_or_update_metadata_attribute("Monitor expressions",
                                                                  self.simulation_settings.monitor_expressions)
        epileptor_model_h5_model.add_or_update_metadata_attribute("Variables names",
                                                                  self.simulation_settings.variables_names)
        return epileptor_model_h5_model

    def configure_model(self, **kwargs):
        self.model = custom_model_builder(self.model_configuration, **kwargs)

    def configure_initial_conditions(self, initial_conditions=None):
        if isinstance(initial_conditions, numpy.ndarray):
            self.initial_conditions = initial_conditions
        else:
            # TODO: have a function to calculate the correct history length when we have time delays
            self.initial_conditions = self.prepare_initial_conditions(history_length=1)


# Some helper functions for model and simulator construction
def custom_model_builder(model_configuration, a=1.0, b=3.0, d=5.0):
    x0 = calc_x0_val__to_model_x0(model_configuration.x0_values, model_configuration.yc,
                                  model_configuration.Iext1, a, b - d)
    model = EpileptorModel(a=a, b=b, d=d, x0=x0, iext=model_configuration.Iext1,
                           ks=model_configuration.K,
                           c=model_configuration.yc)
    return model
