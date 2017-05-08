"""
Python Demo for configuring Custom Simulations from Python.

Classes Settings, EpileptorParams and FullConfiguration are synchronized with the Java code, and should not be changed!
"""

import os
import json
from copy import copy
import subprocess
import warnings
import numpy
from tvb_epilepsy.base.constants import LIB_PATH, HDF5_LIB, JAR_PATH, JAVA_MAIN_SIM
from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import obj_to_dict, assert_arrays
from tvb_epilepsy.base.calculations import calc_rescaled_x0
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings
from tvb_epilepsy.custom.read_write import read_ts


class Settings(object):
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
    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=1.5, tau=10.0, iext=3.1,
                 iext2=0.45, slope=0.0, x0=-2.1, tt=1.0, n_regions=88):
        a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2, slope, x0, tt = \
            assert_arrays([a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2, slope, x0, tt])
        # TODO: add desired shape as argument in assert_arrays
        self._ui_name = "CustomEpileptor"
        self.nvar = 6
        self.n_regions = n_regions
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
        self.dfun = []


class FullConfiguration(object):
    def __init__(self, name="FromPython", connectivity_path="Connectivity.h5", epileptor_paramses=[],
                 settings=Settings(), initial_states=None, initial_states_shape=None):
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

    def __init__(self, hypothesis, model, head_path, json_config_file, head_connectivity_path="Connectivity.h5",
                 results_path=[]):
        self.hypothesis = hypothesis
        self.model = model
        self.head_path = head_path
        self.head_connectivity_path = head_connectivity_path  #os.path.join(self.head_path, head_connectivity_path)
        self.json_config_file = json_config_file
        if len(results_path) == 0:
            self.results_path = os.path.join(self.head_path, self.hypothesis.name, "ts.h5")
        else:
            self.results_path = results_path

    @staticmethod
    def _save_serialized(ep_full_config, result_path):
        json_text = json.dumps(obj_to_dict(ep_full_config), indent=2)
        result_file = open(result_path, 'w')
        result_file.write(json_text)
        result_file.close()

    def config_simulation(self, settings=SimulationSettings()):

        ep_settings = Settings(settings.integration_step, settings.noise_seed, settings.noise_intensity,
                               settings.simulated_period, settings.monitor_sampling_period)

        # TODO: history length has to be computed given the time delays (i.e., the tract lengts...)
        # history_length = ...
        initial_conditions = self.prepare_initial_conditions(self.hypothesis, history_length=1)

        self.ep_config = FullConfiguration(self.hypothesis.name, self.head_connectivity_path,
                                           self.prepare_epileptor_model_for_json(self.model.n_regions), ep_settings,
                                           initial_conditions.flatten(),
                                           numpy.array(initial_conditions.shape))

        self.json_config_path = os.path.join(self.head_path, self.json_config_file)
        self._save_serialized(self.ep_config, self.json_config_path)

        self.ep_settings = ep_settings

        return ep_settings

    def launch_simulation(self, return_output=False):

        hypothesis_file = os.path.join(self.head_path, self.json_config_file)
        opts = "java -Dncsa.hdf.hdf5lib.H5.hdf5lib=" + os.path.join(LIB_PATH, HDF5_LIB) + " " + \
               "-Djava.library.path=" + LIB_PATH + " " + "-cp" + " " + JAR_PATH + " " + \
               JAVA_MAIN_SIM + " " + hypothesis_file + " " + self.head_path

        try:
            status = subprocess.call(opts, shell=True)
            print status

        except:
            status = False

        if not(status):
            warnings.warn("Something went wrong with this simulation...")

        if return_output:
            time, data = read_ts(self.results_path, data="data")
            return time, data, status

        else:
            return None, None, status

    # def launch_pse(self, hypothesis, head, vep_settings=SimulationSettings()):
    #     raise NotImplementedError()

    def prepare_epileptor_model_for_json(self, no_regions=88):
        epileptor_params_list = []

        for idx in xrange(no_regions):
            epileptor_params_list.append(
                EpileptorParams(self.model.a[idx], self.model.b[idx], self.model.c[idx], self.model.d[idx],
                                self.model.aa[idx], self.model.r[idx], self.model.Kvf[idx], self.model.Kf[idx],
                                self.model.Ks[idx], self.model.tau[idx], self.model.Iext[idx], self.model.Iext2[idx],
                                self.model.slope[idx], self.model.x0[idx], self.model.tt[idx]))

        return epileptor_params_list

    def prepare_for_h5(self, settings, monitor_expressions, variables_names):
        settings_h5_model = prepare_for_h5(settings)
        epileptor_model_h5_model = prepare_for_h5(self.model)

        epileptor_model_h5_model.append(settings_h5_model)
        epileptor_model_h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")

        epileptor_model_h5_model.add_or_update_metadata_attribute("Monitor expressions", monitor_expressions)
        epileptor_model_h5_model.add_or_update_metadata_attribute("Variables names", variables_names)

        return epileptor_model_h5_model

    def configure_model(self, hypothesis=None, **kwargs):

        if hypothesis is None:
            hypothesis = self.hypothesis

        self.model = custom_model_builder(hypothesis, **kwargs)

    def conifigure_initial_conditions(self, initial_conditions=None):

        if isinstance(initial_conditions, numpy.ndarray):
            self.initial_conditions = initial_conditions

        else:
            # TODO: have a function to calculate the correct history lenght when we have time delays
            self.initial_conditions = self.prepare_initial_conditions(self.hypothesis, history_length=1)

# Some helper functions for model and simulator construction

def custom_model_builder(hypothesis, a=1.0, b=3.0, d=5.0):
    x0 = calc_rescaled_x0(hypothesis.x0.flatten(), hypothesis.yc.flatten(), hypothesis.Iext1.flatten(), a, b - d)
    model = EpileptorModel(a=a, b=b, d=d, x0=x0, iext=hypothesis.Iext1.flatten(), ks=hypothesis.K.flatten(),
                           c=hypothesis.yc.flatten(), n_regions=hypothesis.n_regions)

    return model


def setup_simulation(hypothesis, head_path, dt, sim_length, monitor_period, scale_time=1,
                     noise_intensity=None, variables_names=None):

    simulator_instance = SimulatorCustom(hypothesis, custom_model_builder(hypothesis), head_path,
                                         hypothesis.name + ".json")

    if variables_names is None:
        variables_names = ['x1', 'z', 'x2']

    if noise_intensity is None:
        noise_intensity = 0  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                  scale_time=scale_time,
                                  noise_intensity=noise_intensity,
                                  monitor_sampling_period=monitor_period,
                                  variables_names=variables_names)

    return simulator_instance, settings, variables_names
