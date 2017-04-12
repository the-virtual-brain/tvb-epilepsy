"""
Python Demo for configuring Custom Simulations from Python.

Classes Settings, EpileptorParams and FullConfiguration are synchronized with the Java code, and should not be changed!
"""

import os
import json
from copy import copy

import subprocess

import numpy

from tvb_epilepsy.base.constants import LIB_PATH, HDF5_LIB, JAR_PATH, JAVA_MAIN_SIM
from tvb_epilepsy.base.utils import obj_to_dict, assert_arrays
from tvb_epilepsy.base.calculations import calc_rescaled_x0
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings


class Settings(object):
    def __init__(self, integration_step=0.01220703125, noise_seed=42, noise_intensity=10 ** -6, simulated_period=5000,
                 downsampling_period=0.9765625):
        self.integration_step = integration_step
        self.integration_noise_seed = noise_seed
        self.noise_intensity = noise_intensity

        self.chunk_length = 2048
        self.downsampling_period = numpy.round(downsampling_period/integration_step)
        self.simulated_period = simulated_period


class EpileptorModel(object):

    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=1.5, tau=10.0, iext=3.1,
                 iext2=0.45, slope=0.0, x0=-2.1, tt=1.0):
        a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2 , slope, x0, tt = \
            assert_arrays([a, b, c, d, aa, r, kvf, kf, ks, tau, iext, iext2 , slope, x0, tt])
        # TODO: add desired shape as argument in assert_arrays
        self._ui_name = "CustomEpileptor"
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
    def __init__(self, name="FromPython", connectivity_path="Connectivity.h5", no_regions=88, model=None,
                 settings=Settings()):
        self.configurationName = name
        self.connectivityPath = connectivity_path
        self.settings = settings
        self.variantName = None
        if model is None:
            self.epileptorParamses = EpileptorModel()
        else:
            self.epileptorParamses = model

    def set(self, at_indices, ep_param):
        for i in at_indices:
            self.epileptorParamses[i] = copy(ep_param)


class SimulatorCustom(ABCSimulator):
    """
    From a VEP Hypothesis, write a custom JSON simulation configuration.
    To run a simulation, we can also open a GUI and import the resulted JSON file.
    """

    def __init__(self, model, path):
        self.path = path
        # TODO: do we need to do something different with the Connectivity (e.g. write it from VEP format) ?
        self.model = model

    @staticmethod
    def _save_serialized(ep_full_config, result_path):
        json_text = json.dumps(obj_to_dict(ep_full_config), indent=2)
        result_file = open(result_path, 'w')
        result_file.write(json_text)
        result_file.close()

    def config_simulation(self, hypothesis, head_connectivity, vep_settings=SimulationSettings()):

        ep_settings = Settings(vep_settings.integration_step, vep_settings.noise_seed,
                               vep_settings.noise_intensity, vep_settings.simulated_period,
                               vep_settings.monitor_sampling_period)

        self.ep_config = FullConfiguration(hypothesis.name, head_connectivity,
                                           hypothesis.n_regions, self.model, ep_settings)

        # TODO: history length has to be computed given the time delays (i.e., the tract lengts...)
        #history_length = ...
        initial_conditions = self.prepare_initial_conditions(hypothesis, history_length=1)
        # TODO: initial_conditions have to be written in the json configuration file
        hypothesis_path = os.path.join(self.path, hypothesis.name + ".json")
        self._save_serialized(self.ep_config, hypothesis_path)

    def launch_simulation(self, hypothesis):
        hypothesis_file = os.path.join(self.head_path, hypothesis.name + ".json")
        opts = "java -Dncsa.hdf.hdf5lib.H5.hdf5lib=" + os.path.join(LIB_PATH, HDF5_LIB) + " " + \
               "-Djava.library.path=" + LIB_PATH + " " + "-cp" + " " + JAR_PATH + " " + \
               JAVA_MAIN_SIM + " " + hypothesis_file + " " + self.head_path

        x = subprocess.call(opts, shell=True)
        print x
        return None, None

    def launch_pse(self, hypothesis, head, vep_settings=SimulationSettings()):
        raise NotImplementedError()


# Some helper functions for model and simulator construction

def custom_model_builder(hypothesis, a=1.0, b=3.0, d=5.0):
    x0 = calc_rescaled_x0(hypothesis.x0.flatten(), hypothesis.yc.flatten(), hypothesis.Iext1.flatten(), a, b - d)
    model = EpileptorModel(x0=x0, iext=hypothesis.Iext1.flatten(), ks=hypothesis.K.flatten(), c=hypothesis.yc.flatten(),
                           a=a, b=b, d=d)
    return model


def setup_simulation(head_path, hypothesis, dt, sim_length, monitor_period, scale_time=1,
                     noise_intensity=None, variables_names=None):

    model = custom_model_builder(hypothesis)

    simulator_instance = SimulatorCustom(model, head_path)

    if variables_names is None:
        variables_names = [ 'x1', 'z', 'x2']

    if noise_intensity is None:
        noise_intensity = numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                  scale_time=scale_time,
                                  noise_intensity=noise_intensity,
                                  monitor_sampling_period=monitor_period,
                                  variables_names=variables_names)

    return simulator_instance, settings, variables_names, model