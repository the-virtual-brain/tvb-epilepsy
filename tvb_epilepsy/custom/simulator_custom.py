"""
Python Demo for configuring Custom Simulations from Python.

Classes Settings, EpileptorParams and FullConfiguration are synchronized with the Java code, and should not be changed!
"""

import os
import json
from copy import copy

import subprocess

from tvb_epilepsy.base.constants import LIB_PATH, HDF5_LIB, JAR_PATH, JAVA_MAIN_SIM
from tvb_epilepsy.base.utils import obj_to_dict
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings
from tvb_epilepsy.base.calculations import calc_rescaled_x0

class Settings(object):
    def __init__(self, integration_step=0.05, noise_seed=42, noise_intensity=0.0001, length=10000):
        self.integration_step = integration_step
        self.integration_noise_seed = noise_seed
        self.noise_intensity = noise_intensity

        self.chunk_length = 2048
        self.downsampling_period = 10
        self.simulated_period = length


class EpileptorParams(object):

    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=1.5, tau=10.0, iext=3.1,
                 iext2=0.45, slope=0.0, x0=-2.1, tt=1.0):
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
    def __init__(self, x0_hyp, name="FromPython", connectivity_path="Connectivity.h5", no_regions=88,
                 settings=Settings()):
        self.configurationName = name
        self.connectivityPath = connectivity_path
        self.settings = settings
        self.variantName = None
        self.epileptorParamses = [EpileptorParams() for ii in xrange(no_regions)]

    def set(self, at_indices, ep_param):
        for i in at_indices:
            self.epileptorParamses[i] = copy(ep_param)


class SimulatorCustom(ABCSimulator):
    """
    From a VEP Hypothesis, write a custom JSON simulation configuration.
    To run a simulation, we can also open a GUI and import the resulted JSON file.
    """

    def __init__(self, head_path):
        self.head_path = head_path
        # TODO: do we need to do something different with the Connectivity (e.g. write it from VEP format) ?
        self.connectivity_path = os.path.join(self.head_path, "Connectivity.h5")

    @staticmethod
    def _save_serialized(ep_full_config, result_path):
        json_text = json.dumps(obj_to_dict(ep_full_config), indent=2)
        result_file = open(result_path, 'w')
        result_file.write(json_text)
        result_file.close()

    def rescale_x0(self, x0_hyp):
        # return r * x0_hyp - x0cr - 5.0 / 3.0
        # rescale_x0(x0_2d, yc, Iext1, a=1.0, b=-2.0, zmode=array("lin"), shape=None)
        return calc_rescaled_x0(x0_hyp, self.c, self.Iext, self.a, self.b-self.d)

    def config_simulation(self, hypothesis, head, vep_settings=SimulationSettings()):
        ep_settings = Settings(vep_settings.integration_step, vep_settings.noise_seed,
                               vep_settings.noise_intensity, vep_settings.simulated_period)

        self.ep_config = FullConfiguration(hypothesis.name, self.connectivity_path,
                                           head.number_of_regions, ep_settings)
        for i in xrange(head.number_of_regions):
            x0 = self.rescale_x0(hypothesis.x0.flatten())
            self.ep_config.set([i], EpileptorParams(iext=hypothesis.Iext1.flatten()[i],
                                                    x0=x0[i],
                                                    ks=hypothesis.K.flatten()[i],
                                                    c=hypothesis.yc.flatten()[i]))

        hypothesis_path = os.path.join(self.head_path, hypothesis.name + ".json")
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
