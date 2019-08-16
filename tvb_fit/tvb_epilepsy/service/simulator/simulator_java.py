"""
Wrapper for configuring Java Simulations from Python.

Classes Settings, EpileptorParams and FullConfiguration are synchronized with the Java code, and should not be changed!
"""

import os
import json
import numpy
import subprocess
from copy import copy

from tvb_fit.base.config import GenericConfig
from tvb_fit.tvb_epilepsy.base.constants.model_constants import TIME_DELAYS_FLAG
from tvb_fit.service.simulator import ABCSimulator
from tvb_fit.tvb_epilepsy.base.computation_utils.calculations_utils import calc_x0_val_to_model_x0
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import compute_initial_conditions_from_eq_point
from tvb_fit.tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, Timeseries

from tvb_utils.log_error_utils import initialize_logger
from tvb_utils.data_structures_utils import obj_to_dict, assert_arrays


class Settings(object):

    def __init__(self, integration_step=1000.0/16384.0, noise_seed=42, noise_intensity=10 ** -6, simulated_period=2000,
                 downsampling_period=1000.0/1024.0):
        self.simulated_period = simulated_period
        self.integration_step = integration_step

        self.integration_noise_seed = noise_seed
        self.noise_intensity = noise_intensity
        self.node_noise_dispersions = None  # Could also be a vector double[nNodes * nStateVars]; first by node then SV

        self.chunk_length = 2048
        self.downsampling_period = numpy.round(downsampling_period / integration_step)

    def set_node_noise_dispersions(self, noise):
        self.node_noise_dispersions = noise

    def set_voi_noise_dispersions(self, noise_voi, no_of_nodes):
        no_voi = len(noise_voi)
        self.node_noise_dispersions = numpy.zeros((no_of_nodes * no_voi,))
        for i in xrange(no_of_nodes):
            self.node_noise_dispersions[i * no_voi: (i + 1) * no_voi] = noise_voi


class EpileptorParams(object):

    # TODO: Figure out the correct sign for ks (negative for TVB, positive for tvb-epilepsy, for the JAVA simulator???)
    def __init__(self, a=1.0, b=3.0, c=1.0, d=5.0, aa=6.0, r=0.00035, kvf=0.0, kf=0.0, ks=-1.5, tau=10.0, iext=3.1,
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


class JavaEpileptor(object):
    _ui_name = "JavaEpileptor"
    _nvar = 6
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

    @property
    def vois(self):
        return ['x1', 'z', 'x2']


# ! The attributes names should not be changed because they are synchronized with simulator config model.
class FullConfiguration(object):

    def __init__(self, name="full-configuration", connectivity_path="Connectivity.h5", epileptor_params=None,
                 settings=Settings(), initial_states=None, initial_states_shape=None):
        self.configurationName = name
        self.connectivityPath = connectivity_path
        self.settings = settings
        self.variantName = None
        self.epileptorParamses = epileptor_params
        if initial_states is not None and initial_states_shape is not None:
            self.initialStates = initial_states
            self.initialStatesShape = initial_states_shape

    def set(self, at_indices, ep_param):
        for i in at_indices:
            self.epileptorParamses[i] = copy(ep_param)


class SimulatorJava(ABCSimulator):
    """
    From a VEP Hypothesis, write a custom JSON simulation configuration.
    To run a simulation, we can also open a GUI and import the resulted JSON file.
    """
    logger = initialize_logger(__name__)
    json_custom_config_file = "SimulationConfiguration.json"

    def __init__(self, model_configuration, connectivity, settings):
        super(SimulatorJava, self).__init__(model_configuration, connectivity, settings)
        self.model = None
        self.head_path = os.path.dirname(self.connectivity.file_path)
        self.connectivity = connectivity
        self.json_config_path = os.path.join(self.head_path, self.json_custom_config_file)
        self.configure_model()

    def get_vois(self):
        return self.model.vois

    @staticmethod
    def _save_serialized(ep_full_config, result_path):
        json_text = json.dumps(obj_to_dict(ep_full_config), indent=2)
        result_file = open(result_path, 'w')
        result_file.write(json_text)
        result_file.close()

    def config_simulation(self):

        ep_settings = Settings(integration_step=self.settings.integration_step,
                               noise_seed=self.settings.noise_seed,
                               simulated_period=self.settings.simulated_period,
                               downsampling_period=self.settings.monitor_sampling_period)
        if isinstance(self.settings.noise_intensity, (float, int)):
            self.logger.info("Using uniform noise %s" % self.settings.noise_intensity)
            ep_settings.noise_intensity = self.settings.noise_intensity
        elif len(self.settings.noise_intensity) == JavaEpileptor._nvar:
            self.logger.info("Using noise/voi %s" % self.settings.noise_intensity)
            ep_settings.set_voi_noise_dispersions(self.settings.noise_intensity,
                                                  self.connectivity.number_of_regions)
        elif len(self.settings.noise_intensity) == JavaEpileptor._nvar * self.connectivity.number_of_regions:
            self.logger.info("Using node noise %s" % self.settings.noise_intensity)
            ep_settings.set_node_noise_dispersions(self.settings.noise_intensity)
        else:
            self.logger.warning("Could not set noise %s" % self.settings.noise_intensity)

        json_model = self.prepare_epileptor_model_for_json(self.connectivity.number_of_regions)
        # TODO: history length has to be computed given the time delays (i.e., the tract lengths...)
        # TODO: when dfun is implemented for JavaEpileptor, we can use commented lines with initial_conditions
        # initial_conditions = self.prepare_initial_conditions(history_length=1)
        # custom_config = FullConfiguration(connectivity_path=os.path.abspath(self.connectivity.file_path),
        #                                        epileptor_params=json_model, settings=ep_settings,
        #                                        initial_states=initial_conditions.flatten(),
        #                                        initial_states_shape=numpy.array(initial_conditions.shape))
        custom_config = FullConfiguration(connectivity_path=os.path.abspath(self.connectivity.file_path),
                                          epileptor_params=json_model, settings=ep_settings,
                                          initial_states=None, initial_states_shape=None)
        self._save_serialized(custom_config, self.json_config_path)

    def launch_simulation(self):
        from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
        opts = "java -Dncsa.hdf.hdf5lib.H5.hdf5lib=" + os.path.join(GenericConfig.LIB_PATH, GenericConfig.HDF5_LIB) + \
               " " + "-Djava.library.path=" + GenericConfig.LIB_PATH + " " + "-cp" + " " + GenericConfig.JAR_PATH + \
               " " + GenericConfig.JAVA_MAIN_SIM + " " + os.path.abspath(self.json_config_path) + " " + \
               os.path.abspath(self.head_path)
        try:
            status = subprocess.call(opts, shell=True)
            print(status)
        except:
            status = False
            self.logger.warning("Something went wrong with this simulation...")
        time, data = H5Reader().read_ts(os.path.join(self.head_path, "full-configuration", "ts.h5"))
        return Timeseries(  # substitute with TimeSeriesRegion fot TVB like functionality
                          data, time=time, connectivity=self._vp2tvb_connectivity(TIME_DELAYS_FLAG),
                          labels_dimensions={TimeseriesDimensions.SPACE.value: self.connectivity.region_labels,
                                            TimeseriesDimensions.VARIABLES.value: self.get_vois()},ts_type="Region"), \
               status

    def prepare_epileptor_model_for_json(self, no_regions=88):
        epileptor_params_list = []
        self.logger.warning("No of regions is " + str(no_regions))
        for idx in range(no_regions):
            epileptor_params_list.append(
                EpileptorParams(self.model.a[idx], self.model.b[idx], self.model.c[idx], self.model.d[idx],
                                self.model.aa[idx], self.model.r[idx], self.model.Kvf[idx], self.model.Kf[idx],
                                self.model.Ks[idx], self.model.tau[idx], self.model.Iext[idx], self.model.Iext2[idx],
                                self.model.slope[idx], self.model.x0[idx], self.model.tt[idx]))

        return epileptor_params_list

    def configure_model(self):
        x0 = calc_x0_val_to_model_x0(self.model_configuration.x0_values, self.model_configuration.yc,
                                     self.model_configuration.Iext1, self.model_configuration.a,
                                     self.model_configuration.b - self.model_configuration.d)
    # TODO: Figure out the correct sign for ks (negative for TVB, positive for tvb-epilepsy, for the JAVA simulator???)
        self.model = JavaEpileptor(a=self.model_configuration.a, b=self.model_configuration.b,
                                   d=self.model_configuration.d, x0=x0, iext=self.model_configuration.Iext1,
                                   ks=-self.model_configuration.K, c=self.model_configuration.yc,
                                   tt=self.model_configuration.tau1, r=1.0/self.model_configuration.tau0)

    def configure_initial_conditions(self):
        initial_conditions = self.model_configuration.initial_conditions
        if isinstance(initial_conditions, numpy.ndarray):
            if len(initial_conditions.shape) < 4:
                initial_conditions = numpy.expand_dims(initial_conditions, 2)
                initial_conditions = numpy.tile(initial_conditions, (1, 1, 1, 1))
            self.initial_conditions = initial_conditions
        else:
            self.initial_conditions = compute_initial_conditions_from_eq_point(self.model_configuration,
                                                                               history_length=1,
                                                                               simulation_shape=True)
