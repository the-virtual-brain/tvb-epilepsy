"""
Mechanism for launching TVB simulations.
"""

import sys
import time
import warnings
import numpy
from tvb.datatypes import connectivity, equations
from tvb.simulator import coupling, integrators, monitors, noise, simulator
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings
from tvb_epilepsy.base.equations import calc_dfun, calc_coupling
from tvb_epilepsy.base.equilibrium_computation import calc_eq_y1, calc_eq_pop2, calc_eq_g, calc_equilibrium_point, \
                                                     calc_eq_z_6d
from tvb_epilepsy.tvb_api.epileptor_models import *


class SimulatorTVB(ABCSimulator):

    def __init__(self, model_instance, builder_initial_conditions):
        self.model = model_instance
        self.builder_initial_conditions = builder_initial_conditions

    @staticmethod
    def _vep2tvb_connectivity(vep_conn):
        return connectivity.Connectivity(use_storage=False, weights=vep_conn.normalized_weights,
                                         tract_lengths=vep_conn.tract_lengths, region_labels=vep_conn.region_labels,
                                         centres=vep_conn.centers, hemispheres=vep_conn.hemispheres,
                                         orientations=vep_conn.orientations, areas=vep_conn.areas)

    def config_simulation(self, head, hypothesis, settings=SimulationSettings()):

        tvb_conn = self._vep2tvb_connectivity(head.connectivity)
        coupl = coupling.Difference(a=1.)

        # Set noise:

        if isinstance(settings.noise_preconfig,noise.Noise):
            integrator = integrators.HeunStochastic(dt=settings.integration_step, noise=settings.noise_preconfig)
        else:
            settings.noise_intensity = numpy.array(settings.noise_intensity)
            if settings.noise_intensity.size == 1:
                settings.noise_intensity = numpy.repeat(numpy.squeeze(settings.noise_intensity),self.model.nvar)
            if numpy.min(settings.noise_intensity) > 0:
                    thisNoise = noise.Additive(nsig=settings.noise_intensity,
                                               random_stream=numpy.random.RandomState(seed=settings.noise_seed))
                    settings.noise_type = "Additive"
                    integrator = integrators.HeunStochastic(dt=settings.integration_step, noise=thisNoise)
            else:
                integrator = integrators.HeunDeterministic(dt=settings.integration_step)
                settings.noise_type = "None"

        #Set monitors:

        what_to_watch = []
        if isinstance(settings.monitors_preconfig, monitors.Monitor):
            what_to_watch = (settings.monitors_preconfig,)
        elif isinstance(settings.monitors_preconfig, tuple) or isinstance(settings.monitors_preconfig, list):
            for monitor in settings.monitors_preconfig:
                if isinstance(monitor, monitors.Monitor):
                    what_to_watch.append(monitor)
                what_to_watch = tuple(what_to_watch)

        # TODO: Find a better way to define monitor expressions without the need to modify the model...
        if settings.monitor_expressions is not None:
            self.model.variables_of_interest = settings.monitor_expressions

        #Create and configure TVB simulator object
        sim = simulator.Simulator(model=self.model, connectivity=tvb_conn, coupling=coupl, integrator=integrator,
                                  monitors=what_to_watch, simulation_length=settings.simulated_period)
        sim.configure()
        sim.initial_conditions = self.builder_initial_conditions(hypothesis, sim.model, sim.good_history_shape[0])

        #Update simulation settings
        settings.integration_step = integrator.dt
        settings.simulated_period = sim.simulation_length
        settings.integrator_type = integrator._ui_name
        settings.noise_ntau = integrator.noise.ntau
        settings.noise_intensity = numpy.array(settings.noise_intensity)
        settings.monitor_type = what_to_watch[0]._ui_name
        # TODO: find a way to store more than one monitors settings
        settings.monitor_sampling_period = what_to_watch[0].period
        settings.monitor_expressions = self.model.variables_of_interest
        settings.initial_conditions = sim.initial_conditions

        return sim, settings


    def launch_simulation(self, sim,  n_report_blocks=1):
        sim._configure_history(initial_conditions=sim.initial_conditions)
        if n_report_blocks<2:
            tavg_time, tavg_data = sim.run()[0]
            return tavg_time, tavg_data
        else:
            sim_length = sim.simulation_length / sim.monitors[0].period
            block_length = sim_length / n_report_blocks
            curr_time_step = 0.0
            curr_block = 1.0

            # Perform the simulation
            tavg_data, tavg_time = [], []

            start = time.time()

            for tavg in sim():

                curr_time_step += 1.0

                if not tavg is None:
                    tavg_time.append(tavg[0][0])
                    tavg_data.append(tavg[0][1])

                if curr_time_step >= curr_block * block_length:
                    end_block = time.time()
                    print_this = "\r" + "..." + str(100 * curr_time_step / sim_length) + "% done in " +\
                                 str(end_block-start) + " secs"
                    sys.stdout.write(print_this)
                    sys.stdout.flush()
                    curr_block += 1.0

            return numpy.array(tavg_time), numpy.array(tavg_data)


    def launch_pse(self, hypothesis, head, settings=SimulationSettings()):
        raise NotImplementedError()



###
# Prepare for TVB configuration
###

def calc_tvb_equilibrium_point(epileptor_model, hypothesis):

    #Calculate equilibrium point
    y1eq = calc_eq_y1(hypothesis.x1EQ, epileptor_model.c.T, d=epileptor_model.d)
    zeq = calc_eq_z_6d(hypothesis.x1EQ, epileptor_model.c.T, epileptor_model.Iext.T, a=epileptor_model.a,
                       b=epileptor_model.b)
    if epileptor_model.Iext2.size == 1:
        epileptor_model.Iext2 = epileptor_model.Iext2[0] * numpy.ones((hypothesis.n_regions, 1))
    (x2eq, y2eq) = calc_eq_pop2(hypothesis.x1EQ, zeq, epileptor_model.Iext2.T, s=epileptor_model.aa)
    geq = calc_eq_g(hypothesis.x1EQ)

    equilibrium_point = numpy.r_[hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq].astype('float32')

    #Assert equilibrium point
    coupl = calc_coupling(hypothesis.x1EQ, epileptor_model.Ks.T, hypothesis.weights)
    coupl = numpy.expand_dims(numpy.r_[coupl, 0.0 * coupl], 2).astype('float32')

    dfun = epileptor_model.dfun(numpy.expand_dims(equilibrium_point, 2).astype('float32'), coupl).squeeze()
    dfun_max = numpy.max(dfun, axis=1)

    dfun_max_cr = 10 ** -6 * numpy.ones(dfun_max.shape)
    dfun_max_cr[2] = 10 ** -2

    dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                      epileptor_model.c.squeeze(), epileptor_model.Iext.squeeze(), epileptor_model.x0.squeeze(),
                      numpy.zeros((hypothesis.n_regions,), dtype=epileptor_model.x0.type),
                      numpy.ones((hypothesis.n_regions,), dtype=epileptor_model.x0.type), epileptor_model.Ks.squeeze(),
                      hypothesis.weights, model="6d", zmode="lin",
                      y1=equilibrium_point[1].squeeze(), x2=equilibrium_point[3].squeeze(),
                      y2=equilibrium_point[4].squeeze(), g=equilibrium_point[5].squeeze(),
                      slope=epileptor_model.slope.squeeze(), a=1.0, b=3.0, d=epileptor_model.d,
                      s=epileptor_model.aa, Iext2=epileptor_model.Iext2.squeeze(),
                      tau1=epileptor_model.tt, tau0=1.0 / epileptor_model.r, tau2=epileptor_model.tau)

    max_dfun_diff = numpy.max(numpy.abs(dfun2 - dfun.squeeze()), axis=1)
    if numpy.any(max_dfun_diff > dfun_max_cr):
        warnings.warn("model dfun and calc_dfun functions do not return the same results!\n"
                      + "maximum difference = " + str(max_dfun_diff) + "\n"
                      + "model dfun = " + str(dfun) + "\n"
                      + "calc_dfun = " + str(dfun2))

    if numpy.any(dfun_max > dfun_max_cr):
        # raise ValueError("Equilibrium point for initial condition not accurate enough!\n" \
        #                  + "max(dfun) = " + str(dfun_max) + "\n"
        #                  + "model dfun = " + str(dfun))
        warnings.warn("Equilibrium point for initial condition not accurate enough!\n"
                         + "max(dfun) = " + str(dfun_max) + "\n"
                         + "model dfun = " + str(dfun))



    return hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq

def prepare_for_tvb_model(hypothesis, model, history_length):
    #Set default initial conditions right on the resting equilibrium point of the model...
    #...after computing the equilibrium point
    (x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ) = calc_tvb_equilibrium_point(model, hypothesis)
    initial_conditions = numpy.expand_dims(numpy.r_[x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ],2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# Prepare for tvb-epilepsy epileptor_models
###

def prepare_initial_conditionsl(hypothesis, model, history_length):
    # Set default initial conditions right on the resting equilibrium point of the model...
    # ...after computing the equilibrium point (and correct it for zeql for a >=6D model
    initial_conditions = calc_equilibrium_point(model, hypothesis)
    #-------------------The lines below are for a specific "realistic" demo simulation:---------------------------------
    #if isinstance(mode,EpileptorDPrealistic):
    #   shape = initial_conditions[6].shape
    #   type = initial_conditions[6].dtype
    #   initial_conditions[6] = 0.0** numpy.ones(shape,dtype=type) # hypothesis.x0.T
    #   initial_conditions[7] = 1.0 * numpy.ones((1,hypothesis.n_regions))#model.slope * numpy.ones((hypothesis.n_regions,1))
    #   initial_conditions[9] = 0.0 * numpy.ones((1,hypothesis.n_regions))#model.Iext2.T * numpy.ones((hypothesis.n_regions,1))
    # ------------------------------------------------------------------------------------------------------------------
    initial_conditions = numpy.expand_dims(initial_conditions, 2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# A helper function to make good choices for simulation settings, noise and monitors
###

def setup_simulation(model, dt, sim_length, monitor_period, scale_time=1,
                     noise_instance=None, noise_intensity=None,
                     monitor_expressions=None, monitors_instance=None, variables_names=None):

    if isinstance(model,EpileptorDP):
        #                                               history
        simulator_instance = SimulatorTVB(model, prepare_initial_conditionsl)
        if variables_names is None:
            variables_names = ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp']
    elif isinstance(model,EpileptorDP2D):
        simulator_instance = SimulatorTVB(model, prepare_initial_conditionsl)
        if variables_names is None:
            variables_names = ['x1', 'z']
    elif isinstance(model,EpileptorDPrealistic):
        simulator_instance = SimulatorTVB(model, prepare_initial_conditionsl)
        if variables_names is None:
            variables_names = ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x0ts', 'slopeTS', 'Iext1ts', 'Iext2ts', 'Kts', 'lfp']
    elif isinstance(model,Epileptor):
        simulator_instance = SimulatorTVB(model, prepare_for_tvb_model)
        if variables_names is None:
            variables_names = ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp']

    if monitor_expressions is None:
        monitor_expressions = []
        for i in range(model._nvar):
            monitor_expressions.append("y" + str(i))
        # Monitor adjusted to the model
        if not(isinstance(model,EpileptorDP2D)):
            monitor_expressions.append("y3 - y0")

    if monitors_instance is None:
        monitors_instance = monitors.TemporalAverage(period=monitor_period)
    else:
        if monitor_period is not None:
            monitors_instance.period = monitor_period

    if noise_instance is None:
        if noise_intensity is None:
            if numpy.all(noise_intensity is None):
                # Noise configuration
                if isinstance(model,EpileptorDPrealistic):
                    #                             x1  y1   z     x2   y2    g   x0   slope  Iext1 Iext2 K
                    noise_intensity = numpy.array([0., 0., 1e-7, 0.0, 1e-7, 0., 1e-8, 1e-3, 1e-8, 1e-3, 1e-9])
                elif isinstance(model,EpileptorDP2D):
                    #                              x1   z
                    noise_intensity = numpy.array([0., 5e-5])
                else:
                    #                              x1  y1   z     x2   y2   g
                    noise_intensity = numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

        # Preconfigured noise
        if isinstance(model,EpileptorDPrealistic):
            # Colored noise for realistic simulations
            eq = equations.Linear(parameters={"a": 1.0, "b": 0.0})  #  a*y+b, default = (1.0, 1.0)
            noise_instance = noise.Multiplicative(ntau=10, nsig=noise_intensity, b=eq,
                                                  random_stream=numpy.random.RandomState(seed=NOISE_SEED))
            noise_type = "Multiplicative"
            noise_shape = noise_instance.nsig.shape
            noise_instance.configure_coloured(dt=dt, shape=noise_shape)
        else:
            # White noise as a default choice:
            noise_instance = noise.Additive(nsig=noise_intensity,
                                            random_stream=numpy.random.RandomState(seed=NOISE_SEED))
            noise_instance.configure_white(dt=dt)
            noise_type = "Additive"
    else:
        if noise_intensity is not None:
            noise_instance.nsig = noise_intensity

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                  scale_time=scale_time,
                                  noise_preconfig=noise_instance, noise_type=noise_type,
                                  noise_intensity=noise_intensity, noise_ntau=noise_instance.ntau,
                                  noise_seed=NOISE_SEED,
                                  monitors_preconfig=monitors_instance, monitor_type=monitors_instance._ui_name,
                                  monitor_sampling_period=monitor_period, monitor_expressions=monitor_expressions,
                                  variables_names=variables_names)

    return simulator_instance, settings, variables_names
