"""
@version $Id: simulator_tvb.py 1664 2016-09-05 16:13:48Z denis $

Mechanism for launching TVB simulations.
"""

import numpy
from tvb.datatypes import connectivity
from tvb.simulator import coupling, integrators, models, monitors, noise, simulator
from tvb_epilepsy.tvb_api import epileptor_models
from tvb_epilepsy.base.simulators import ABCSimulator, SimulationSettings
from tvb_epilepsy.base.equilibrium_computation import coupling_calc, x0_calc, x0cr_rx0_calc, zeq_6d_calc,\
                                                      calc_equilibrium_point

class SimulatorTVB(ABCSimulator):

    def __init__(self, builder_model, builder_initial_conditions):
        self.builder_model = builder_model
        self.builder_initial_conditions = builder_initial_conditions

    @staticmethod
    def _vep2tvb_connectivity(vep_conn):
        return connectivity.Connectivity(use_storage=False, weights=vep_conn.normalized_weights,
                                         tract_lengths=vep_conn.tract_lengths, region_labels=vep_conn.region_labels,
                                         centres=vep_conn.centers, hemispheres=vep_conn.hemispheres,
                                         orientations=vep_conn.orientations, areas=vep_conn.areas)

    def config_simulation(self, hypothesis, head, vep_settings=SimulationSettings(),zmode=numpy.array("lin")):

        tvb_conn = self._vep2tvb_connectivity(head.connectivity)
        self.model = self.builder_model(hypothesis,variables_of_interest=vep_settings.monitor_expr, zmode=zmode)
        coupl = coupling.Difference(a=1.)
        
        if isinstance(vep_settings.noise_preconfig,noise.Noise):
            integrator = integrators.HeunStochastic(dt=vep_settings.integration_step, noise=vep_settings.noise_preconfig) 
        else:
            vep_settings.noise_intensity = numpy.array(vep_settings.noise_intensity)
            if vep_settings.noise_intensity.size==1:
                vep_settings.noise_intensity = numpy.repeat(numpy.squeeze(vep_settings.noise_intensity),self.model.nvar)
            if numpy.min(vep_settings.noise_intensity) > 0:
                    thisNoise = noise.Additive(nsig=vep_settings.noise_intensity,
                                               random_stream=numpy.random.RandomState(seed=vep_settings.integration_noise_seed))
                    integrator = integrators.HeunStochastic(dt=vep_settings.integration_step, noise=thisNoise)                           
            else:
                integrator = integrators.HeunDeterministic(dt=vep_settings.integration_step)
        
        mon_tavg = monitors.TemporalAverage(period=vep_settings.monitor_sampling_period)
        what_to_watch = mon_tavg

        sim = simulator.Simulator(model=self.model, connectivity=tvb_conn, coupling=coupl,
                                  integrator=integrator, monitors=what_to_watch,
                                  simulation_length=vep_settings.simulated_period)        
        return sim
        
    def launch_simulation(self, sim, hypothesis):
        sim.configure()
        self.initial_conditions = self.builder_initial_conditions(hypothesis, self.model, sim.good_history_shape[0])
        sim._configure_history(initial_conditions=self.initial_conditions)
        tavg_time, tavg_data = sim.run()[0]
        return tavg_time, tavg_data

    def launch_pse(self, hypothesis, head, vep_settings=SimulationSettings()):
        raise NotImplementedError()


###
# Prepare for TVB configuration
###


def build_tvb_model(hypothesis,variables_of_interest=["y3 - y0", "y2"], zmode="lin"):
    x0_transformed = _rescale_x0(hypothesis.x0, hypothesis.rx0, hypothesis.x0cr)
    model_instance = models.Epileptor(x0=x0_transformed,
                                      Iext=hypothesis.Iext1, 
                                      Ks=hypothesis.K,
                                      c =hypothesis.y0,
                                      variables_of_interest=variables_of_interest)
    return model_instance


def _rescale_x0(x0_orig, r, x0cr):
    return r*x0_orig-x0cr-5.0/3.0
    
#def _rescale_x0(original, to_min=-5, to_max=-1):
#    current_min = original.min()
#    current_max = original.max()
#    scaling_factor = (to_max - to_min) / (current_max - current_min)
#    return to_min + (original - current_min) * scaling_factor


def prepare_for_tvb_model(hypothesis, model, history_length):
    #Set default initial conditions right on the resting equilibrium point of the model...
    #...after computing the equilibrium point (and correct it for zeq, and for x1eq for original tvb model) for a >=6D model
    (x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ) = calc_equilibrium_point(model, hypothesis)
    initial_conditions = numpy.expand_dims(numpy.r_[x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ],2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions

###
# Prepare for epileptor_models.EpileptorDP2D
###


def build_ep_2sv_model(hypothesis, variables_of_interest=["y0", "y1"], zmode=numpy.array("lin")):
    if zmode=="lin":
        x0 = hypothesis.x0
        x0cr = hypothesis.x0cr
        r = hypothesis.rx0
    elif zmode == 'sig':
        #Correct Ceq, x0cr, rx0 and x0 for sigmoidal fz(x1)
        ceq = coupling_calc(hypothesis.x1EQ, hypothesis.K, hypothesis.weights)
        (x0cr,r)=x0cr_rx0_calc(hypothesis.y0, hypothesis.Iext1, epileptor_model="2d", zmode=zmode)
        x0 = x0_calc(hypothesis.x1EQ, hypothesis.zEQ, x0cr, r, ceq, zmode=zmode)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')
    model = epileptor_models.EpileptorDP2D(x0=x0,
                                           Iext1=hypothesis.Iext1, 
                                           K=hypothesis.K,
                                           yc =hypothesis.y0,
                                           r=r,
                                           x0cr=x0cr,
                                           variables_of_interest=variables_of_interest,
                                           zmode=zmode)
    return model


def prepare_for_2sv_model(hypothesis, model, history_length):
    # Set default initial conditions right on the resting equilibrium point of the model...
    # ...after computing it
    (x1EQ, zEQ) = calc_equilibrium_point(model, hypothesis)
    initial_conditions = numpy.expand_dims(numpy.r_[x1EQ, zEQ],2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# Prepare for epileptor_models.EpileptorDP
###


def build_ep_6sv_model(hypothesis,variables_of_interest=["y3 - y0", "y2"],zmode=numpy.array("lin")):
    #Correct Ceq, x0cr, rx0, zeq and x0 for 6D model
    ceq = coupling_calc(hypothesis.x1EQ, hypothesis.K, hypothesis.weights)
    (x0cr,r)=x0cr_rx0_calc(hypothesis.y0, hypothesis.Iext1, epileptor_model="6d", zmode=zmode)
    zeq=zeq_6d_calc(hypothesis.x1EQ, hypothesis.y0, hypothesis.Iext1)
    x0 = x0_calc(hypothesis.x1EQ, zeq, x0cr, r, ceq, zmode=zmode)
    model = epileptor_models.EpileptorDP(x0=x0,
                                         Iext1=hypothesis.Iext1, 
                                         K=hypothesis.K,
                                         yc =hypothesis.y0,
                                         r=r,
                                         x0cr=x0cr,
                                         variables_of_interest=variables_of_interest,
                                         zmode=zmode)
    return model


def prepare_for_6sv_model(hypothesis, model, history_length):
    # Set default initial conditions right on the resting equilibrium point of the model...
    # ...after computing the equilibrium point (and correct it for zeql for a >=6D model
    (x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ) = calc_equilibrium_point(model, hypothesis)
    initial_conditions = numpy.expand_dims(numpy.r_[x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ],2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# Prepare for epileptor_models.EpileptorDPrealistic
###


def build_ep_11sv_model(hypothesis, variables_of_interest=["y3 - y0", "y2"], zmode=numpy.array("lin")):
    # Correct Ceq, x0cr, rx0, zeq and x0 for >=6D model
    ceq = coupling_calc(hypothesis.x1EQ, hypothesis.K, hypothesis.weights)
    (x0cr, r) = x0cr_rx0_calc(hypothesis.y0, hypothesis.Iext1, epileptor_model="11d", zmode=zmode)
    zeq = zeq_6d_calc(hypothesis.x1EQ, hypothesis.y0, hypothesis.Iext1)
    x0 = x0_calc(hypothesis.x1EQ, zeq, x0cr, r, ceq, zmode=zmode)
    model = epileptor_models.EpileptorDPrealistic(x0=x0,
                                                  Iext1=hypothesis.Iext1, 
                                                  K=hypothesis.K,
                                                  yc =hypothesis.y0,
                                                  r=r,
                                                  x0cr=x0cr,
                                                  variables_of_interest=variables_of_interest,
                                                  zmode=zmode)
    return model


def prepare_for_11sv_model(hypothesis, model, history_length):
    # Set default initial conditions right on the resting equilibrium point of the model...
    # ...after computing the equilibrium point (and correct it for zeql for a >=6D model
    (x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ,  \
                                   x0o, slope0, Iext1, Iext2o, K) = calc_equilibrium_point(model, hypothesis)
    #-------------------The lines below are for a specific "realistic" demo simulation:---------------------------------
    shape = x1EQ.shape
    type = x1EQ.dtype
    x0o = 0.0** numpy.ones(shape,dtype=type) # hypothesis.x0.T
    slope0 = 1.0 * numpy.ones((1,hypothesis.n_regions))#model.slope * numpy.ones((hypothesis.n_regions,1))
    Iext2o = 0.0 * numpy.ones((1,hypothesis.n_regions))#model.Iext2.T * numpy.ones((hypothesis.n_regions,1))
    # ------------------------------------------------------------------------------------------------------------------
    initial_conditions = numpy.expand_dims(numpy.r_[x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ, x0o, slope0, Iext1, Iext2o, K],2)
    initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions

