"""
@version $Id: simulator_tvb.py 1664 2016-09-05 16:13:48Z denis $

Mechanism for launching TVB simulations.
"""

import numpy as np
from tvb.datatypes import connectivity
from tvb.simulator import coupling, integrators, models, monitors, noise, simulator
from tvb.epilepsy.tvb_api import epileptor_models
from tvb.epilepsy.base.simulators import ABCSimulator, SimulationSettings


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

    def config_simulation(self, hypothesis, head, vep_settings=SimulationSettings()):

        tvb_conn = self._vep2tvb_connectivity(head.connectivity)
        self.model = self.builder_model(hypothesis,variables_of_interest=vep_settings.monitor_expr)
        coupl = coupling.Difference(a=1.)
        
        if isinstance(vep_settings.noise_preconfig,noise.Noise):
            integrator = integrators.HeunStochastic(dt=vep_settings.integration_step, noise=vep_settings.noise_preconfig) 
        else:
            vep_settings.noise_intensity = np.array(vep_settings.noise_intensity)
            if vep_settings.noise_intensity.size==1:
                vep_settings.noise_intensity = np.repeat(np.squeeze(vep_settings.noise_intensity),self.model.nvar)
            if np.min(vep_settings.noise_intensity) > 0:
                    thisNoise = noise.Additive(nsig=vep_settings.noise_intensity,
                                               random_stream=np.random.RandomState(seed=vep_settings.integration_noise_seed))
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



def get_y1eq(x1eq,y0,d):
    return y0-d*x1eq**2
    
def get_2eq(n_regions,x1eq,zeq,Iext2):
    #g_eq = 0.1*x1eq (1)
    #y2eq = 0 (2)
    y2eq = np.zeros((n_regions,1))
    #-x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
    #-x2eq**3 + x2eq +2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
    #p3        p1                   p0 
    #-x2eq**3 + x2eq +0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
    p0 = 0.2*x1eq-0.3*(zeq-3.5)+Iext2  
    x2eq = np.zeros((n_regions,1))
    for i in range(n_regions):
        x2eq[i,0] = np.min( np.real( np.roots([-1.0, 0.0, 1.0, p0[i,0] ]) ) )   
    return (x2eq, y2eq)
    
#def get_2eq(n_regions,x1eq,zeq,Iext2):
#    #g_eq = 0.1*x1eq (1)
#    #y2eq = 6*(x2eq+0.25)*x1eq (2)
#    #-x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
#    #-x2eq**3 + x2eq -6*(x2eq+0.25)*x1eq+2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.5*x1eq+ 0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
#    #p3                p1                           p0   
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.3*x1eq -0.3*(zeq-3.5) +Iext2 =0
#    p0 = -1.3*x1eq-0.3*(zeq-3.5)+Iext2  
#    p1 = 1.0-6*x1eq
#    x2eq = np.zeros((n_regions,1))
#    for i in range(n_regions):
#        x2eq[i,0] = np.min( np.real( np.roots([-1.0, 0.0, p1[i,0], p0[i,0] ]) ) )   
#    #(2):
#    y2eq = 6*(x2eq+0.25)*x1eq
#    return (x2eq, y2eq)    

def get_geq(x1eq):
    return 0.1*x1eq
###
# Prepare for TVB configuration
###


def build_tvb_model(hypothesis,variables_of_interest=["y3 - y0", "y2"]):
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
    x1EQ = hypothesis.x1EQ.T-5.0/3.0
    y1EQ = get_y1eq(x1EQ,hypothesis.y0.T,model.d)
    zEQ = hypothesis.zEQ.T
    (x2EQ, y2EQ) = get_2eq(hypothesis.n_regions,x1EQ,zEQ,model.Iext2)
    gEQ = get_geq(x1EQ)
    initial_conditions = np.array((x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ ))
    initial_conditions = np.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions

###
# Prepare for epileptor_models.EpileptorDP2D
###


def build_ep_2sv_model(hypothesis,variables_of_interest=["y0", "y1"]):
    model = epileptor_models.EpileptorDP2D(x0=hypothesis.x0, 
                                           Iext1=hypothesis.Iext1, 
                                           K=hypothesis.K,
                                           yc =hypothesis.y0,
                                           r=hypothesis.rx0,
                                           x0cr=hypothesis.x0cr,
                                           variables_of_interest=variables_of_interest)
    return model


def prepare_for_2sv_model(hypothesis, model, history_length):
    initial_conditions = np.array((hypothesis.x1EQ.T,
                                   hypothesis.zEQ.T
                                   ))
    initial_conditions = np.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# Prepare for epileptor_models.EpileptorDP
###


def build_ep_6sv_model(hypothesis,variables_of_interest=["y3 - y0", "y2"]):
    model = epileptor_models.EpileptorDP(x0=hypothesis.x0, 
                                         Iext1=hypothesis.Iext1, 
                                         K=hypothesis.K,
                                         yc =hypothesis.y0,
                                         r=hypothesis.rx0,
                                         x0cr=hypothesis.x0cr,
                                         variables_of_interest=variables_of_interest)
    return model


def prepare_for_6sv_model(hypothesis, model, history_length):
    x1EQ = hypothesis.x1EQ.T
    x1EQ53 = x1EQ-5.0/3.0
    zEQ = hypothesis.zEQ.T
    y1EQ = get_y1eq(x1EQ53,hypothesis.y0.T,5.0)
    (x2EQ, y2EQ) = get_2eq(hypothesis.n_regions,x1EQ53,zEQ,model.Iext2)
    gEQ = get_geq(x1EQ53)
    initial_conditions = np.array((x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ))
    initial_conditions = np.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions


###
# Prepare for epileptor_models.EpileptorDPrealistic
###


def build_ep_11sv_model(hypothesis,variables_of_interest=["y3 - y0", "y2"]):
    model = epileptor_models.EpileptorDPrealistic(x0=hypothesis.x0, 
                                                  Iext1=hypothesis.Iext1, 
                                                  K=hypothesis.K,
                                                  yc =hypothesis.y0,
                                                  r=hypothesis.rx0,
                                                  x0cr=hypothesis.x0cr,
                                                  variables_of_interest=variables_of_interest)
    return model


def prepare_for_11sv_model(hypothesis, model, history_length):
    x1EQ = hypothesis.x1EQ.T
    x1EQ53 = x1EQ-5.0/3.0
    zEQ = hypothesis.zEQ.T
    y1EQ = get_y1eq(x1EQ53,hypothesis.y0.T,5.0)
    (x2EQ, y2EQ) = get_2eq(hypothesis.n_regions,x1EQ53,zEQ,model.Iext2)
    gEQ = get_geq(x1EQ53)
    x0o = 0.0** np.ones((hypothesis.n_regions,1)) # hypothesis.x0.T
    slope0 = 1.0 * np.ones((hypothesis.n_regions,1))#model.slope * np.ones((hypothesis.n_regions,1))
    Iext1o = hypothesis.Iext1.T
    Iext2o = 0.0 * np.ones((hypothesis.n_regions,1))#model.Iext2.T * np.ones((hypothesis.n_regions,1))
    Ko = hypothesis.K.T
    initial_conditions = np.array((x1EQ, y1EQ, zEQ, x2EQ, y2EQ, gEQ,
                                   x0o, slope0, Iext1o, Iext2o,Ko))
    initial_conditions = np.tile(initial_conditions, (history_length, 1, 1, 1))
    return initial_conditions

