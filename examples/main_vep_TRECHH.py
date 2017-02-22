"""
@version $Id: main_vep_TREC159.py 1651 2016-09-01 17:47:05Z denis $

Entry point for working with VEP
"""
import os
import numpy as np
import copy as cp
from scipy.io import savemat
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.hypothesis import Hypothesis, X0_DEF
from tvb_epilepsy.base.utils import initialize_logger, filter_data, calculate_projection
from tvb_epilepsy.base.plot_tools import plot_head, plot_hypothesis, plot_timeseries, plot_raster
from tvb_epilepsy.tvb_api.readers_tvb import TVBReader
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.custom.readers_episense import EpisenseReader
from tvb.simulator.models.epileptor import *
from tvb.datatypes import equations
from tvb.simulator import noise

SHOW_FLAG=False
SHOW_FLAG_SIM=True
SAVE_FLAG=True

#Modify data folders for this example:
DATA_TRECHH = '/Users/dionperd/Dropbox/Work/VBtech/DenisVEP/Results/TRECHH'
#CON_DATA = 'connectivity_2_hypo.zip'
CONNECT_DATA = 'connectivity_hypo.zip'

#Set a special scaling for HH regions, for this example:
#Set Khyp >=1.0     
Khyp = 5.0

#A few functions for prettier code:

def set_time_scales(fs=4096.0, dt=None, time_length=1000.0, scale_time=1.0, scale_fsavg=8.0,
                    hpf_low=None, hpf_high=None, report_percentage=10,):
    if dt is None:
        dt = 1000.0 / fs / scale_time # msec
    else:
        dt = dt / scale_time
    fsAVG = fs / scale_fsavg
    monitor_period = scale_fsavg * dt
    sim_length = time_length / scale_time
    time_length_avg = numpy.round(sim_length / scale_fsavg)
    n_report_blocks = report_percentage * numpy.round(time_length_avg / 100)
    hpf_fs = fsAVG * scale_time
    if hpf_low is None:
        hpf_low = max(16.0 , 1000.0 / time_length) * scale_time  # msec
    else:
        hpf_low = hpf_low * scale_time
    if hpf_high is None:
        hpf_high = min(250.0 * scale_time, hpf_fs)
    else:
        hpf_high = hpf_high * scale_time
    return fs, dt, fsAVG, scale_time, sim_length, monitor_period, n_report_blocks, hpf_fs, hpf_low, hpf_high


def set_simulation(model,dt):

    if model == '6D':
        #                                        epileptor model,      history
        simulator_instance = SimulatorTVB(build_ep_6sv_model, prepare_for_6sv_model)
        nvar = 6
    elif model == '2D':
        simulator_instance = SimulatorTVB(build_ep_2sv_model, prepare_for_2sv_model)
        nvar = 2
    elif model == '11D':
        simulator_instance = SimulatorTVB(build_ep_11sv_model, prepare_for_11sv_model)
        nvar = 11
    elif model == 'tvb':
        simulator_instance = SimulatorTVB(build_tvb_model, prepare_for_tvb_model)
        nvar = 6

    # Monitor adjusted to the model
    if model != '2D':
        monitor_expr = ["y3-y0"]
        for i in range(nvar):
            monitor_expr.append("y" + str(i))
    else:
        monitor_expr = []
        for i in range(nvar):
            monitor_expr.append("y" + str(i))

    # Noise configuration
    if model == '11D':
        #                             x1  y1   z     x2   y2   g   x0   slope Iext1 Iext2  K
        noise_intensity = 0.1 * numpy.array([0., 0., 1e-6, 0.0, 1e-6, 0., 1e-7, 1e-2, 1e-7, 1e-2, 1e-8])
    elif model == '2D':
        #                                     x1   z
        noise_intensity = 1.0 * numpy.array([0., 5e-5])
    else:
        #                                     x1  y1   z     x2   y2   g
        noise_intensity = 0.1 * numpy.array([0., 0., 5e-5, 0.0, 5e-5, 0.])

    # Preconfigured noise
    if model == '11D':
        # Colored noise for realistic simulations
        eq = equations.Linear(parameters={"a": 0.0, "b": 1.0})  # default = a*y+b
        this_noise = noise.Multiplicative(ntau=10, nsig=noise_intensity, b=eq,
                                          random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_shape = this_noise.nsig.shape
        this_noise.configure_coloured(dt=dt, shape=noise_shape)
    else:
        # White noise as a default choice:
        this_noise = noise.Additive(nsig=noise_intensity, random_stream=np.random.RandomState(seed=NOISE_SEED))
        this_noise.configure_white(dt=dt)


    return simulator_instance, monitor_expr, this_noise


def  unpack_sim_results(model, ttavg, tavg_data, projections, scale_time, hpf_low, hpf_high, hpf_fs):
    res=dict()
    if isinstance(model, EpileptorDP2D):
        res['x1'] = np.array(tavg_data[:, 0, :, 0], dtype='float32')
        res['z'] = np.array(tavg_data[:, 1, :, 0], dtype='float32')
        i = 1
        res['seeg'] = []
        for i in range(len(projections)):
            res['seeg'].append(np.dot(res['x1'], projections[i].T))
    else:
        res['lfp'] = np.array(tavg_data[:, 0, :, 0], dtype='float32')
        res['z'] = np.array(tavg_data[:, 3, :, 0], dtype='float32')
        res['x1'] = np.array(tavg_data[:, 1, :, 0], dtype='float32')
        res['y1'] = np.array(tavg_data[:, 2, :, 0], dtype='float32')
        res['x2'] = np.array(tavg_data[:, 4, :, 0], dtype='float32')
        res['y2'] = np.array(tavg_data[:, 5, :, 0], dtype='float32')
        res['g'] = np.array(tavg_data[:, 6, :, 0], dtype='float32')
        hpf = np.empty((ttavg.size, hyp.n_regions)).astype(np.float32)
        for i in range(hyp.n_regions):
            res['hpf'][:, i] = filter_data(res['lfp'][:, i], hpf_low, hpf_high, hpf_fs)  # .astype(np.float32) #.transpose()
        res['hpf'] = np.array(res['hpf'], dtype='float32')
        i = 1
        res['seeg'] = []
        for i in range(len(projections)):
            res['seeg'].append(np.dot(res['hpf'], projections[i].T))
    if isinstance(model, EpileptorDPrealistic):
        res['x0ts'] = np.array(tavg_data[:, 7, :, 0], dtype='float32')
        res['slopeTS'] = np.array(tavg_data[:, 8, :, 0], dtype='float32')
        res['Iext1ts'] = np.array(tavg_data[:, 9, :, 0], dtype='float32')
        res['Iext2ts'] = np.array(tavg_data[:, 10, :, 0], dtype='float32')
        res['Kts'] = np.array(tavg_data[:, 11, :, 0], dtype='float32')
    res['time'] = scale_time * np.array(ttavg, dtype='float32')
    return res


def plot_results(model, hyp, head, res, sensorsSEEG):

    if isinstance(sim.model, EpileptorDP2D):
        plot_timeseries(res['time'], {'x1': res['x1'], 'z(t)': res['z']},
                        hyp.seizure_indices, title=" Simulated TAVG for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
    else:
        plot_timeseries(res['time'], {'LFP(t)': res['lfp'], 'z(t)': res['z']},
                        hyp.seizure_indices, title=" Simulated LFP-z for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x1(t)': res['x1'], 'y1(t)': res['y1']},
                        hyp.seizure_indices, title=" Simulated pop1 for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']}, seizure_indices,
                        title=" Simulated pop2-g for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        start_plot = numpy.round(0.01 * res['hpf'].shape[0])
        plot_raster(res['time'][start_plot:], {'hpf': res['hpf'][start_plot:, :]}, seizure_indices,
                    title=" Simulated hfp rasterplot for " + hyp.name, offset=10.0,
                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                    labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)

    if isinstance(sim.model, EpileptorDPrealistic):
        plot_timeseries(res['time'], {'1/(1+exp(-10(z-3.03))': 1 / (1 + np.exp(-10 * (res['z'] - 3.03))),
                                      'slope': res['slopeTS'], 'Iext2': res['Iext2ts']},
                        hyp.seizure_indices, title=" Simulated controlled parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x0': res['x0ts'], 'Iext1':  res['Iext1ts'] , 'K': res['Kts']},
                        hyp.seizure_indices, title=" Simulated parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)

        #            plot_timeseries(ttavg[100:], {'SEEG': seeg[100:,:]}, title=" Simulated SEEG"+str(i)+" for " + hyp.name,
        #                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
        #                        labels = sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
    for i in range(len(sensorsSEEG)):
        start_plot = numpy.round(0.01*res['seeg'][i].shape[0])
        plot_raster(ttavg[start_plot:], {'SEEG': res['seeg'][i][start_plot:, :]}, title=" Simulated SEEG" + str(i) + " rasterplot for " + hyp.name,
                    offset=10.0, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES,
                    labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)


#The main function...
if __name__ == "__main__":

#-------------------------------Reading data-----------------------------------
    logger = initialize_logger(__name__)

    # if DATA_MODE == 'ep':
    #     logger.info("Reading from EPISENSE")
    #     data_folder = os.path.join(DATA_EPISENSE, 'Head_TREC')  # Head_TREC 'Head_JUNCH'
    #     from tvb_epilepsy.custom.readers_episense import EpisenseReader
    #
    #     reader = EpisenseReader()
    # else:
    #     logger.info("Reading from TVB")
    #     data_folder = DATA_TVB
    #     reader = TVBReader()

    data_folder = os.path.join(DATA_TRECHH, 'Head_TREC') #Head_TREC 'Head_JUNCH'
    reader = EpisenseReader()
    logger.info("We will be reading from location " + data_folder)
    #Read standard TREC head
    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    logger.debug("Loaded Head " + str(head))
    
    #Read TREC connectivity with hypothalamus pathology
    data_folder = os.path.join(DATA_TRECHH, CONNECT_DATA) 
    reader = TVBReader()
    TRECHHcon = reader.read_connectivity(data_folder)
    logger.debug("Loaded Connectivity " + str(head.connectivity))

    #Create missing hemispheres:
    nRegions = TRECHHcon.region_labels.shape[0]
    TRECHHcon.hemispheres = np.ones((nRegions,),dtype='int')
    for ii in range(nRegions):
        if (TRECHHcon.region_labels[ii].find('Right') == -1) and \
           (TRECHHcon.region_labels[ii].find('-rh-') == -1):  # -1 will be returned when a is not in b
           TRECHHcon.hemispheres[ii]=0
           
    #Adjust pathological connectivity     
    w_hyp = np.ones((nRegions,nRegions),dtype = 'float')
    if Khyp>1.0:
        w_hyp[(nRegions-2):,:] = Khyp 
        w_hyp[:,(nRegions-2):] = Khyp 
    TRECHHcon.normalized_weights = w_hyp*TRECHHcon.normalized_weights   
    
    #Update head with the correct connectivity and sensors' projections
    head.connectivity = TRECHHcon
    sensorsSEEG=[]
    projections=[]    
    for sensors, projection in head.sensorsSEEG.iteritems():
        if projection is None:
            continue
        else:
            projection = calculate_projection(sensors, head.connectivity)
            head.sensorsSEEG[sensors]=projection
            print projection.shape
            sensorsSEEG.append(sensors)
            projections.append(projection) 
    plot_head(head, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, 
              figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)

             
#--------------------------Hypothesis and LSA-----------------------------------
       
    # Next configure two seizure hypothesis
       
    #%Tim:
    #% EZ:
    #% 10 ctx-lh-entorhinal -1.7          15
    #% 11 ctx-lh-entorhinal -1.7          15
    #% 146 Left-Hippocampus -1.7          7  
    #% 147 Left-Amygdala -1.7             8
    #% 154 Right-Hippocampus -1.2        50
    #% 157 Left-Hypothalamus -1.7        88
    #% 158 Right-Hypothalamus -1.7       89
    #%  
    #% PZ:
    #% 31 ctx-lh-parahippocampal -1.74   25
    #% 13 ctx-lh-fusiform -1.74          16
    #% 30 ctx-lh-parahippocampal -1.98   25
    #% 140 Brain-Stem -1.93              1
    #% 142 Left-Thalamus-Proper -1.93    3
    #
    #%Denis:
    #% PZ=Ev:
    #%     'Left-Thalamus-Proper'      3
    #%     'Right-Thalamus-Proper'     46
    #%     'ctx-lh-parahippocampal'    25
    #%     'ctx-lh-fusiform           16
    #%     'Brain-Stem'                1
    #%     'ctx-lh-parahippocampal'   25
    #%     'ctx-rh-entorhinal'        58
    #%     'ctx-rh-parahippocampal'   68
    #%     'Right-Cerebellum-Cortex'   45
    #%     'ctx-rh-fusiform'           59   
       
    #seizure_indices = np.array([1,    3,   7,   8,  15,  16,  25,  50,   88, 89], dtype=np.int32)
    seizure_indices = np.array([50], dtype=np.int32)
    
    hyp_ep = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights, \
                        "EP Hypothesis", x1eq_mode = "linTaylor")  #"optimize"
    hyp_ep.K = 0.1*hyp_ep.K # 1.0/np.max(head.connectivity.normalized_weights)
    # iE = np.array([1,    3,   7,   8,  15,  16,  25,  50, 88,  89])
    # E = 0*np.array(iE, dtype=np.float32)
    # E[7]=0.65
    # E[2]=0.800
    # E[4]=0.3775
    # E[8]=0.425
    # E[9]=0.900
    iE = np.array([50])
    E = np.array([0.9], dtype=numpy.float32)
    hyp_ep.configure_e_hypothesis(iE, E, seizure_indices)
    logger.debug(str(hyp_ep))
    plot_hypothesis(hyp_ep, head.connectivity.region_labels,
                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, 
                    figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)

#    hyp_exc = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights, "x0 Hypothesis")
#    #hyp_exc = cp.deepcopy(hyp_ep)
#    #hyp_exc.name = "EP & x0 Hypothesis"
#    hyp_exc.K = 0.1*hyp_exc.K 
#    ii = np.array(range(head.number_of_regions), dtype=np.int32)
#    ix0 = ii#np.delete(ii, 0)
#    x0 = X0_DEF*ix0
#    x0[50] = 0.8
#    hyp_exc.configure_x0_hypothesis(ix0, x0, seizure_indices)
#    logger.debug(str(hyp_exc))
#    plot_hypothesis(hyp_exc, head.connectivity.region_labels,
#                    save_flag=SAVE_FLAG, show_flag=True, 
#                    figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)

    
#------------------------------Simulation--------------------------------------
    hyp_ep.K = hyp_ep.K

    # Time scales (all times should be in msecs):
    (fs, dt, fsAVG, scale_time, sim_length, monitor_period, n_report_blocks, hpf_fs, hpf_low, hpf_high) = \
                                                                set_time_scales(fs=4096.0, dt=None, time_length=100.0,
                                                                                scale_time=1.0, scale_fsavg=2.0,
                                                                                hpf_low=None, hpf_high=None,
                                                                                report_percentage=10.0)

     # Now Simulate
    # Choosing the model:
    model = '2D' #'6D', '2D', '11D', 'tvb'
    (simulator_instance, monitor_expr, this_noise) = set_simulation(model,dt)
    
    #Now simulate and plot
    for hyp in (hyp_ep,): # ,hyp_exc #length=30000
        settings = SimulationSettings(length=sim_length, integration_step=dt, monitor_sampling_period=monitor_period,
                                      noise_preconfig=this_noise, monitor_expr=monitor_expr)
        sim= simulator_instance.config_simulation(hyp, head, settings)
        #Here make all changes you want to the model and simulator
        if isinstance(sim.model,Epileptor):
            sim.model.tt = scale_time * 0.25 * sim.model.tt #default = 1.0
            #sim.model.r = 1.0/10000  # default = 1.0 / 2857.0
        else:
            #sim.model.tau0 = 2857.0 # default = 2857.0
            sim.model.tau1 = scale_time * sim.model.tau1 # default = 0.25
        if isinstance(sim.model, EpileptorDPrealistic):
            #sim.model.tau0 = 10000 # default = 10000
            sim.model.tau1 = scale_time * sim.model.tau1 # default = 0.25
            sim.model.slope = 0.25
            sim.model.pmode = np.array("z")  # "z","g","z*g", default="const"
        #sim.model.zmode = np.array("sig")
        ttavg, tavg_data = simulator_instance.launch_simulation(sim, hyp, n_report_blocks=n_report_blocks)
        logger.info("Simulated signal return shape: " + str(tavg_data.shape))
        logger.info("Time: " + str(ttavg[0]) + " - " + str(ttavg[-1]))
        logger.info("Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max()))

        res = unpack_sim_results(sim.model, ttavg, tavg_data, projections, scale_time, hpf_high, hpf_fs)

        plot_results(sim.model, hyp, head, res, sensorsSEEG)

        res['time_units'] = 'msec'
        savemat(os.path.join(FOLDER_RES, hyp.name + "_ts.mat"), res)

