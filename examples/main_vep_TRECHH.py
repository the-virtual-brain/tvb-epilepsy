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
from tvb.simulator import noise
from tvb.simulator.models.epileptor import *
from tvb.datatypes import equations

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

             
#--------------------------Hypotheis and LSA-----------------------------------
       
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
    iE = np.array([1,    3,   7,   8,  15,  16,  25,  50, 88,  89])
    E = 0*np.array(iE, dtype=np.float32)
    E[7]=0.65
    E[2]=0.800
    E[4]=0.3775
    E[8]=0.425
    E[9]=0.900
#    iE = np.array([50])
#    E = np.array[0.8], dtype=np.float32)
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
    
    # Now Simulate
    # Choosing the model:
    model='6D' #'6D', '2D', '11D', 'tvb'
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

    #Monitor adjusted to the model
    if model != '2D':
        monitor_expr = ["y3-y0"]
        for i in range(nvar):
            monitor_expr.append("y" + str(i))
    else:
        monitor_expr = []
        for i in range(nvar):
            monitor_expr.append("y" + str(i))
        
    #Time scales:    
    fs = 2*1024.0
    dt = 1000.0/fs
    fsAVG = 512.0 #fs/10
    fsSEEG = fsAVG
    
    #Noise configuration

    if model == '11D':
        #                             x1  y1   z     x2   y2   g   x0   slope Iext1 Iext2  K
        noise_intensity=0.1*np.array([0., 0., 1e-6, 0.0, 1e-6, 0., 1e-7, 1e-2, 1e-7, 1e-2, 1e-8])
    elif model == '2D':
        #                                     x1   z
        noise_intensity = 0.1 ** numpy.array([0., 5e-5])
    else:
        #                                     x1  y1   z     x2   y2   g
        noise_intensity = 0.1 ** numpy.array([0., 0., 5e-5, 0.0, 5e-5, 0.])
    #Preconfigured noise                                     
    #Colored noise:
    dtN = 1000.0/fsAVG
    eq=equations.Linear(parameters={"a": 0.0, "b": 1.0}) #default = a*y+b
    noise_color=noise.Multiplicative(ntau = 10, nsig=noise_intensity, b=eq,
                     random_stream=np.random.RandomState(seed=NOISE_SEED))
    noise_shape = noise_color.nsig.shape    
    noise_color.configure_coloured(dt=dt,shape=noise_shape)
    #White noise:
    #noise_white.configure_white(dt=dt)
    
    #Now simulate and plot
    for hyp in (hyp_ep,): # ,hyp_exc
        settings = SimulationSettings(length=30000,integration_step=dt, monitor_sampling_period=fs/fsSEEG*dt,
                                      noise_preconfig=noise_color,
                                      monitor_expr=monitor_expr) #noise_intensity=noise_intensity, 
        sim= simulator_instance.config_simulation(hyp, head, settings)
        #Here make all changes you want to the model and simulator
        sim.model.slope = 0.25
        if isinstance(sim.model,Epileptor):
            sim.model.tt = 0.25*sim.model.tt
            sim.model.r = 1.0/10000
        else:
            sim.model.tau0 = 10000
            sim.model.tau1 = 0.25*sim.model.tau1
        if isinstance(sim.model, EpileptorDPrealistic):
            sim.model.pmode = np.array("z")  # "z","g","z*g", default="const"
        #sim.model.zmode = np.array("sig")
        ttavg, tavg_data = simulator_instance.launch_simulation(sim,hyp)
        logger.info("Simulated signal return shape: " + str(tavg_data.shape))
        logger.info("Time: " + str(ttavg[0]) + " - " + str(ttavg[-1]))
        logger.info("Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max()))
        if ~isinstance(sim.model, EpileptorDP2D):
            lfp = np.array(tavg_data[:, 0, :, 0], dtype='float32')
            z = np.array(tavg_data[:, 3, :, 0], dtype='float32')
            x1 = np.array(tavg_data[:, 1, :, 0], dtype='float32')
            y1 = np.array(tavg_data[:, 2, :, 0],dtype='float32')
            x2 = np.array(tavg_data[:, 4, :, 0],dtype='float32')
            y2 = np.array(tavg_data[:, 5, :, 0],dtype='float32')
            g = np.array(tavg_data[:, 6, :, 0],dtype='float32')
        else:
            lfp = np.array(tavg_data[:, 0, :, 0], dtype='float32')
            z = np.array(tavg_data[:, 1, :, 0], dtype='float32')
        if isinstance(sim.model,EpileptorDPrealistic):
            x0ts=np.array(tavg_data[:, 7, :, 0],dtype='float32')
            slopeTS=np.array(tavg_data[:, 8, :, 0],dtype='float32')
            Iext1ts=np.array(tavg_data[:, 9, :, 0],dtype='float32')
            Iext2ts=np.array(tavg_data[:, 10, :, 0],dtype='float32')
            Kts=np.array(tavg_data[:, 11, :, 0],dtype='float32')
        hpf = np.empty((ttavg.size,hyp.n_regions)).astype(np.float32)
        for i in range(hyp.n_regions):
            hpf[:,i] = filter_data(lfp[:,i], 10, 250, fsAVG)#.astype(np.float32) #.transpose()
        hpf=np.array(hpf,dtype='float32')   
        ttavg = np.array(ttavg,dtype='float32')
        #hpf = filter_data(lfp, fsAVG/30, fsAVG/3, fsAVG)
        plot_timeseries(ttavg, {'LFP': lfp, 'z(t)': z, 'HPF LFP': hpf},
                        seizure_indices, title=" Simulated TAVG for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        if ~isinstance(sim.model, EpileptorDP2D):
            plot_timeseries(ttavg, {'x1(t)': x1, 'y1(t)': y1, 'z(t)': z},
                        seizure_indices, title=" Simulated pop1-z for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_timeseries(ttavg, {'x2(t)': x2, 'y2(t)': y2, 'g(t)': g},seizure_indices,
                        title=" Simulated pop2-g for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        if isinstance(sim.model, EpileptorDPrealistic):
            plot_timeseries(ttavg, {'1/(1+exp(-10(z-3.03))': 1/(1+np.exp(-10*(z-3.03))), 'slope': slopeTS, 'Iext2': Iext2ts},
                        seizure_indices, title=" Simulated controlled parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_timeseries(ttavg, {'x0': x0ts, 'Iext1': Iext1ts,'K': Kts},
                        seizure_indices, title=" Simulated parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
                       
        #for i in range(len(projections)):
        i=1
        seeg = np.dot(hpf,projections[i].T)
#            plot_timeseries(ttavg[100:], {'SEEG': seeg[100:,:]}, title=" Simulated SEEG"+str(i)+" for " + hyp.name,
#                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
#                        labels = sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
        plot_raster(ttavg[100:], {'SEEG': seeg[100:,:]}, title=" Simulated SEEG"+str(i)+" rasterplot for " + hyp.name,
                        offset=10.0,save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
                        
        plot_raster(ttavg[100:], {'hpf': hpf[100:,:]}, seizure_indices,
                        title=" Simulated hfp"+str(i)+" rasterplot for " + hyp.name,offset=10.0,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG_SIM, figure_dir=FOLDER_FIGURES, 
                        labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        if isinstance(sim.model, EpileptorDPrealistic):
            savemat(os.path.join(FOLDER_RES, hyp.name+"_ts.mat"),{'lfp':lfp,
                                                                  'seeg': seeg,
                                                                  'hpf': hpf,
                                                                  'z': z,
                                                                  'x1': x1,
                                                                  'x2': x2,
                                                                  'y1': y1,
                                                                  'y2': y2,
                                                                  'g': g,
                                                                  'x0ts': x0ts,
                                                                  'slopeTS': slopeTS,
                                                                  'Iext1ts': Iext1ts,
                                                                  'Iext2ts': Iext2ts,
                                                                  'Kts': Kts,
                                                                  'time_in_ms': ttavg})
        else:
            savemat(os.path.join(FOLDER_RES, hyp.name + "_ts.mat"), {'lfp': lfp,
                                                                     'seeg': seeg,
                                                                     'hpf': hpf,
                                                                     'z': z,
                                                                     'x1': x1,
                                                                     'x2': x2,
                                                                     'y1': y1,
                                                                     'y2': y2,
                                                                     'g': g,
                                                                     'time_in_ms': ttavg})