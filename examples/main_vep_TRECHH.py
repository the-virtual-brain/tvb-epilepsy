"""
@version $Id: main_vep_TREC159.py 1651 2016-09-01 17:47:05Z denis $

Entry point for working with VEP
"""
import os
import numpy as np
import copy as cp
from scipy.io import savemat
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.base.utils import initialize_logger, calculate_projection, set_time_scales, filter_data
from tvb_epilepsy.tvb_api.readers_tvb import TVBReader
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.custom.readers_episense import EpisenseReader
from tvb_epilepsy.base.plot_tools import plot_head, plot_hypothesis, plot_sim_results


SHOW_FLAG = False
SAVE_FLAG = True
SHOW_FLAG_SIM = False

#Modify data folders for this example:
DATA_TRECHH = '/Users/dionperd/Dropbox/Work/VBtech/DenisVEP/Results/TRECHH'
#CON_DATA = 'connectivity_2_hypo.zip'
CONNECT_DATA = 'connectivity_hypo.zip'

#Set a special scaling for HH regions, for this example:
#Set Khyp >=1.0     
Khyp = 5.0


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
    #plot_head(head, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
             
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
                        "EP Hypothesis", x1eq_mode="optimize")  #"linTaylor"
    hyp_ep.K = 0.1*hyp_ep.K # 1.0/np.max(head.connectivity.normalized_weights)
    # iE = np.array([1,    3,   7,   8,  15,  16,  25,  50, 88,  89])
    # E = 0*np.array(iE, dtype=np.float32)
    # E[7]=0.65
    # E[2]=0.800
    # E[4]=0.3775
    # E[8]=0.425
    # E[9]=0.900
    #iE = np.array([50])
    #E = np.array([0.25], dtype=numpy.float32)
    iE = np.array(range(hyp_ep.n_regions))
    E = (0.5 * np.ones((1,hyp_ep.n_regions))).tolist()
    hyp_ep.configure_e_hypothesis(iE, E, seizure_indices)
    logger.debug(str(hyp_ep))
    #plot_hypothesis(hyp_ep, head.connectivity.region_labels, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
    #               figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)

    # hyp_exc = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights,
    #                     "x0 Hypothesis", x1eq_mode="optimize")  #"optimize" or "linTaylor"
    # hyp_exc.K = 0.1 * hyp_exc.K
    # ix0 =range(head.number_of_regions)
    # x0 = (X0_DEF * numpy.ones((len(ix0),), dtype='float32')).tolist()

    hyp_exc = cp.deepcopy(hyp_ep)
    hyp_exc.name = "EP & x0 Hypothesis"
    #hyp_exc.x1eq_mode = "linTaylor"
    ix0 = [51]
    x0 = (0.5 * numpy.ones((len(ix0),), dtype='float32')).tolist()

    hyp_exc.configure_x0_hypothesis(ix0, x0, seizure_indices)

    x0_opt = numpy.array(hyp_exc.x0)
    x1EQ_opt = numpy.array(hyp_exc.x1EQ)
    E_opt = numpy.array(hyp_exc.E)

    hyp_exc = cp.deepcopy(hyp_ep)
    hyp_exc.name = "EP & x0 Hypothesis"
    hyp_exc.x1eq_mode = "linTaylor"
    ix0 = [51]
    x0 = (0.5 * numpy.ones((len(ix0),), dtype='float32')).tolist()

    hyp_exc.configure_x0_hypothesis(ix0, x0, seizure_indices)


    logger.debug(str(hyp_exc))
    plot_hypothesis(hyp_exc, head.connectivity.region_labels,
                    save_flag=SAVE_FLAG, show_flag=True,
                    figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)

    
#------------------------------Simulation--------------------------------------

    #We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    # Set time scales (all times should be in msecs):
    (fs, dt, fsAVG, scale_time, sim_length, monitor_period,
     n_report_blocks, hpf_fs, hpf_low, hpf_high) = set_time_scales(fs=2*4096.0, dt=None, time_length=1000.0,
                                                                   scale_time=2.0, scale_fsavg=2.0,
                                                                   hpf_low=None, hpf_high=None,
                                                                   report_every_n_monitor_steps=10.0)

    #Now simulate and plot for each hypothesis
    for hyp in (hyp_ep,): # ,hyp_exc #length=30000

        # Choose the model and build it on top of the specific hypothesis, adjust parameters:
        model = '6D'  # '6D', '2D', '11D', 'tvb'
        if model == '6D':
            model = build_ep_6sv_model(hyp_ep, zmode=numpy.array("lin"))
            # model.tau0 = 2857.0 # default = 2857.0
            model.tau1 *= scale_time  # default = 0.25
        elif model == '2D':
            model = build_ep_2sv_model(hyp_ep, zmode=numpy.array("lin"))
            # model.tau0 = 2857.0 # default = 2857.0
            model.tau1 *= scale_time  # default = 0.25
        elif model == '11D':
            model = build_ep_11sv_model(hyp_ep, zmode=numpy.array("lin"))
            # model.tau0 = 10000 # default = 10000
            model.tau1 *= scale_time  # default = 0.25
            model.slope = 0.25
            model.pmode = np.array("z")  # "z","g","z*g", default="cons
        elif model == 'tvb':
            model = build_tvb_model(hyp_ep, zmode=numpy.array("lin"))
            model.tt *= scale_time * 0.25  # default = 1.0
            # model.r = 1.0/2857.0  # default = 1.0 / 2857.0

        # Setup and configure the simulator according to the specific model (and, therefore, hypothesis)
        # Good choices for noise and monitor expressions are made in this helper function
        # It returns name strings for the variables of interest (vois) accordingly
        # monitor_expr and vois have to be list of strings of the same length
        # noise_intensity overwrites the one inside noise_instance if given additionally
        # monitor_period overwrites the one inside monitor_instance if given additionally
        (simulator_instance, sim_settings, vois) = setup_simulation(model, dt, sim_length, monitor_period,
                                                                    monitor_expr=None, monitors_instance=None,
                                                                    noise_instance=None, noise_intensity=10 ** -8,
                                                                    variables_names=None)

        sim = simulator_instance.config_simulation(head, settings=sim_settings)

        #Launch simulation
        ttavg, tavg_data = simulator_instance.launch_simulation(sim, hyp, n_report_blocks=n_report_blocks)
        logger.info("Simulated signal return shape: " + str(tavg_data.shape))
        logger.info("Time: " + str(scale_time*ttavg[0]) + " - " + str(scale_time*ttavg[-1]))
        logger.info("Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max()))

        #Pack results into a dictionary, high pass filter, and compute SEEG
        res = dict()
        for iv in range(len(vois)):
            res[vois[iv]] = numpy.array(tavg_data[:, iv, :, 0], dtype='float32')

        res['seeg'] = []
        if isinstance(sim.model, EpileptorDP2D):
            for i in range(len(projections)):
                res['seeg'+str(i)] = numpy.dot(res['z'], projections[i].T)
        else:
            res['hpf'] = numpy.empty((ttavg.size, hyp.n_regions)).astype(numpy.float32)
            for i in range(hyp.n_regions):
                res['hpf'][:, i] = filter_data(res['lfp'][:, i], hpf_low, hpf_high, hpf_fs)
                res['hpf'] = numpy.array(res['hpf'], dtype='float32')
                res['seeg'] = []
                for i in range(len(projections)):
                    res['seeg' + str(i)] = numpy.dot(res['hpf'], projections[i].T)

        res['time'] = scale_time * numpy.array(ttavg, dtype='float32')

        del ttavg, tavg_data

        #Plot results
        plot_sim_results(model, hyp_ep, head, res, sensorsSEEG)

        #Save results
        res['time_units'] = 'msec'
        savemat(os.path.join(FOLDER_RES, hyp.name + "_ts.mat"), res)
