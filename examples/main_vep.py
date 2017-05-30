"""
Entry point for working with VEP
"""
import os
import warnings
import numpy as np
import copy as cp
from scipy.io import savemat
from tvb_epilepsy.base.constants import SIMULATION_MODE, DATA_MODE, DATA_TVB, DATA_CUSTOM, FOLDER_RES, FOLDER_FIGURES, \
                                        VERY_LARGE_SIZE, X0_DEF, DEF_EIGENVECTORS_NUMBER
from tvb_epilepsy.base.utils import initialize_logger, calculate_projection, set_time_scales, filter_data
from tvb_epilepsy.base.plot_tools import plot_head, plot_hypothesis, plot_sim_results, plot_nullclines_eq
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.custom.read_write import read_ts, write_ts_epi, write_ts_seeg_epi, write_h5_model

if SIMULATION_MODE == "custom":
    from tvb_epilepsy.custom.simulator_custom import setup_simulation
else:
    from tvb_epilepsy.tvb_api.simulator_tvb import setup_simulation


SHOW_FLAG = False
SAVE_FLAG = True
SHOW_FLAG_SIM = False


#The main function...
if __name__ == "__main__":

    logger = initialize_logger(__name__)

    # -------------------------------Reading data-----------------------------------

    if DATA_MODE == 'custom':
        logger.info("Reading from custom")
        data_folder = os.path.join(DATA_CUSTOM, 'Head')
        from tvb_epilepsy.custom.readers_custom import CustomReader
        reader = CustomReader()
        head_connectivity_path = os.path.join(data_folder, 'Connectivity.h5')
    else:
        logger.info("Reading from TVB")
        data_folder = DATA_TVB
        from tvb_epilepsy.tvb_api.readers_tvb import TVBReader
        reader = TVBReader()

    #Read standard  head
    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    #logger.debug("Loaded Head " + str(head))

# # ---------------------------------Hypothalamus pathology addition--------------------------------------------------
# 
# #Modify data folders for this example:
# DATA_HH = '/Users/dionperd/Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH'
# #CON_DATA = 'connectivity_2_hypo.zip'
# CONNECT_DATA = 'connectivity_hypo.zip'
# 
# #Set a special scaling for HH regions, for this example:
# #Set Khyp >=10.0 / hypothesis.n_regions
# Khyp = 15.0 / hypothesis.n_regions
# 
# # Read  connectivity with hypothalamus pathology
# data_folder = os.path.join(DATA_HH, CONNECT_DATA)
# reader = TVBReader()
# HHcon = reader.read_connectivity(data_folder)
# logger.debug("Loaded Connectivity " + str(head.connectivity))
# 
# # Create missing hemispheres:
# nRegions = HHcon.region_labels.shape[0]
# HHcon.hemispheres = np.ones((nRegions,), dtype='int')
# for ii in range(nRegions):
#     if (HHcon.region_labels[ii].find('Right') == -1) and \
#             (HHcon.region_labels[ii].find('-rh-') == -1):  # -1 will be returned when a is not in b
#         HHcon.hemispheres[ii] = 0
# 
# # Adjust pathological connectivity
# w_hyp = np.ones((nRegions, nRegions), dtype='float')
# if Khyp > 10.0 / hypothesis.n_regions:
#     w_hyp[(nRegions - 2):, :] = Khyp
#     w_hyp[:, (nRegions - 2):] = Khyp
# HHcon.normalized_weights = w_hyp * HHcon.normalized_weights
# 
# # Update head with the correct connectivity and sensors' projections
# head.connectivity = HHcon
# 
# # --------------------------------------------------------------------------------------------------------------------

    #Compute projections
    sensorsSEEG=[]
    projections=[]    
    for sensors, projection in head.sensorsSEEG.iteritems():
        if projection is None:
            continue
        else:
            projection = calculate_projection(sensors, head.connectivity)
            head.sensorsSEEG[sensors] = projection
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
    #%     'Brain-Stem'                1 (This should be wrong...)
    #%     'ctx-lh-parahippocampal'   25
    #%     'ctx-rh-entorhinal'        58
    #%     'ctx-rh-parahippocampal'   68
    #%     'Right-Cerebellum-Cortex'   45
    #%     'ctx-rh-fusiform'           59

    # iE = [20]; %JUCH, cj-p04
    # EZ: lLOC: left occipital
    # PZ: lFuG [16]: fuciform,
    # lSPC [38]: superior parietal,
    # lITG [18]: inferior temporal,
    # lIPC [17]: inferior parietal,
    # lPC [30]: pericalcarine,
    # lLgG [22]: Lingual
    # perfect repetition of Tim's result with Tim's connectome
    # perfect repetition of Tim's result with MY connectome as well

    EZ = [20] #[7, 8, 15, 50]  #7, 8, 15, 50
    seizure_indices = np.array([16, 17, 18, 20, 22, 30, 38], dtype=np.int32)
    #seizure_indices = np.array([7, 8, 15, 16, 25, 45, 50, 59, 68], dtype=np.int32)
    #seizure_indices = np.array([50], dtype=np.int32)
    n_eigenvectors = "auto" # len(EZ), "auto", "all" #DEF_EIGENVECTORS_NUMBER

    # hyp_ep = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights, "EP Hypothesis",
    #                     x1eq_mode="optimize")    #"optimize" or "linTaylor"
    # # iE = np.array([1,    3,   7,   8,  15,  16,  25,  50, 88,  89])
    # # E = 0*np.array(iE, dtype=np.float32)
    # # E[7]=0.65
    # # E[2]=0.800
    # # E[4]=0.3775
    # # E[8]=0.425
    # # E[9]=0.900
    # iE = seizure_indices
    # #E = np.array([0.85], dtype=np.float32)
    # E = np.random.normal(0.85, 0.02, (len(iE), ))
    # hyp_ep.configure_hypothesis(ie=iE, e=E, seizure_indices=seizure_indices, n_eigenvectors=n_eigenvectors)
    # #logger.debug(str(hyp_ep))
    # plot_hypothesis(hyp_ep, head.connectivity.region_labels, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
    #                figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
    # write_h5_model(hyp_ep.prepare_for_h5(), folder_name=FOLDER_RES, file_name="hyp_ep.h5")

    hyp_exc = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights, "x0 Hypothesis",
                         x1eq_mode="optimize")
    #hyp_exc.K = 0.0
    hyp_exc.interactive = True
    ix0 = range(hyp_exc.n_regions)
    x0 = (X0_DEF * np.ones((len(ix0),), dtype='float32'))
    #x0[51] = 0.5
    #seizure_indices = np.array([51], dtype=np.int32)
    x0[EZ] = np.random.normal(0.85, 0.02, (len(EZ), ))

    hyp_exc.configure_hypothesis(ix0=ix0, x0=x0, seizure_indices=seizure_indices,
                                 n_eigenvectors=n_eigenvectors)
    # logger.debug(str(hyp_exc))
    plot_hypothesis(hyp_exc, head.connectivity.region_labels,
                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
                    figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
    write_h5_model(hyp_exc.prepare_for_h5(), folder_name=FOLDER_RES, file_name="hyp_exc.h5")

    # hyp_ep_exc = Hypothesis(head.number_of_regions, head.connectivity.normalized_weights, "EP & x0 Hypothesis",
    #                         x1eq_mode="optimize")
    # iE = np.array([3, 7, 8, 15, 16, 25])
    # #E = np.array([0.5], dtype=np.float32)
    # E = np.random.normal(0.85, 0.02, (len(iE), ))
    # # or create the new hypothesis as a deep copy of the previous one:
    # # hyp_ep_exc = cp.deepcopy(hyp_ep)
    # # hyp_ep_exc.name = "EP & x0 Hypothesis"
    # # seizure_indices = np.array([50], dtype=np.int32)
    #
    # #configure first the EP, and then the x0 hypothesis,
    # # hyp_ep_exc.configure_hypothesis(ie=iE, e=E, seizure_indices=seizure_indices)
    # #x0 = (0.5 * np.ones((len(ix0),), dtype='float32'))
    # #seizure_indices = np.array([50, 51], dtype=np.int32)
    # ix0 = [45, 50, 59, 68]
    # x0 = np.random.normal(0.85, 0.02, (len(ix0), ))
    # # hyp_ep_exc.configure_hypothesis(ix0=ix0, x0=x0, seizure_indices=seizure_indices, n_eigenvectors=n_eigenvectors)
    #
    # # or configure them with one line:
    # hyp_ep_exc.configure_hypothesis(iE, E, ix0, x0, seizure_indices, n_eigenvectors=n_eigenvectors)
    #
    # #logger.debug(str(hyp_ep_exc))
    # plot_hypothesis(hyp_ep_exc, head.connectivity.region_labels,
    #                 save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
    #                 figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
    # write_h5_model(hyp_ep_exc.prepare_for_h5(), folder_name=FOLDER_RES, file_name="hyp_ep_exc.h5")

#------------------------------Simulation--------------------------------------

    #We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    # Set time scales (all times should be in msecs and Hz):
    (fs, dt, fsAVG, scale_time, sim_length, monitor_period,
     n_report_blocks, hpf_fs, hpf_low, hpf_high) = set_time_scales(fs=2*4096.0, dt=None, time_length=3000.0,
                                                                   scale_time=2.0, scale_fsavg=2.0,
                                                                   hpf_low=None, hpf_high=None,
                                                                   report_every_n_monitor_steps=10.0)
    # Choose the model and build it on top of the specific hypothesis, adjust parameters:
    if SIMULATION_MODE == "custom":
        model_name = 'CustomEpileptor'
    else:
        model_name = 'EpileptorDP'  # 'EpileptorDP2D', 'EpileptorDPrealistic', "Epileptor"

    #Now simulate and plot for each hypothesis
    hpf_flag = False #Flag to compute and plot high pass filtered SEEG
    for hyp in (hyp_exc, ): #hyp_ep,  hyp_ep_exc

        #Launch simulation
        if SIMULATION_MODE == "custom":
            (simulator_instance, sim_settings, vois) = setup_simulation(hyp, data_folder, dt, sim_length,
                                                                        monitor_period, scale_time=scale_time,
                                                                        noise_intensity=10 ** -8, variables_names=None)
            custom_settings = simulator_instance.config_simulation(settings=sim_settings)
            ttavg, tavg_data, status = simulator_instance.launch_simulation() #return_output=True
            if status and tavg_data is None:
                ttavg, tavg_data = read_ts(simulator_instance.results_path, data="data")

        else:
            # Setup and configure the simulator according to the specific model (and, therefore, hypothesis)
            # Good choices for noise and monitor expressions are made in this helper function
            # It returns name strings for the variables of interest (vois) accordingly
            # monitor_expr and vois have to be list of strings of the same length
            # noise_intensity overwrites the one inside noise_instance if given additionally
            # monitor_period overwrites the one inside monitor_instance if given additionally
            (simulator_instance, sim_settings, vois) = setup_simulation(hyp, head.connectivity, dt, sim_length,
                                                                        monitor_period, model_name=model_name,
                                                                        scale_time=scale_time,
                                                                        noise_instance=None, noise_intensity=10 ** -8,
                                                                        monitor_expressions=None,
                                                                        monitors_instance=None, variables_names=None)

            simTVB, sim_settings = simulator_instance.config_simulation(settings=sim_settings)
            #print "Initial conditions at equilibrium point: ", np.squeeze(simTVB.initial_conditions)
            ttavg, tavg_data, status = simulator_instance.launch_simulation(n_report_blocks=n_report_blocks)
            if not(status):
                warnings.warn("Simulation failed!")
            else:
                tavg_data = tavg_data[:,:,:,0]
            del simTVB

        if status:

            model = simulator_instance.model

            logger.info("\nSimulated signal return shape: " + str(tavg_data.shape))
            logger.info("Time: " + str(scale_time*ttavg[0]) + " - " + str(scale_time*ttavg[-1]))
            logger.info("Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max()))

            simulation_h5_model = simulator_instance.prepare_for_h5(sim_settings)
            write_h5_model(simulation_h5_model, folder_name=FOLDER_RES, file_name=hyp.name+"_sim_settings.h5")

            #Pack results into a dictionary, high pass filter, and compute SEEG
            res = dict()
            time = scale_time * np.array(ttavg, dtype='float32')
            dt = np.min(np.diff(time))
            for iv in range(len(vois)):
                res[vois[iv]] = np.array(tavg_data[:, iv, :], dtype='float32')

            if model._ui_name == "EpileptorDP2D":
                raw_data = np.dstack([res["x1"], res["z"], res["x1"]])
                lfp_data = res["x1"]
                for i in range(len(projections)):
                    res['seeg'+str(i)] = np.dot(res['z'], projections[i].T)
            else:
                if model._ui_name == "CustomEpileptor":
                    raw_data = np.dstack([res["x1"], res["z"], res["x2"]])
                    lfp_data = res["x2"] - res["x1"]
                else:
                    raw_data = np.dstack([res["x1"], res["z"], res["x2"]])
                    lfp_data = res["lfp"]
                for i in range(len(projections)):
                    res['seeg' + str(i)] = np.dot(res['lfp'], projections[i].T)
                    if hpf_flag:
                        for i in range(res['seeg'].shape[0]):
                            res['seeg_hpf'+ str(i)][:, i] = filter_data(res['seeg' + str(i)][:, i], hpf_low, hpf_high, hpf_fs)

            write_ts_epi(raw_data, dt, lfp_data, path=os.path.join(FOLDER_RES, hyp.name + "_ep_ts.h5"))
            del raw_data, lfp_data

            for i in range(len(projections)):
                write_ts_seeg_epi(res['seeg' + str(i)], dt, path=os.path.join(FOLDER_RES, hyp.name + "_ep_ts.h5"))

            res['time'] = time

            del ttavg, tavg_data

            #Plot results
            plot_nullclines_eq(hyp, head.connectivity.region_labels, special_idx=hyp.seizure_indices,
                               model=str(model.nvar) + "d", zmode=model.zmode,
                               figure_name="Nullclines and equilibria", save_flag=SAVE_FLAG,
                               show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES)
            plot_sim_results(model, hyp, head, res, sensorsSEEG, hpf_flag)

            #Save results
            res['time_units'] = 'msec'
            savemat(os.path.join(FOLDER_RES, hyp.name + "_ts.mat"), res)

            del hyp, model, sim_settings, simulator_instance, res

        else:
            warnings.warn("Simulation failed!")
