"""
Entry point for working with VEP
"""
import os
import copy as cp
import numpy as np
from scipy.io import savemat
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.hypothesis import Hypothesis, X0_DEF
from tvb_epilepsy.base.utils import filter_data
from tvb_epilepsy.base.utils import initialize_logger
from tvb_epilepsy.base.plot_tools import plot_head, plot_hypothesis, plot_timeseries, plot_raster, plot_trajectories
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb.simulator import noise
from tvb.datatypes import equations

SHOW_FLAG=True
SAVE_FLAG=True

if __name__ == "__main__":


    logger = initialize_logger(__name__)
    
#-------------------------------Reading data-----------------------------------    

    if DATA_MODE == 'ep':
        logger.info("Reading from EPISENSE")
        data_folder = os.path.join(DATA_EPISENSE, 'Head_TREC') #Head_TREC 'Head_JUNCH'
        from tvb_epilepsy.custom.readers_episense import EpisenseReader
        reader = EpisenseReader()
    else:
        logger.info("Reading from TVB")
        data_folder = DATA_TVB
        from tvb_epilepsy.tvb_api.readers_tvb import TVBReader
        reader = TVBReader()

    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    logger.debug("Loaded Head " + str(head))
    logger.debug("Loaded Connectivity " + str(head.connectivity))

#--------------------------Hypotheis and LSA-----------------------------------

    
    # Next configure two seizure hypothesis
    # Create a new hypothesis for this Head:
    hyp_ep = Hypothesis(head.number_of_regions, head.connectivity.weights, "EP Hypothesis")
    hyp_ep.K = 0.1*hyp_ep.K
    print 'Configure the hypothesis...'
    print '...by defining the indices...'
    print 'iE:'
    iE = np.array([7, 50])
    print iE
    print '...the epileptogenicity values of the regions of interest...'
    print 'E:'
    E = np.array([0.5,0.8], dtype=np.float32)
    print E
    print '...as well as the regions of reference (e.g., the regions the seizure is starting from):'
    seizure_indices = np.array([7, 50], dtype=np.int32)
    print seizure_indices
    
    hyp_ep.configure_e_hypothesis(iE, E, seizure_indices)
    logger.debug(str(hyp_ep))
    print 'Printing a summary of this hypothesis, together with the result of the Linear Stability Analysis:'
    print hyp_ep

    # Create a new hypothesis for this Head:
    #hyp_exc = Hypothesis(head.number_of_regions, head.connectivity.weights, "Excitability Hypothesis")
    hyp_exc = cp.deepcopy(hyp_ep)
    print 'Configure the hypothesis...'
    print '...by defining the indices of all regions assumed to neither generate nor propagate the seizure...'
    ii = np.array(range(head.number_of_regions), dtype=np.int32)
    ix0 = np.delete(ii, iE)
    print 'ix0:'
    print ix0
    print '...and setting their excitability values x0 equal to the default "healthy" one:'
    print 'x0:' 
    print X0_DEF
    
    hyp_exc.configure_x0_hypothesis(ix0, X0_DEF, seizure_indices)
    logger.debug(str(hyp_exc))
    print 'Printing a summary of this hypothesis, together with the result of the Linear Stability Analysis:'
    print hyp_exc

#------------------------------Simulation--------------------------------------

    print "Getting SEEG sensors and projections from head"
    sensorsSEEG=[]
    projections=[]    
    for sensors, projection in head.sensorsSEEG.iteritems():
        if projection is None:
            continue
        else:
            sensorsSEEG.append(sensors)
            projections.append(projection) 
            
    print 'Select TVB or Episense simulator and the desired Epileptor model, as well as set simulations settings'
    SIMULATION_MODE ='tvb_epilepsy' #'ep'
    print 'Selected simulator: '+SIMULATION_MODE

    MODEL = '11v' #'6v','2v','11v','tvb_epilepsy'
    print 'Selected model: '+MODEL
    
    #Monitor adjusted to the model    
    nvar = {'tvb_epilepsy':6,
            '6v':6,
            '2v':2,
            '11v':11}
    monitor_expr = {'tvb_epilepsy':['x2 - x1','x1', 'y1', 'z', 'x2', 'y2', 'g' ],
                    '6v':["y3 - y0","y0", "y1", "y2", "y3", "y4", "y5"],
                    '2v':["y0", "y1"],
                    '11v':["y3 - y0","y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]}          
    #print "Expressions monitored: "
    #for expr in monitor_expr[MODEL]:
    #    print expr
    
    #Time scales:    
    fs = 2*1024.0
    dt = 1000.0/fs
    print "Integration (sampling) frequency: "+str(fs)
    print "...and time step: "+ str(dt) 
    fsAVG = 512.0 #fs/10
    fsSEEG = fsAVG
    print "Monitor (moving average) frequency: "+str(fsAVG)
    print "...SEEG frequency: "+ str(fsSEEG) 
    
    
    #Noise configuration
    #                                  x1  y1   z     x2   y2   g 
    noise_intensity = {'tvb_epilepsy':np.array([0., 0., 5e-8, 0.0, 5e-8, 0.]),
                        '6v':np.array([ 0., 0., 5e-8, 0.0, 5e-8, 0.]),
                        '2v':np.array([ 0.,     5e-8]) ,
                        '11v':np.array([0., 0., 1e-5, 0.0, 1e-5, 0., 
    #                                   #x0   slope Iext1 Iext2 K 
                                        1e-7, 1e-2, 1e-7, 1e-2, 1e-8])} 
    #Preconfigured noise
    eq=equations.Linear(parameters={"a": 0.0, "b": 1.0}) #default = a*y+b
    noise=noise.Multiplicative(nsig=noise_intensity[MODEL], b=eq,
                               random_stream=np.random.RandomState(seed=NOISE_SEED)) #define ntau for colored noise: ntau = 10, 
    noise_shape = noise.nsig.shape                    
    #Colored noise:                
    #noise.configure_coloured(dt=dt,shape=noise_shape)
    #White noise:               
    noise.configure_white(dt=dt)
    #Get sensors and projections:
    print "Summary of noise settings: "
    print noise
         

    settings = SimulationSettings(length=10000,integration_step=dt, monitor_sampling_period=fs/fsSEEG*dt,
                                  noise_preconfig=noise, monitor_expr=monitor_expr[MODEL]) #noise_intensity=noise_intensity, 
    #print "Summary of simulation settings: "
    #print settings
    
    #Build the model

    build_model = {'tvb_epilepsy':build_tvb_model,
                    '6v':build_ep_6sv_model,
                    '2v':build_ep_2sv_model,
                    '11v':build_ep_11sv_model}
    prepare_model = {'tvb_epilepsy':prepare_for_tvb_model,
                     '6v':prepare_for_6sv_model,
                     '2v':prepare_for_2sv_model,
                     '11v':prepare_for_11sv_model}        
    # Choosing the model:           epileptor model,      history
    simulator_instance = SimulatorTVB(build_model[MODEL], prepare_model[MODEL])
    print "Simulator instance: "
    print settings  
    
    #Prepare output
    save_dict = dict()
    seeg=dict()
    x1=dict()
    z=dict()
    if MODEL!='2v':
        lfp=dict()
        y1=dict()
        x2=dict()
        y2=dict()
        g=dict()
        hpf=dict()   
    
        if MODEL=='11v':       
            x0ts=dict()
            slopeTS=dict()
            Iext1ts=dict()
            Iext2ts=dict()
            Kts=dict()  

    
    hyps = {'ep':hyp_ep,'exc':hyp_exc}
    
    for hyp in ('ep','exc'):
        
        print "Configuring simulation for hypothesis "+hyps[hyp].name
        sim= simulator_instance.config_simulation(hyps[hyp], head, settings)
        print "Summary of simulation configuration: "
        print sim
        print "Further specifying parameters of the model"
        if MODEL=='tvb_epilepsy':
                sim.model.tt = 0.25*sim.model.tt
                sim.model.r = 0.001
        else:
            #sim.model.zmode = np.array("sig")
            sim.model.tau0 = 10000.0
            sim.model.tau1 = 0.25*sim.model.tau1
            if MODEL=='11v':    
                sim.model.pmode=np.array("z") #"z","g","z*g", default="const"
                sim.model.slope = 0.25
                
        print "Integration..."        
        ttavg, tavg_data = simulator_instance.launch_simulation(sim,hyps[hyp])
        print "Simulated signal return shape: " + str(tavg_data.shape)
        print "Time: " + str(ttavg[0]) + " - " + str(ttavg[-1])
        print "Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max())
        print "Simulated signal return shape: " + str(tavg_data.shape)
        print "Time: " + str(ttavg[0]) + " - " + str(ttavg[-1])
        print "Values: " + str(tavg_data.min()) + " - " + str(tavg_data.max())
    
        print "Unpacking state variables, calculating observables (lfp, hpf, seeg) and saving results..."        
        #Unpacking state variables, calculating observables (lfp, hpf, seeg) and saving results...
        #Time:
        ttavg = np.array(ttavg,dtype='float32')
    
        if MODEL=='2v':
            x1[hyp] = np.array(tavg_data[:, 0, :, 0],dtype='float32')
            z[hyp] = np.array(tavg_data[:, 1, :, 0],dtype='float32')
            save_dict = {'x1':x1[hyp],'z':z[hyp],'time_in_ms':ttavg}
    
            seeg[hyp]=[]                    
            for i in range(len(projections)):
                seeg_i = np.dot(x1[hyp],projections[i].T)
                seeg.append(seeg_i)
                save_dict['seeg'+str(i)]=seeg_i
            savemat(os.path.join(FOLDER_RES, hyps[hyp].name+"_ts.mat"),save_dict) 
            del save_dict
            #del tavg_data
    
        else:
            lfp[hyp] = np.array(tavg_data[:, 0, :, 0],dtype='float32')
            z[hyp] = np.array(tavg_data[:, 3, :, 0],dtype='float32')
            x1[hyp] = np.array(tavg_data[:, 1, :, 0],dtype='float32')
            y1[hyp] = np.array(tavg_data[:, 2, :, 0],dtype='float32')
            x2[hyp] = np.array(tavg_data[:, 4, :, 0],dtype='float32')
            y2[hyp] = np.array(tavg_data[:, 5, :, 0],dtype='float32')
            g[hyp] = np.array(tavg_data[:, 6, :, 0],dtype='float32')
            hpf[hyp] = np.empty((ttavg.size,hyps[hyp].n_regions)).astype(np.float32)
            for i in range(hyps[hyp].n_regions):
                hpf[hyp][:,i] = filter_data(lfp[hyp][:,i], 10, 250, fsAVG)#.astype(np.float32) #.transpose()
            hpf[hyp]=np.array(hpf[hyp],dtype='float32')   
            hpf[hyp] = filter_data(lfp[hyp], fsAVG/30, fsAVG/3, fsAVG)
            save_dict = {'lfp':lfp[hyp],'hpf':hpf[hyp],'x1':x1[hyp],'y1':y1[hyp],'z':z[hyp],
                         'x2':x2[hyp],'y2':y2[hyp],'g':g[hyp],'time_in_ms':ttavg}
    
            seeg[hyp]=[]                    
            for i in range(len(projections)):
                seeg_i = np.dot(hpf[hyp],projections[i].T)
                seeg[hyp].append(seeg_i)
                save_dict['seeg'+str(i)]=seeg_i
    
            if MODEL=='11v':       
                x0ts[hyp]=np.array(tavg_data[:, 7, :, 0],dtype='float32')
                slopeTS[hyp]=np.array(tavg_data[:, 8, :, 0],dtype='float32')
                Iext1ts[hyp]=np.array(tavg_data[:, 9, :, 0],dtype='float32')
                Iext2ts[hyp]=np.array(tavg_data[:, 10, :, 0],dtype='float32')
                Kts[hyp]=np.array(tavg_data[:, 11, :, 0],dtype='float32')
                save_dict['x0']=x0ts[hyp]
                save_dict['slope']=slopeTS[hyp]
                save_dict['Iext1']=Iext1ts[hyp]
                save_dict['Iext2']=Iext2ts[hyp]
                save_dict['K']=Kts[hyp]
    
        savemat(os.path.join(FOLDER_RES, hyps[hyp].name+"_ts.mat"),save_dict)  
        del save_dict
        #del tavg_data
         
    #-------------------------------Plotting---------------------------------------

    # Figures related settings:
    VERY_LARGE_SIZE = (30, 15)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    FOLDER_FIGURES = os.path.join(FOLDER_VEP, 'figures')
    FIG_FORMAT = 'eps'
    SAVE_FLAG = True
    SHOW_FLAG = True
    
    print 'Plotting the Head connectivity and statistics'
    plot_head(head, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, 
                  figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
    
    for hyp in ('ep','exc'):
    
        print 'Plotting the epileptogenicity hypothesis '+hyps[hyp].name
        plot_hypothesis(hyps[hyp], head.connectivity.region_labels,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, 
                            figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE)
    
        print '...and the respective simulated time series and trajectories:'
        #Plotting
        if MODEL=='2v':
            plot_timeseries(ttavg, {'x1(t)': x1[hyp], 'z(t)': z[hyp]},
                            seizure_indices, title=" Simulated time series for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_trajectories({'x1(t)': x1[hyp], 'z(t)': z[hyp]},
                            seizure_indices, title=" Simulated x1-z trajectories for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)            
                  
        else:
            plot_timeseries(ttavg, {'LFP = x2(t) - x1(t)': lfp[hyp], 'z(t)': z[hyp], 'HPF LFP': hpf[hyp]},
                            seizure_indices, title=" Simulated TAVG for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_timeseries(ttavg, {'x1(t)': x1[hyp], 'y1(t)': y1[hyp], 'z(t)': z[hyp]},
                                seizure_indices, title=" Simulated pop1-z for " + hyps[hyp].name,
                                save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_timeseries(ttavg, {'x2(t)': x2[hyp], 'y2(t)': y2[hyp], 'g(t)': g[hyp]},seizure_indices, 
                                title=" Simulated pop2-g for " + hyps[hyp].name,
                                save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE) 
            plot_raster(ttavg[100:], {'hpf': hpf[hyp][100:,:]}, seizure_indices,
                                title=" Simulated hfp"+str(i)+" rasterplot for " + hyps[hyp].name,offset=10.0,
                                save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_trajectories({'x1(t)': x1[hyp], 'y1(t)': y1[hyp],'z(t)': z[hyp]},
                            seizure_indices, title=" Simulated x1-y1-z trajectories for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
            plot_trajectories({'x2(t)': x2[hyp], 'y2(t)': y2[hyp],'z(t)': z[hyp]},
                            seizure_indices, title=" Simulated x2-y2-z trajectories for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE) 
            plot_trajectories({'LFP = x2(t) - x1(t)': lfp[hyp], 'z(t)': z[hyp]},
                            seizure_indices, title=" Simulated lfp-z trajectories for " + hyps[hyp].name,
                            save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                            labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)             
                    
            if MODEL=='11v':       
                plot_timeseries(ttavg, {'1/(1+exp(-10(z-3.03))': 1/(1+np.exp(-10*(z[hyp]-3.03))), 'slope': slopeTS[hyp], 'Iext2': Iext2ts[hyp]},
                                    seizure_indices, title=" Simulated controlled parameters for " + hyps[hyp].name,
                                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                    labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)               
                plot_timeseries(ttavg, {'x0': x0ts[hyp], 'Iext1': Iext1ts[hyp],'K': Kts[hyp]},
                                    seizure_indices, title=" Simulated parameters for " + hyps[hyp].name,
                                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                    labels = head.connectivity.region_labels, figsize=VERY_LARGE_SIZE) 
    
        for i in range(len(projections)):
        #            plot_timeseries(ttavg[100:], {'SEEG': seeg[hyp][i][100:,:]}, title=" Simulated SEEG"+str(i)+" for " + hyps[hyp].name,
        #                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
        #                        labels = sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
            plot_raster(ttavg[100:], {'SEEG': seeg[hyp][i][100:,:]}, title=" Simulated SEEG"+str(i)+" rasterplot for " + hyps[hyp].name,
                                offset=10.0,save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, 
                                labels = sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE) 