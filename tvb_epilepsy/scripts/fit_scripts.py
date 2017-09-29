import os
import time
import numpy as np
import pystan as ps
from scipy.io import savemat, loadmat
import pickle

from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, X0_DEF, X0_CR_DEF, VOIS, X0_DEF, E_DEF, TVB, DATA_MODE, \
                                        SIMULATION_MODE
from tvb_epilepsy.base.configurations import FOLDER_RES, DATA_CUSTOM, STATISTICAL_MODELS_PATH, FOLDER_VEP_HOME
from tvb_epilepsy.base.utils import warning, raise_not_implemented_error, initialize_logger
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_z
from tvb_epilepsy.service.sampling_service import gamma_from_mu_std, gamma_to_mu_std
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDPrealistic, EpileptorDP
from tvb_epilepsy.base.plot_utils import plot_sim_results, plot_fit_results
from tvb_epilepsy.scripts.simulation_scripts import set_time_scales, prepare_vois_ts_dict, \
                                                    compute_seeg_and_write_ts_h5_file

from tvb.simulator.models import Epileptor

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

if SIMULATION_MODE is TVB:
    from tvb_epilepsy.scripts.simulation_scripts import setup_TVB_simulation_from_model_configuration \
        as setup_simulation_from_model_configuration
else:
    from tvb_epilepsy.scripts.simulation_scripts import setup_custom_simulation_from_model_configuration \
        as setup_simulation_from_model_configuration


logger = initialize_logger(__name__)


def compile_model(model_stan_code_path=os.path.join(STATISTICAL_MODELS_PATH, "vep_autoregress.stan"), **kwargs):
    tic = time.time()
    logger.info("Compiling model...")
    model = ps.StanModel(file=model_stan_code_path, model_name=kwargs.get("model_name", 'vep_epileptor2D_autoregress'))
    logger.info(str(time.time() - tic) + ' sec required to compile')
    return model


def prepare_data_for_fitting(model_configuration, hypothesis, fs, sim_ts, dynamic_model=None, noise_intensity=None,
                             active_regions=None, active_regions_th=0.1, observation_model=3, mixing=None, **kwargs):

    logger.info("Constructing data dictionary...")
    active_regions_flag = np.zeros((hypothesis.number_of_regions, ), dtype="i")

    if active_regions is None:
        if len(hypothesis.propagation_strengths) > 0:
            active_regions = np.where(hypothesis.propagation_strengths / np.max(hypothesis.propagation_strengths)
                                      > active_regions_th)[0]
        else:
            raise_not_implemented_error("There is no other way of automatic selection of " +
                                        "active regions implemented yet!")

    active_regions_flag[active_regions] = 1
    n_active_regions = len(active_regions)

    if isinstance(dynamic_model, (Epileptor, EpileptorModel)):
        tau1_def = np.mean(1.0 / dynamic_model.r)
        tau0_def = np.mean(dynamic_model.tt)
    elif isinstance(dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
        tau1_def = np.mean(dynamic_model.tau1)
        tau0_def = np.mean(dynamic_model.tau0)
    else:
        tau1_def = 0.2
        tau0_def = 40000

    # Gamma distributions' parameters
    # visualize gamma distributions here: http://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html
    tau1_mu = tau1_def
    tau1 = gamma_from_mu_std(kwargs.get("tau1_mu", tau1_mu), kwargs.get("tau1_std", 3*tau1_mu))
    tau0_mu = tau0_def
    tau0 = gamma_from_mu_std(kwargs.get("tau0_mu", tau0_mu), kwargs.get("tau0_std", 3*10000.0))
    K_def = np.mean(model_configuration.K)
    K = gamma_from_mu_std(kwargs.get("K_mu", K_def),
                          kwargs.get("K_std", 10*K_def))
    # zero effective connectivity:
    conn0 = gamma_from_mu_std(kwargs.get("conn0_mu", 0.001), kwargs.get("conn0_std", 0.001))
    if noise_intensity is None:
        sig_mu = np.mean(model_noise_intensity_dict["EpileptorDP2D"])
    else:
        sig_mu = noise_intensity
    sig = gamma_from_mu_std(kwargs.get("sig_mu", sig_mu), kwargs.get("sig_std", 3*sig_mu))
    sig_eq_mu = (X1_EQ_CR_DEF - X1_DEF) / 3.0
    sig_eq_std = 3*sig_eq_mu
    sig_eq = gamma_from_mu_std(kwargs.get("sig_eq_mu", sig_eq_mu), kwargs.get("sig_eq_std", sig_eq_std))
    sig_init_mu = sig_eq_mu
    sig_init_std = sig_init_mu
    sig_init = gamma_from_mu_std(kwargs.get("sig_init_mu", sig_init_mu), kwargs.get("sig_init_std", sig_init_std))

    # signals = (sim_ts["x1"][:, active_regions].T - np.expand_dims(model_configuration.x1EQ[active_regions], 1)).T + \
    #           (sim_ts["z"][:, active_regions].T - np.expand_dims(model_configuration.zEQ[active_regions], 1)).T
    # signals = signals / 2.75

    signals = (sim_ts["x1"][:, active_regions].T - np.expand_dims(model_configuration.x1EQ[active_regions], 1)).T / 2.0

    if mixing is None:
        if observation_model == 2:
            mixing = np.random.rand(n_active_regions, n_active_regions)
            for ii in range(n_active_regions):
                mixing[ii, :] = mixing[ii, :]/np.sum(mixing[ii, :])
        else:
            observation_model = 3
            mixing = np.eye(n_active_regions)
    signals = (np.dot(mixing, signals.T)).T

    data = {"n_regions": hypothesis.number_of_regions,
            "n_active_regions": n_active_regions,
            "n_nonactive_regions": hypothesis.number_of_regions-n_active_regions,
            "active_regions_flag": active_regions_flag,
            "n_time": signals.shape[0],
            "n_signals": signals.shape[1],
            "x0_nonactive": model_configuration.x0[~active_regions_flag.astype("bool")],
            "x1eq0": model_configuration.x1EQ,
            "zeq0": model_configuration.zEQ,
            "x1eq_lo": kwargs.get("x1eq_lo", -2.0),
            "x1eq_hi": kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
            "x1init_lo": kwargs.get("x1init_lo", -2.0),
            "x1init_hi": kwargs.get("x1init_hi", -1.0),
            "x1_lo": kwargs.get("x1_lo", -2.5),
            "x1_hi": kwargs.get("x1_hi", 1.5),
            "z_lo": kwargs.get("z_lo", 2.0),
            "z_hi": kwargs.get("z_hi", 5.0),
            "tau1_lo": kwargs.get("tau1_lo", tau1_mu / 10.0),
            "tau1_hi": kwargs.get("tau1_hi", np.min([10 * tau1_mu, 1.0])),
            "tau0_lo": kwargs.get("tau0_lo", tau0_mu/10.0),
            "tau0_hi": kwargs.get("tau0_hi", np.min([10 * tau1_mu, 10.0])),
            "tau1_a": kwargs.get("tau1_a", tau1["alpha"]),
            "tau1_b": kwargs.get("tau1_b", tau1["beta"]),
            "tau0_a": kwargs.get("tau0_a", tau0["alpha"]),
            "tau0_b": kwargs.get("tau0_b", tau0["beta"]),
            "SC": model_configuration.connectivity_matrix,
            "SC_sig": kwargs.get("SC_sig", 0.1),
            "K_lo": kwargs.get("K_lo", K_def / 10.0),
            "K_hi": kwargs.get("K_hi", 30.0 * K_def),
            "K_a": kwargs.get("K_a", K["alpha"]),
            "K_b": kwargs.get("K_b", K["beta"]),
            "gamma0": kwargs.get("gamma0", np.array([conn0["alpha"], conn0["beta"]])),
            "dt": 1000.0 / fs,
            "sig_hi": kwargs.get("sig_hi", 2*sig_mu),
            "sig_a": kwargs.get("sig_a", sig["alpha"]),
            "sig_b": kwargs.get("sig_b", sig["beta"]),
            "sig_eq_hi": kwargs.get("sig_eq_hi", 3*sig_eq_std),
            "sig_eq_a": kwargs.get("sig_eq_a", sig_eq["alpha"]),
            "sig_eq_b": kwargs.get("sig_eq_b", sig_eq["beta"]),
            "sig_init_hi": kwargs.get("sig_init_hi", 3 * sig_init_std),
            "sig_init_a": kwargs.get("sig_init_a", sig_init["alpha"]),
            "sig_init_b": kwargs.get("sig_init_b", sig_init["beta"]),
            "observation_model": observation_model,
            "signals": signals,
            "mixing": mixing,
            "eps_hi": kwargs.get("eps_hi", (np.max(signals.flatten()) - np.min(signals.flatten()) / 100.0)),
            "eps_x0": kwargs.get("eps_x0", 0.1),
    }

    for p in ["a", "b", "d", "yc", "Iext1", "slope"]:

        temp = getattr(model_configuration, p)
        if isinstance(temp, (np.ndarray, list)):
            if np.all(temp[0], np.array(temp)):
                temp = temp[0]
            else:
                raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                            " for parameter " + p + " are not implemented yet!")
        data.update({p: temp})

    zeq_lo = calc_eq_z(data["x1eq_hi"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                       b=data["b"], d=data["d"])
    zeq_hi = calc_eq_z(data["x1eq_lo"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                       b=data["b"], d=data["d"])
    data.update({"zeq_lo": kwargs.get("zeq_lo", zeq_lo),
                 "zeq_hi": kwargs.get("zeq_hi", zeq_hi)})
    data.update({"zinit_lo": kwargs.get("zinit_lo", zeq_lo - sig_init_std),
                 "zinit_hi": kwargs.get("zinit_hi", zeq_hi + sig_init_std)})

    x0cr, rx0 = calc_x0cr_r(data["yc"], data["Iext1"], data["a"], data["b"], data["d"], zmode=np.array("lin"),
                            x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                            shape=None, calc_mode="non_symbol")

    data.update({"x0cr": x0cr, "rx0": rx0})
    logger.info("data dictionary completed with " + str(len(data)) + " fields:\n" + str(data.keys()))

    return data, tau0_def, tau1_def


def prepare_data_for_fitting_vep_original(model_configuration, hypothesis, fs, sim_ts, dynamic_model=None,
                                          noise_intensity=None, active_regions=None, active_regions_th=0.1,
                                          observation_model=3, mixing=None, **kwargs):

    p, tau0_def, tau1_def = prepare_data_for_fitting(model_configuration, hypothesis, fs, sim_ts, dynamic_model,
                                                     noise_intensity, active_regions, active_regions_th,
                                                     observation_model, mixing, **kwargs)

    active_regions = np.where(p["active_regions_flag"])[0]
    non_active_regions = np.where(1-p["active_regions_flag"])[0]

    data = {"observation_model": p["observation_model"]}
    data.update({"nn": p["n_active_regions"]})
    data.update({"nt": p["n_time"]})
    data.update({"ns": p["n_signals"]})
    data.update({"I1": p["Iext1"]})
    data.update({"tau0": tau0_def})
    data.update({"dt": p["dt"]})
    data.update({"xeq": p["x1eq0"][active_regions]})
    data.update({"zeq": p["zeq0"][active_regions]})
    data.update({"gain": p["mixing"]})
    data.update({"signals": p["signals"]})
    data.update({"Ic": np.sum(p["SC"][active_regions][:, non_active_regions], axis=1)})
    data.update({"SC": p["SC"][active_regions][:, active_regions]})
    data.update({"SC_var": p["SC_sig"]})
    data.update({"K_lo": p["K_lo"]})
    data.update({"K_hi": p["K_hi"]})
    K = {"alpha": p["K_a"], "beta": p["K_b"]}
    K_mu, K_std = gamma_to_mu_std(K)
    K = gamma_from_mu_std(K_mu, K_std)
    data.update({"K_u": K["k"]})
    data.update({"K_v": K["theta"]})
    data.update({"x0_lo": -4.0})
    data.update({"x0_hi": -1.0})
    data.update({"eps_hi": p["eps_hi"]})
    data.update({"sig_hi": p["sig_hi"]})
    data.update({"zlim": np.array([p["z_lo"], p["z_hi"]])})
    data.update({"tt": tau1_def})
    data.update({"amp": 1.0})
    data.update({"offset": 0.0})

    logger.info("data dictionary completed with " + str(len(data)) + " fields:\n" + str(data.keys()))

    return data, active_regions


def stanfit_model(model, data, mode="sampling", **kwargs):

    logger.info("Model fitting with " + mode + "...")
    fit = getattr(model, mode)(data=data, **kwargs)

    if mode is "optimizing":
        return fit, None
    else:
        logger.info("Extracting estimates...")
        if mode is "sampling":
            est = fit.extract(permuted=True)
        elif mode is "vb":
            est = read_vb_results(fit)
        return est, fit



def read_vb_results(fit):
    est = {}
    for ip, p in enumerate(fit['sampler_param_names']):
        p_split = p.split('.')
        p_name = p_split.pop(0)
        p_name_samples = p_name + "_s"
        if est.get(p_name) is None:
            est.update({p_name_samples: []})
            est.update({p_name: []})
        if len(p_split) == 0:
            # scalar parameters
            est[p_name_samples] = fit["sampler_params"][ip]
            est[p_name] = fit["mean_pars"][ip]
        else:
            if len(p_split) == 1:
                # vector parameters
                est[p_name_samples].append(fit["sampler_params"][ip])
                est[p_name].append(fit["mean_pars"][ip])
            else:
                ii = int(p_split.pop(0)) - 1
                if len(p_split) == 0:
                    # 2D matrix parameters
                    if len(est[p_name]) < ii + 1:
                        est[p_name_samples].append([fit["sampler_params"][ip]])
                        est[p_name].append([fit["mean_pars"][ip]])
                    else:
                        est[p_name_samples][ii].append(fit["sampler_params"][ip])
                        est[p_name][ii].append(fit["mean_pars"][ip])
                else:
                    if len(est[p_name]) < ii + 1:
                        est[p_name_samples].append([])
                        est[p_name].append([])
                    jj = int(p_split.pop(0)) - 1
                    if len(p_split) == 0:
                        # 3D matrix parameters
                        if len(est[p_name][ii]) < jj + 1:
                            est[p_name_samples][ii].append([fit["sampler_params"][ip]])
                            est[p_name][ii].append([fit["mean_pars"][ip]])
                        else:
                            if len(est[p_name][ii]) < jj + 1:
                                est[p_name_samples][ii].append([])
                                est[p_name][ii].append([])
                            est[p_name_samples][ii][jj].append(fit["sampler_params"][ip])
                            est[p_name][ii][jj].append(fit["mean_pars"][ip])
                    else:
                        raise_not_implemented_error("Extracting of parameters of more than 3 dimensions is not " +
                                    "implemented yet for vb!", logger)
    for key in est.keys():
        if isinstance(est[key], list):
            est[key] = np.squeeze(np.array(est[key]))
    return est


def main_fit_sim_hyplsa():

    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    # head.plot()

    # --------------------------Hypothesis definition-----------------------------------

    n_samples = 100

    # # Manual definition of hypothesis...:
    # x0_indices = [20]
    # x0_values = [0.9]
    # e_indices = [70]
    # e_values = [0.9]
    # disease_values = x0_values + e_values
    # disease_indices = x0_indices + e_indices

    # ...or reading a custom file:
    ep_name = "ep_test1"
    # FOLDER_RES = os.path.join(data_folder, ep_name)
    from tvb_epilepsy.custom.readers_custom import CustomReader

    if not isinstance(reader, CustomReader):
        reader = CustomReader()
    disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    disease_indices, = np.where(disease_values > np.min([X0_DEF, E_DEF]))
    disease_values = disease_values[disease_indices]
    if disease_values.size > 1:
        inds_split = np.ceil(disease_values.size * 1.0 / 2).astype("int")
        x0_indices = disease_indices[:inds_split].tolist()
        e_indices = disease_indices[inds_split:].tolist()
        x0_values = disease_values[:inds_split].tolist()
        e_values = disease_values[inds_split:].tolist()
    else:
        x0_indices = disease_indices.tolist()
        x0_values = disease_values.tolist()
        e_indices = []
        e_values = []
    disease_indices = list(disease_indices)

    n_x0 = len(x0_indices)
    n_e = len(e_indices)
    n_disease = len(disease_indices)
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)

    # This is an example of Excitability Hypothesis:
    hyp_x0 = DiseaseHypothesis(head.connectivity.number_of_regions,
                               excitability_hypothesis={tuple(disease_indices): disease_values},
                               epileptogenicity_hypothesis={}, connectivity_hypothesis={})

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                              excitability_hypothesis={},
                              epileptogenicity_hypothesis={tuple(disease_indices): disease_values},
                              connectivity_hypothesis={})

    if len(e_indices) > 0:
        # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis:
        hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                                     excitability_hypothesis={tuple(x0_indices): x0_values},
                                     epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                     connectivity_hypothesis={})
        hypotheses = (hyp_x0, hyp_E, hyp_x0_E)

    else:
        hypotheses = (hyp_x0, hyp_E)

    # --------------------------Simulation preparations-----------------------------------
    tau1 = 0.2
    # TODO: maybe use a custom Monitor class
    fs = 10*2048.0*(2*tau1)  # this is the simulation sampling rate that is necessary for the simulation to be stable
    time_length = 50.0 / tau1  # msecs, the final output nominal time length of the simulation
    report_every_n_monitor_steps = 100.0
    (dt, fsAVG, sim_length, monitor_period, n_report_blocks) = \
        set_time_scales(fs=fs, time_length=time_length, scale_fsavg=1,
                        report_every_n_monitor_steps=report_every_n_monitor_steps)

    # Choose model
    # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
    # EpileptorDP: similar to the TVB Epileptor + optional variations,
    # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
    # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
    #      -x0, Iext1, Iext2, slope and K become noisy state variables,
    #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
    #      -multiplicative correlated noise is also used
    # Optional variations:
    zmode = "lin"  # by default, or "sig" for the sigmoidal expression for the slow z variable in Proix et al. 2014
    pmode = "z"  # by default, "g" or "z*g" for the feedback coupling to Iext2 and slope for EpileptorDPrealistic

    model_name = "EpileptorDP2D"
    if model_name is "EpileptorDP2D":
        spectral_raster_plot = False
        trajectories_plot = True
    else:
        spectral_raster_plot = "lfp"
        trajectories_plot = False
    # We don't want any time delays for the moment
    # head.connectivity.tract_lengths *= TIME_DELAYS_FLAG

    # --------------------------Hypothesis and LSA-----------------------------------

    for hyp in (hyp_x0, ): #hypotheses:

        logger.info("\n\nRunning hypothesis: " + hyp.name)

        # hyp.write_to_h5(FOLDER_RES, hyp.name + ".h5")

        logger.info("\n\nCreating model configuration...")
        model_configuration_service = ModelConfigurationService(hyp.number_of_regions)
        # model_configuration_service.write_to_h5(FOLDER_RES, hyp.name + "_model_config_service.h5")

        if hyp.type == "Epileptogenicity":
            model_configuration = model_configuration_service. \
                configure_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
        else:
            model_configuration = model_configuration_service. \
                configure_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
        # model_configuration.write_to_h5(FOLDER_RES, hyp.name + "_ModelConfig.h5")

        # Plot nullclines and equilibria of model configuration
        # model_configuration_service.plot_nullclines_eq(model_configuration, head.connectivity.region_labels,
        #                                                special_idx=disease_indices, model="6d", zmode="lin",
        #                                                figure_name=hyp.name + "_Nullclines and equilibria")

        logger.info("\n\nRunning LSA...")
        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)

        # lsa_hypothesis.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")
        # lsa_service.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")
        #
        # lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels, None)

        # ------------------------------Model code--------------------------------------
        # Compile or load model:
        model_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_stan_model.pkl")
        if os.path.isfile(model_file):
            model = pickle.load(open(model_file, 'rb'))
        else:
            model = compile_model(model_stan_code_path=os.path.join(STATISTICAL_MODELS_PATH, "vep_original_DP.stan"))
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        # ------------------------------Simulation--------------------------------------
        logger.info("\n\nConfiguring simulation...")
        noise_intensity = 10 ** -4
        sim = setup_simulation_from_model_configuration(model_configuration, head.connectivity, dt,
                                                        sim_length, monitor_period, model_name,
                                                        zmode=np.array(zmode), pmode=np.array(pmode),
                                                        noise_instance=None, noise_intensity=noise_intensity,
                                                        monitor_expressions=None)
        sim.model.tau1 = tau1
        sim.model.tau0 = 100.0

        # Integrator and initial conditions initialization.
        # By default initial condition is set right on the equilibrium point.
        sim.config_simulation(initial_conditions=None)

        # convert_to_h5_model(sim.model).write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_sim_model.h5")

        ts_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts.mat")
        if os.path.isfile(ts_file):
            logger.info("\n\nLoading previously simulated time series...")
            vois_ts_dict = loadmat(ts_file)
        else:
            logger.info("\n\nSimulating...")
            ttavg, tavg_data, status = sim.launch_simulation(n_report_blocks)

            # convert_to_h5_model(sim.simulation_settings).write_to_h5(FOLDER_RES,
            #                                                          lsa_hypothesis.name + "_sim_settings.h5")

            if not status:
                warning("\nSimulation failed!")

            else:

                time = np.array(ttavg, dtype='float32').flatten()

                output_sampling_time = np.mean(np.diff(time))
                tavg_data = tavg_data[:, :, :, 0]

                logger.info("\n\nSimulated signal return shape: %s", tavg_data.shape)
                logger.info("Time: %s - %s", time[0], time[-1])
                logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())

                # Variables of interest in a dictionary:
                vois_ts_dict = prepare_vois_ts_dict(VOIS[model_name], tavg_data)
                vois_ts_dict['time'] = time
                vois_ts_dict['time_units'] = 'msec'

                vois_ts_dict=compute_seeg_and_write_ts_h5_file(FOLDER_RES, lsa_hypothesis.name + "_ts.h5", sim.model,
                                                               vois_ts_dict, output_sampling_time, time_length,
                                                               hpf_flag=True, hpf_low=10.0, hpf_high=512.0,
                                                               sensor_dicts_list=[head.sensorsSEEG])

                # Plot results
                plot_sim_results(sim.model, lsa_hypothesis.propagation_indices, lsa_hypothesis.name, head, vois_ts_dict,
                                 head.sensorsSEEG.keys(), hpf_flag=False, trajectories_plot=trajectories_plot,
                                 spectral_raster_plot=spectral_raster_plot, log_scale=True)

                # Optionally save results in mat files
                savemat(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_ts.mat"), vois_ts_dict)

        # Get data and observation signals:
        data, active_regions = prepare_data_for_fitting_vep_original(model_configuration, lsa_hypothesis, fsAVG,
                                                                     vois_ts_dict, sim.model,
                                                                     noise_intensity, active_regions=None,
                                                                     active_regions_th=0.1, observation_model=3,
                                                                     mixing=None)
        savemat(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_fit_data.mat"), data)

        # Fit and get estimates:
        est, fit = stanfit_model(model, data, mode="optimizing", iter=60000)
        savemat(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_fit_est.mat"), est)

        plot_fit_results(lsa_hypothesis.name, head, est, data, active_regions,
                         time=vois_ts_dict['time'], seizure_indices=[0, 1], trajectories_plot=True)


        print("Done!")



if __name__ == "__main__":

    main_fit_sim_hyplsa()