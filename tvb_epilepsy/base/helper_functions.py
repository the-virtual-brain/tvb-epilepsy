import numpy as np


###
# This function is a helper function to run parameter search exploration (pse) for Linear Stability Analysis (LSA).
###
def pse_from_hypothesis(hypothesis, n_samples, half_range=0.1, global_coupling=[],
                        healthy_regions_parameters=[], model_configuration_service=None, lsa_service=None, 
                        save_services=False, **kwargs):

    from tvb_epilepsy.base.constants import MAX_DISEASE_VALUE, K_DEF, FOLDER_RES
    from tvb_epilepsy.base.utils import initialize_logger, linear_index_to_coordinate_tuples, \
                                        dicts_of_lists_to_lists_of_dicts, list_of_dicts_to_dicts_of_ndarrays
    from tvb_epilepsy.base.sampling_service import StochasticSamplingService
    from tvb_epilepsy.base.pse_service import PSEService

    logger = initialize_logger(__name__)
    
    all_regions_indices = range(hypothesis.get_number_of_regions())
    disease_indices = hypothesis.get_regions_disease_indices()
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()

    pse_params = {"path": [], "indices": [], "name": [], "samples": []}

    # First build from the hypothesis the input parameters of the parameter search exploration.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)
    for ii in range(len(hypothesis.x0_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.x0_indices[ii]]) +
                                  " Excitability")

        # Now generate samples using a truncated uniform distribution
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=1, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            trunc_limits={"high": MAX_DISEASE_VALUE},
                                            sampler="uniform",
                                            loc=hypothesis.x0_values[ii] - half_range, scale=2 * half_range)
        pse_params["samples"].append(sampler.generate_samples(**kwargs))

    for ii in range(len(hypothesis.e_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.e_indices[ii]]) +
                                  " Epileptogenicity")

        # Now generate samples using a truncated uniform distribution
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=1, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            trunc_limits={"high": MAX_DISEASE_VALUE},
                                            sampler="uniform",
                                            loc=hypothesis.e_values[ii] - half_range, scale=2 * half_range)
        pse_params["samples"].append(sampler.generate_samples(**kwargs))

    for ii in range(len(hypothesis.w_values)):
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(hypothesis.w_indices[ii], hypothesis.connectivity.weights.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(hypothesis.connectivity.region_labels[inds[0][0]]) + "-" +
                                str(hypothesis.connectivity.region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds), + "]")

        # Now generate samples using a truncated normal distribution
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=1, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            trunc_limits={"high": MAX_DISEASE_VALUE},
                                            sampler="norm", loc=hypothesis.w_values[ii], scale=half_range)
        pse_params["samples"].append(sampler.generate_samples(**kwargs))

    if model_configuration_service is None:
        kloc = K_DEF
    else:
        kloc = model_configuration_service.K_unscaled[0]
    for val in global_coupling:
        pse_params["path"].append("model.configuration.service.K_unscaled")
        inds = val.get("indices", all_regions_indices)
        if np.all(inds == all_regions_indices):
            pse_params["name"].append("Global coupling")
        else:
            pse_params["name"].append("Afferent coupling[" + str(inds) + "]")
        pse_params["indices"].append(inds)

        # Now generate samples susing a truncated normal distribution
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=1, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            trunc_limits={"low": 0.0}, sampler="norm", loc=kloc, scale=30*half_range)
        pse_params["samples"].append(sampler.generate_samples(**kwargs))

    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)

    # Add a random jitter to the healthy regions if required...:
    for val in healthy_regions_parameters:
        inds = val.get("indices", healthy_indices)
        name = val.get("name", "x0")
        n_params = len(inds)
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=n_params, sampler="uniform",
                                            trunc_limits={"low": 0.0}, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            loc=kwargs.get("loc", 0.0), scale=kwargs.get("scale", 2*half_range))

        samples = sampler.generate_samples(**kwargs)
        for ii in range(n_params):
            pse_params_list.append({"path": "model_configuration_service." + name, "samples": samples[ii],
                                    "indices": [inds[ii]], "name": name})

    # Now run pse service to generate output samples:

    pse = PSEService("LSA", hypothesis=hypothesis, params_pse=pse_params_list)
    pse_results, execution_status = pse.run_pse(grid_mode=False, lsa_service_input=lsa_service,
                                                model_configuration_service_input=model_configuration_service)

    pse_results = list_of_dicts_to_dicts_of_ndarrays(pse_results)

    if save_services:
        logger.info(pse.__repr__())
        pse.write_to_h5(FOLDER_RES, "test_pse_service.h5")
        
    return pse_results, pse_params_list



# This function is a helper function to run sensitivity analysis parameter search exploration (pse)
# for Linear Stability Analysis (LSA).

def sensitivity_analysis_pse_from_hypothesis(hypothesis, n_samples, method="sobol", half_range=0.1, global_coupling=[],
                                             healthy_regions_parameters=[], model_configuration_service=None, 
                                             lsa_service=None, save_services=False, **kwargs):

    from tvb_epilepsy.base.constants import MAX_DISEASE_VALUE, FOLDER_RES
    from tvb_epilepsy.base.utils import initialize_logger, linear_index_to_coordinate_tuples, \
                                        list_of_dicts_to_dicts_of_ndarrays, dicts_of_lists_to_lists_of_dicts
    from tvb_epilepsy.base.sampling_service import StochasticSamplingService
    from tvb_epilepsy.base.pse_service import PSEService
    from tvb_epilepsy.base.sensitivity_analysis_service import SensitivityAnalysisService, METHODS
    
    logger = initialize_logger(__name__)
    
    method = method.lower()
    if np.in1d(method, METHODS):
        if np.in1d(method, ["delta", "dgsm"]):
            sampler = "latin"
        elif method == "sobol":
            sampler = "saltelli"
        elif method == "fast":
            sampler = "fast_sampler"
        else:
            sampler = method
    else:
        raise ValueError(
            "Method " + str(method) + " is not one of the available methods " + str(METHODS) + " !")

    all_regions_indices = range(hypothesis.get_number_of_regions())
    disease_indices = hypothesis.get_regions_disease_indices()
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()

    pse_params = {"path": [], "indices": [], "name": [], "bounds": []}
    n_inputs = 0

    # First build from the hypothesis the input parameters of the sensitivity analysis.
    # These can be either originating from excitability, epileptogenicity or connectivity hypotheses,
    # or they can relate to the global coupling scaling (parameter K of the model configuration)
    for ii in range(len(hypothesis.x0_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.x0_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.x0_indices[ii]]) +
                                  " Excitability")
        pse_params["bounds"].append([hypothesis.x0_values[ii]-half_range,
                       np.min([MAX_DISEASE_VALUE, hypothesis.x0_values[ii]+half_range])])

    for ii in range(len(hypothesis.e_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.e_values")
        pse_params["name"].append(str(hypothesis.connectivity.region_labels[hypothesis.e_indices[ii]]) +
                                  " Epileptogenicity")
        pse_params["bounds"].append([hypothesis.e_values[ii]-half_range,
                       np.min([MAX_DISEASE_VALUE, hypothesis.e_values[ii]+half_range])])

    for ii in range(len(hypothesis.w_values)):
        n_inputs += 1
        pse_params["indices"].append([ii])
        pse_params["path"].append("hypothesis.w_values")
        inds = linear_index_to_coordinate_tuples(hypothesis.w_indices[ii], hypothesis.connectivity.weights.shape)
        if len(inds) == 1:
            pse_params["name"].append(str(hypothesis.connectivity.region_labels[inds[0][0]]) + "-" +
                                str(hypothesis.connectivity.region_labels[inds[0][0]]) + " Connectivity")
        else:
            pse_params["name"].append("Connectivity[" + str(inds), + "]")
            pse_params["bounds"].append([np.max([hypothesis.w_values[ii]-half_range, 0.0]),
                                                 hypothesis.w_values[ii]+half_range])

    for val in global_coupling:
        n_inputs += 1
        pse_params["path"].append("model.configuration.service.K_unscaled")
        inds = val.get("indices", all_regions_indices)
        if np.all(inds == all_regions_indices):
            pse_params["name"].append("Global coupling")
        else:
            pse_params["name"].append("Afferent coupling[" + str(inds) + "]")
        pse_params["indices"].append(inds)
        pse_params["bounds"].append(val["bounds"])

    # Now generate samples suitable for sensitivity analysis
    sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=n_inputs, sampler=sampler, trunc_limits={},
                                        sampling_module="salib", random_seed=kwargs.get("random_seed", None),
                                        bounds=pse_params["bounds"])

    input_samples = sampler.generate_samples(**kwargs)
    n_samples = input_samples.shape[1]
    pse_params.update({"samples": [np.array(value) for value in input_samples.tolist()]})

    pse_params_list = dicts_of_lists_to_lists_of_dicts(pse_params)

    # Add a random jitter to the healthy regions if required...:
    for val in healthy_regions_parameters:
        inds = val.get("indices", healthy_indices)
        name = val.get("name", "x0")
        n_params = len(inds)
        sampler = StochasticSamplingService(n_samples=n_samples, n_outputs=n_params, sampler="uniform",
                                            trunc_limits={"low": 0.0}, sampling_module="scipy",
                                            random_seed=kwargs.get("random_seed", None),
                                            loc=kwargs.get("loc", 0.0), scale=kwargs.get("scale", 2*half_range))

        samples = sampler.generate_samples(**kwargs)
        for ii in range(n_params):
            pse_params_list.append({"path": "model_configuration_service." + name, "samples": samples[ii],
                                    "indices": [inds[ii]], "name": name})

    # Now run pse service to generate output samples:

    pse = PSEService("LSA", hypothesis=hypothesis, params_pse=pse_params_list)
    pse_results, execution_status = pse.run_pse(grid_mode=False, lsa_service_input=lsa_service,
                                                model_configuration_service_input=model_configuration_service)

    pse_results = list_of_dicts_to_dicts_of_ndarrays(pse_results)

    # Now prepare inputs and outputs and run the sensitivity analysis:
    # NOTE!: Without the jittered healthy regions which we don' want to include into the sensitivity analysis!
    inputs = dicts_of_lists_to_lists_of_dicts(pse_params)

    outputs = [{"names": ["LSA Propagation Strength"], "values": pse_results["propagation_strengths"]}]
    sensitivity_analysis_service = SensitivityAnalysisService(inputs, outputs, method=method,
                                                              calc_second_order=kwargs.get("calc_second_order", True),
                                                              conf_level=kwargs.get("conf_level", 0.95))

    results = sensitivity_analysis_service.run(**kwargs)

    if save_services:
        logger.info(pse.__repr__())
        pse.write_to_h5(FOLDER_RES, method+"_test_pse_service.h5")

        logger.info(sensitivity_analysis_service.__repr__())
        sensitivity_analysis_service.write_to_h5(FOLDER_RES, method+"_test_sa_service.h5")

    return results, pse_results


###
# A helper function to make good choices for simulation settings, noise and monitors for a TVB simulator
###
def setup_TVB_simulation_from_model_configuration(model_configuration, connectivity, dt, sim_length, monitor_period,
                                              model_name="EpileptorDP", zmode=np.array("lin"), scale_time=1,
                                              noise_instance=None, noise_intensity=None, monitor_expressions=None,
                                              monitors_instance=None):

    from tvb_epilepsy.base.constants import ADDITIVE_NOISE, NOISE_SEED
    from tvb_epilepsy.base.simulators import SimulationSettings
    from tvb_epilepsy.base.epileptor_model_factory import model_build_dict, model_noise_intensity_dict, \
                                                          model_noise_type_dict
    from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB
    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic, EpileptorDP2D
    from tvb.datatypes import equations
    from tvb.simulator import monitors, noise
    from tvb.simulator.models import Epileptor

    model = model_build_dict[model_name](model_configuration, zmode=zmode)

    if isinstance(model, Epileptor):
        model.tt *= scale_time * 0.25
    else:
        model.tau1 *= scale_time
        if isinstance(model, EpileptorDPrealistic):
            model.slope = 0.25
            model.pmode = np.array("z")

    if monitor_expressions is None:
        monitor_expressions = []
        for i in range(model._nvar):
            monitor_expressions.append("y" + str(i))
        if not (isinstance(model, EpileptorDP2D)):
            monitor_expressions.append("y3 - y0")

    if monitor_expressions is not None:
        model.variables_of_interest = monitor_expressions

    if monitors_instance is None:
        monitors_instance = monitors.TemporalAverage()

    if monitor_period is not None:
        monitors_instance.period = monitor_period

    default_noise_intensity = model_noise_intensity_dict[model_name]
    default_noise_type = model_noise_type_dict[model_name]

    if noise_intensity is None:
        noise_intensity = default_noise_intensity

    if noise_instance is not None:
        noise_instance.nsig = noise_intensity

    else:
        if default_noise_type is ADDITIVE_NOISE:
            noise_instance = noise.Additive(nsig=noise_intensity,
                                            random_stream=np.random.RandomState(seed=NOISE_SEED))
            noise_instance.configure_white(dt=dt)

        else:
            eq = equations.Linear(parameters={"a": 1.0, "b": 0.0})
            noise_instance = noise.Multiplicative(ntau=10, nsig=noise_intensity, b=eq,
                                                  random_stream=np.random.RandomState(seed=NOISE_SEED))
            noise_shape = noise_instance.nsig.shape
            noise_instance.configure_coloured(dt=dt, shape=noise_shape)

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt, scale_time=scale_time,
                                  noise_preconfig=noise_instance, noise_type=default_noise_type,
                                  noise_intensity=noise_intensity, noise_ntau=noise_instance.ntau,
                                  monitors_preconfig=monitors_instance, monitor_type=monitors_instance._ui_name,
                                  monitor_sampling_period=monitor_period, monitor_expressions=monitor_expressions,
                                  variables_names=model.variables_of_interest)

    simulator_instance = SimulatorTVB(connectivity, model_configuration, model, settings)

    return simulator_instance



###
# A helper function to make good choices for simulation settings for a custom simulator
###
def setup_custom_simulation_from_model_configuration(model_configuration, connectivity, dt, sim_length, monitor_period, model_name, scale_time=1,
                     noise_intensity=None):

    from tvb_epilepsy.custom.simulator_custom import EpileptorModel, custom_model_builder, \
                                                     SimulationSettings, SimulatorCustom
    if model_name != EpileptorModel._ui_name:
        print "You can use only " + EpileptorModel._ui_name + "for custom simulations!"

    model = custom_model_builder(model_configuration)

    if noise_intensity is None:
        noise_intensity = 0  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

    settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                  scale_time=scale_time,
                                  noise_intensity=noise_intensity,
                                  monitor_sampling_period=monitor_period)

    simulator_instance = SimulatorCustom(connectivity, model_configuration, model, settings)

    return simulator_instance


def set_time_scales(fs=4096.0, dt=None, time_length=1000.0, scale_time=1.0, scale_fsavg=8.0, report_every_n_monitor_steps=10,):
    if dt is None:
        dt = 1000.0 / fs

    dt /= scale_time

    fsAVG = fs / scale_fsavg
    monitor_period = scale_fsavg * dt
    sim_length = time_length / scale_time
    time_length_avg = np.round(sim_length / monitor_period)
    n_report_blocks = max(report_every_n_monitor_steps * np.round(time_length_avg / 100), 1.0)

    return dt, fsAVG, sim_length, monitor_period, n_report_blocks