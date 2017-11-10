functions {

    real EpileptorDP2D_fun_x1(real x1, real z, real yc, real Iext1, real a, real db, real d, real slope, real tau1) {

        real fx1;

        // population 1
        if (x1 <= 0.0) {
            // if_ydot0 = a * x1 ** 2 + (d - b) * y[0],
            fx1 = a * x1 * x1 + db * x1;
        } else {
            // d * y[0] - 0.6 * (y[1] - 4.0) ** 2 - slope,
            fx1 =  z - 4.0;
            fx1 = d * x1 - 0.6 * fx1 * fx1 - slope;
        }
        // ydot[0] = tau1 * (yc - y[1] + Iext1 - where(y[0] < 0.0, if_ydot0, else_ydot0) * y[0])
        fx1 = tau1 * (yc - z + Iext1 - fx1 * x1);

        return fx1;
    }

    real EpileptorDP2D_fun_z_lin(real x1, real z, real x0, real coupling, real tau0, real tau1) {

        real fz;

        // slow energy variable with a linear form (original Epileptor)
        // ydot[1] = tau1 * (4 * (y[0] - x0) + where(y[1] < 0.0, if_ydot1, else_ydot1) - y[1] + K * c_pop1) / tau0
        fz = 4.0 * (x1 - x0) - z - coupling;
        if (z < 0.0) {
            // if_ydot1 = - 0.1 * y[1] ** 7
            fz = fz - 0.1 * z * z * z * z * z * z * z;
        }
        fz =  tau1 *  fz / tau0;
        return fz;
    }

}


data {

    int n_regions;
    int n_times;
    int n_signals;
    int n_active_regions;
    int n_nonactive_regions;

    /* Integer flags and indices for (non)active regions */
    int active_regions_flag[n_regions];
    int active_regions[n_active_regions];
    int nonactive_regions[n_nonactive_regions];

    /* _lo stands for parameters' lower limits
       _hi stands for parameters' higher limits
       _pdf stands for an integer index of the distribution to be used for sampling, for the moment among:
            0. uniform
            1. normal
            2. gamma
            3. lognormal
            5. exponential
            5. beta
       _p1 stands for a distribution's first parameter
       _p2 stands for a distribution's second parameter, if any, otherwise, _p2 = _p1 */

    /* Generative model */
    /* Epileptor */
    // int zmode;
    real a;
    real d;
    real b;
    real yc;
    real Iext1;
    real slope;
    real x0cr;
    real rx0;
    real zeq_lo;
    real zeq_hi;
    // real x0_lo;
    // real x0_hi;
    /* x1eq parameter (only normal distribution, ignoring _pdf) */
    real x1eq_lo;
    real x1eq_hi;
    vector[n_regions] x1eq0;
    /* x1init parameter (only normal distribution, ignoring _pdf) */
    real x1init_lo;
    real x1init_hi;
    /* zinit parameter (only normal distribution, ignoring _pdf) */
    real zinit_lo;
    real zinit_hi;
    /* tau1 parameter (default: gamma distribution) */
    real<lower=0.0> tau1_lo;
    real<lower=0.0> tau1_hi;
    real<lower=0.0> tau1_p1;
    real<lower=0.0> tau1_p2;
    int<lower=0.0> tau1_pdf;
    /* tau0 parameter (default: gamma distribution) */
    real<lower=0.0> tau0_lo;
    real<lower=0.0> tau0_hi;
    real<lower=0.0> tau0_p1;
    real<lower=0.0> tau0_p2;
    int<lower=0.0> tau0_pdf;
    /* x1 parameter (only normal distribution for autoregressive model) */
    real x1_lo;
    real x1_hi;
    /* z parameter (only normal distribution for autoregressive model) */
    real z_lo;
    real z_hi;

    /* Coupling */
    /* EC (effective connectivity) parameter (default: gamma distribution) */
    matrix<lower=0.0>[n_regions, n_regions] EC_p1;
    matrix<lower=0.0>[n_regions, n_regions] EC_p2;
    real<lower=0.0> EC_lo
    real<lower=0.0> EC_hi
    int<lower=0.0> EC_pdf;
    /* K (global coupling) parameter (default: gamma distribution) */
    real<lower=0.0> K_lo;
    real<lower=0.0> K_hi;
    real<lower=0.0> K_p1;
    real<lower=0.0> K_p2;
    int<lower=0.0> K_pdf;

    /* Integration */
    int euler_method;
    real dt;
    /* Dynamic noise strength parameter (default: gamma distribution) */
    real<lower=0.0> sig_hi;
    real<lower=0.0> sig_lo;
    real<lower=0.0> sig_p1;
    real<lower=0.0> sig_p2;
    int<lower=0.0> sig_pdf;
    /* Equilibrium point variability parameter (default: gamma distribution) */
    real<lower=0.0> sig_eq_lo;
    real<lower=0.0> sig_eq_hi;
    real<lower=0.0> sig_eq_p1;
    real<lower=0.0> sig_eq_p2;
    int<lower=0.0> sig_pdf;
    /* Initial condition variability parameter (default: gamma distribution) */
    real<lower=0.0> sig_init_hi;
    real<lower=0.0> sig_init_lo;
    real<lower=0.0> sig_init_p1;
    real<lower=0.0> sig_init_p2;
    int<lower=0.0> sig_init_pdf;

    /* Observation model */
    int observation_expression;
    int observation_model;
    matrix[n_signals, n_active_regions] mixing;
    matrix[n_times, n_signals] signals;
    /* Observation variability parameter (default: gamma distribution) */
    real<lower=0.0> eps_lo;
    real<lower=0.0> eps_hi;
    real<lower=0.0> eps_p1;
    real<lower=0.0> eps_p2;
    int<lower=0.0> eps_pdf;
    /* Observation signal scaling parameter (default: uniform distribution) */
    real<lower=0.0> scale_signal_lo;
    real<lower=0.0> scale_signal_hi;
    real<lower=0.0> scale_signal_p1;
    real<lower=0.0> scale_signal_p2;
    int<lower=0.0> scale_signal_pdf;
    /* Observation signal offset parameter (default: uniform distribution) */
    real<upper=0.0> offset_signal_lo;
    real<lower=0.0> offset_signal_hi;
    real offset_signal_p1;
    real offset_signal_p2;
    int<lower=0.0> offset_signal_pdf;
}


transformed data {

    real db;

    // Calculate db parameter, which corresponds to parameter b for the 2D reduced Epileptor (Proix etal 2014)
    db = d - b;
}


parameters {

    /* Generative model */
    /* Epileptor */
    vector<lower=x1eq_lo, upper=x1eq_hi>[n_regions] x1eq; // x1 equilibrium point coordinate
    vector<lower=x1init_lo, upper=x1init_hi>[n_active_regions] x1init; // x1 equilibrium point coordinate
    real<lower=K_lo, upper=K_hi> K; // global coupling scaling
    vector<lower=tau1_lo, upper=tau1_hi> tau1; // time scale [n_active_regions]
    vector<lower=tau0_lo, upper=tau0_hi> tau0; // time scale separation [n_active_regions]
    /* Coupling */
    matrix<lower=0.0>[n_regions, n_regions] EC; // Effective connectivity

    /* Integration */
    real<lower=sig_lo, upper=sig_hi> sig; // variance of phase flow, i.e., dynamic noise
    real<lower=sig_eq_lo, upper=sig_eq_hi> sig_eq; // variance of equilibrium point
    real<lower=sig_init_lo, upper=sig_init_hi> sig_init; // variance of initial condition

    /* Observation model */
    real<lower=eps_lo, upper=eps_hi> eps; // variance of observation noise
    real<lower=scale_signal_lo, upper=scale_signal_hi> scale_signal; // observation signal scaling
    real<lower=offset_signal_lo, upper=offset_signal_hi> offset_signal; // observation signal offset

}


transformed parameters {

    /* Generative model */
    /* Epileptor */
    vector[n_regions] zeq; // z equilibrium point coordinate
    vector[n_regions] x0; // excitability parameter

    /* Coupling */
    vector[n_regions] coupling_eq; // coupling at equilibrium
    matrix[n_active_regions, n_times] coupling; // actual effective coupling per time point

    /* zeq */
    for (ii in 1:n_regions) {
        // solving for zeq (z->0, tau1->1)
        zeq[ii] = EpileptorDP2D_fun_x1(x1eq[ii], 0.0, yc, Iext1, a, db, d, slope, 1.0);
    }

    /* Coupling
    We place it here for the moment because it has a high diagnostic value */
    {

        for (ii in 1:n_regions) {
            coupling_eq[ii] = 0.0;
            for (jj in 1:n_regions) {
                if (ii!=jj) {
                    coupling_eq[ii] = coupling_eq[ii] + EC[ii, jj] * (x1eq[jj] - x1eq[ii]);
                }
            }
        }

        for (tt in 1:n_times) {
            for (ii in 1:n_active_regions) {
               coupling[ii, tt] = 0.0;
               // coupling active -> active regions excluding self-coupling
               for (jj in 1:n_active_regions) {
                   if (ii!=jj) {
                       coupling[ii, tt] =
                              coupling[ii, tt] + EC[active_regions[ii], active_regions[jj]] * (x1[jj, tt] - x1[ii, tt]);
                   }
               }
               // coupling nonactive -> active regions 
               for (jj in 1:n_nonactive_regions) {
                   // non active regions are assumed to always stay close to their equilibrium point
                   coupling[ii, tt] =
                            coupling[ii, tt] +
                             EC[active_regions[ii], nonactive_regions[jj]] * (x1eq[nonactive_regions[jj]] - x1[ii, tt]);
               }

        }

    }

    /* x0 */
    for (ii in 1:n_regions) {
        // solving for x0 (x0->0, tau0->1, tau1->1)
        x0[ii] = EpileptorDP2D_fun_z_lin(x1eq[ii], zeq[ii], 0.0, K * coupling_eq[ii], 1.0, 1.0) / 4.0;
    }

}


model {

    vector[nn] observation;

    /* Sampling of global coupling scaling */
    K ~ sample_lpdf(K_pdf, K_lo, K_hi, K_p1, K_p2);

    /* Sampling of the various variances */
    sig ~ sample_lpdf(sig_pdf, sig_lo, sig_hi, sig_p1, sig_p2);
    sig_eq ~ sample_lpdf(sig_eq_pdf, sig_eq_lo, sig_eq_hi, sig_eq_p1, sig_eq_p2);
    sig_init ~ sample_lpdf(sig_init_pdf, sig_init_lo, sig_init_hi, sig_init_p1, sig_init_p2);

    /* Sampling of x1 equilibrium point coordinate and effective connectivity */
    for (ii in 1:n_regions) {
        x1eq[ii] ~ normal(x1eq0[ii], sig_eq)T[x1eq_lo, x1eq_hi];
        for (jj in 1:n_regions) {
            EC[ii, jj] ~ sample_lpdf(EC_lo, EC_hi, EC_p1[ii, jj], EC_p2[ii, jj]);
        }
    }

    /* Sampling of initial condition*/
    for (ii in 1:n_active_regions) {
        x1[ii, 1] ~ normal(x1eq[active_regions[ii]], sig_init)T[x1init_lo, x1init_hi];
        z[ii, 1] ~ normal(zeq[active_regions[ii]], sig_init)T[zinit_lo, zinit_hi];
//        tau0[ii] ~ sample_lpdf(tau1_lo, tau1_hi, tau1_p1, tau1_p2);
//        tau0[ii] ~ sample_lpdf(tau0_lo, tau0_hi, tau0_p1, tau0_p2);
    }

    /* Sampling of time scales */
    tau1[ii] ~ sample_lpdf(tau1_lo, tau1_hi, tau1_p1, tau1_p2);
    tau0[ii] ~ sample_lpdf(tau0_lo, tau0_hi, tau0_p1, tau0_p2);

    /* Integrate & predict  */
    for (tt in 2:n_times) {
        /* Auto-regressive generative model  */
        if (euler_method==-1){ // backward euler method
            for (ii in 1:n_active_regions) {
                dx1 = EpileptorDP2D_fun_x1(x1[ii, tt-1], z[ii, tt-1], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                dz = EpileptorDP2D_fun_z_lin(x1[ii, tt-1], z[ii, tt-1], x0[active_regions[ii]], K*coupling[ii, tt-1], tau0, tau1); // tau0
                x1[ii, tt] ~ normal(x1[ii, tt-1] + dt*dx1, sig); // T[x1_lo, x1_hi];
                z[ii, tt] ~ normal(z[ii, tt-1] + dt*dz, sig); // T[z_lo, z_hi];
            }
            // TODO: code for midpoint euler method
        } else {// forward euler method
            for (ii in 1:n_active_regions) {
                dx1 = EpileptorDP2D_fun_x1(x1[ii, tt], z[ii, tt], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                dz = EpileptorDP2D_fun_z_lin(x1[ii, tt], z[ii, tt], x0[active_regions[ii]], K*coupling[ii, tt], tau0, tau1); // tau0[ii]
                x1[ii, tt] ~ normal(x1[ii, tt-1] + dt*dx1, sig); // T[x1_lo, x1_hi];
                z[ii, tt] ~ normal(z[ii, tt-1] + dt*dz, sig); // T[z_lo, z_hi];
            }
        }

        /* Observation model  */
        if (observation_expression == 1) {
            observation = col(x1, t);
        } else if (observation_expression == 2){
            observation = (col(x1, t) - xeq) / 2.0;
        } else {
            observation = (col(x1, t) - xeq + col(z, t) - zeq) / 2.75;
        }

        if  (observation_model == 1) {
            // seeg log power
            signals[tt] ~ normal(scale_signal * log(mixing * exp(observation)) + offset_signal, eps);
        } else if (observation_model == 2){
            // just x1 with some mixing, scaling and offset_signal
            signals[tt] ~ normal(scale_signal * mixing * observation + offset_signal, eps);
        } else {
            // just x1 with some mixing, scaling and offset_signal
            signals[tt] ~ normal(scale_signal * observation + offset_signal, eps);
        }
    }

    /* Predict nonactive x0
    for (ii in 1:n_nonactive_regions) {
        // solving for x0 (x0->0, tau0->1, tau1->1)
        x0_nonactive[ii] ~ normal(EpileptorDP2D_fun_z_lin(x1eq[nonactive_regions[ii]], zeq[nonactive_regions[ii]], 0.0,
                                                       K * coupling_eq[nonactive_regions[ii]], 1.0, 1.0) / 4.0, eps_x0);
    }
    */


}


generated quantities {

    vector[n_regions] ModelEpileptogenicity;
    vector[n_regions] PathologicalExcitability;
    matrix[nt, ns] fit_signals;

    for (ii in 1:n_active_regions) {
        ModelEpileptogenicity[active_regions[ii]] = 3.0*x1eq[ii] + 5.0;
        PathologicalExcitability[active_regions[ii]] = rx0 * x0[ii] - x0cr;
    }

    for (ii in 1:n_nonactive_regions) {
        ModelEpileptogenicity[nonactive_regions[ii]] = 3.0*x1eq0[nonactive_regions[ii]] + 5.0;
        PathologicalExcitability[nonactive_regions[ii]] = rx0 * x0_nonactive[ii] - x0cr;
    }

    {
        vector[nn] observation;

        for (t in 1:nt) {
            /* Observation model  */
            if (observation_expression == 1) {
                observation = (col(x1, t) - xeq + col(z, t) - zeq) / 2.75;
            } else if (observation_expression == 2){
                observation = (col(x1, t) - xeq) / 2.0;
            } else {
                observation = col(x1, t);
            }
            if  (observation_model == 1) {
                // seeg log power
                fit_signals[t] = (scale_signal * (log(mixing * exp(observation)) + offset_signal))';
            } else if (observation_model == 2){
                // just x1 with some mixing, scaling and offset_signal
                fit_signals[tt] = (scale_signal * mixing * observation + offset_signal)';
            } else {
                // just x1 with some mixing, scaling and offset_signal
                fit_signals[t] = (scale_signal * observation + offset_signal)';
            }
        }
    }

}
