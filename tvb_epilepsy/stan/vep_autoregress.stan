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

    real sample_lpdf(real x, int pdf, real p1, real p2) {
        if (pdf == 1) {
            // normal
            return normal_lpdf(x | p1, p2);
        } else if (pdf == 2) {
            // gamma: convert to rate from scale!
            return gamma_lpdf(x | p1, 1.0 / p2);
        } else if (pdf == 3) {
            // lognormal
            return lognormal_lpdf(x | p1, p2);
        } else if (pdf == 4) {
            // exponential
            return exponential_lpdf(x | p1);
        /* Beta not used for now...
        } else if (pdf == 5) {
            // beta
            return beta_lpdf(x | p1, p2);
        */
        } else {
            // uniform
            return uniform_lpdf(x | p1, p2);
        }
    }

    real sample_from_stdnormal_lpdf(real x, int pdf, real p1, real p2) {
        real n01
        n01 = normal_lpdf(x | 0.0, 1.0)
        if (pdf == 1) {
            // Normal(mean, sigma**2) =  mean + sigma*Normal(0,1)
            return p1 + p2 * n01;
        } else if (pdf == 2) {
        /* not possible! requires some of INDEPENDENT Gamma random variables!
            // Gamma(shape, scale) = 2*shape*Gamma(1/2, scale) = 2*shape*Normal(0, scale/2)**2
            // because:
            // Normal(0,sigma)**2 = Gamma(1/2, 2*sigma**2)
            //k*Gamma(shape, scale) = Gamma(k*shape, scale)
            p2 = sqrt(p2/2)
            n01 = p2 * n01
            return 2 * p1 * n01*n01;*/
            return gamma_lpdf(x | p1, 1.0 / p2);
        } else if (pdf == 3) {
            // lognormal(mean, sigma**2) = exp(Normal(mean, sigma**2)) = exp(mean + sigma*Normal(0,1))
            return exp(p1 + p2 * n01);
        } else if (pdf == 4) {
        /* not possible! requires some of INDEPENDENT Gamma random variables!
            // Exponential(scale) = Gamma(1, scale) and following the above for Gamma(), we get:
            //Exponential(scale) = 2*Normal(0, scale/2)**2
            p1 = sqrt(p1/2)
            n01 = p1 * n01
            return 2 * n01*n01;
        */
            return exponential_lpdf(x | p1);
        /* Beta not used for now...
        } else if (pdf == 5) {
            // Beta(alpha, beta) = Gamma(alpha, c) / (Gamma(alpha, c) + Gamma(beta, c))
            return beta_lpdf(x | p1, p2);
        */
        } else {
            // Uniform(a, b) = a + (b-a)*Uniform(0, 1)  =  a + (b-a)*Normal(0, 1)_CDF
            return p1 + (p2-p1)*normal_lcdf(x | 0.0, 1.0);
        }
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
    int<lower=0> tau1_pdf;
    /* tau0 parameter (default: gamma distribution) */
    real<lower=0.0> tau0_lo;
    real<lower=0.0> tau0_hi;
    real<lower=0.0> tau0_p1;
    real<lower=0.0> tau0_p2;
    int<lower=0> tau0_pdf;
    /* x1 parameter (only normal distribution for autoregressive model) */
    real x1_lo;
    real x1_hi;
    /* z parameter (only normal distribution for autoregressive model) */
    real z_lo;
    real z_hi;

    /* Coupling */
    /* K (global coupling) parameter (default: gamma distribution) */
    real<lower=0.0> K_lo;
    real<lower=0.0> K_hi;
    real<lower=0.0> K_p1;
    real<lower=0.0> K_p2;
    int<lower=0> K_pdf;
    /* EC (effective connectivity) parameter (default: gamma distribution) */
    matrix<lower=0.0>[n_regions, n_regions] EC_p1;
    matrix<lower=0.0>[n_regions, n_regions] EC_p2;
    real<lower=0.0> EC_lo;
    real<lower=0.0> EC_hi;
    int<lower=0> EC_pdf;

    /* Integration */
    int euler_method;
    real dt;
    /* Dynamic noise strength parameter (default: gamma distribution) */
    real<lower=0.0> sig_hi;
    real<lower=0.0> sig_lo;
    real<lower=0.0> sig_p1;
    real<lower=0.0> sig_p2;
    int<lower=0> sig_pdf;
    /* Equilibrium point variability parameter (default: gamma distribution) */
    real<lower=0.0> sig_eq_lo;
    real<lower=0.0> sig_eq_hi;
    real<lower=0.0> sig_eq_p1;
    real<lower=0.0> sig_eq_p2;
    int<lower=0> sig_eq_pdf;
    /* Initial condition variability parameter (default: gamma distribution) */
    real<lower=0.0> sig_init_hi;
    real<lower=0.0> sig_init_lo;
    real<lower=0.0> sig_init_p1;
    real<lower=0.0> sig_init_p2;
    int<lower=0> sig_init_pdf;

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
    int<lower=0> eps_pdf;
    /* Observation signal scaling parameter (default: uniform distribution) */
    real<lower=0.0> scale_signal_lo;
    real<lower=0.0> scale_signal_hi;
    real<lower=0.0> scale_signal_p1;
    real<lower=0.0> scale_signal_p2;
    int<lower=0> scale_signal_pdf;
    /* Observation signal offset parameter (default: uniform distribution) */
    real<upper=0.0> offset_signal_lo;
    real<lower=0.0> offset_signal_hi;
    real offset_signal_p1;
    real offset_signal_p2;
    int<lower=0> offset_signal_pdf;
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
    matrix<lower=x1_lo, upper=x1_hi>[n_active_regions, n_times] x1; // x1 state variable
    matrix<lower=z_lo, upper=z_hi>[n_active_regions, n_times] z; // z state variable
    real<lower=tau1_lo, upper=tau1_hi> tau1; // time scale [n_active_regions]
    real<lower=tau0_lo, upper=tau0_hi> tau0; // time scale separation [n_active_regions]
    /* Coupling */
    real<lower=K_lo, upper=K_hi> K; // global coupling scaling
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
               coupling[ii, tt] = coupling[ii, tt] +
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

    vector[n_active_regions] observation;
    real df;

    /* Sampling of global coupling scaling */
    K ~ sample(K_pdf, K_p1, K_p2);

    /* Sampling of the various variances */
    sig ~ sample(sig_pdf, sig_p1, sig_p2);
    sig_eq ~ sample(sig_eq_pdf, sig_eq_p1, sig_eq_p2);
    sig_init ~ sample(sig_init_pdf, sig_init_p1, sig_init_p2);

    /* Sampling of x1 equilibrium point coordinate and effective connectivity */
    for (ii in 1:n_regions) {
        x1eq[ii] ~ normal(x1eq0[ii], sig_eq);
        for (jj in 1:n_regions) {
            EC[ii, jj] ~ sample(EC_pdf, EC_p1[ii, jj], EC_p2[ii, jj]);
        }
    }

    /* Sampling of initial condition*/
    for (ii in 1:n_active_regions) {
        x1[ii, 1] ~ normal(x1eq[active_regions[ii]], sig_init);
        z[ii, 1] ~ normal(zeq[active_regions[ii]], sig_init);
//        tau0[ii] ~ sample(tau1_pdf, tau1_p1, tau1_p2);
//        tau0[ii] ~ sample(tau0_pdf, tau0_p1, tau0_p2);
    }

    /* Sampling of time scales */
    tau1 ~ sample(tau1_pdf, tau1_p1, tau1_p2);
    tau0 ~ sample(tau1_pdf, tau0_p1, tau0_p2);

    /* Sampling of observation scaling and offset */
    scale_signal ~ sample(scale_signal_pdf, scale_signal_p1, scale_signal_p2);
    offset_signal ~ sample(offset_signal_pdf, offset_signal_p1, offset_signal_p2);

    /* Integrate & predict  */
    for (tt in 2:n_times) {
        /* Auto-regressive generative model  */
        if (euler_method==-1){ // backward euler method
            for (ii in 1:n_active_regions) {
                df = EpileptorDP2D_fun_x1(x1[ii, tt-1], z[ii, tt-1], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                x1[ii, tt] ~ normal(x1[ii, tt-1] + dt*df, sig); // T[x1_lo, x1_hi];
                df = EpileptorDP2D_fun_z_lin(x1[ii, tt-1], z[ii, tt-1], x0[active_regions[ii]], K*coupling[ii, tt-1], tau0, tau1); // tau0
                z[ii, tt] ~ normal(z[ii, tt-1] + dt*df, sig); // T[z_lo, z_hi];
            }
            // TODO: code for midpoint euler method
        } else {// forward euler method
            for (ii in 1:n_active_regions) {
                df = EpileptorDP2D_fun_x1(x1[ii, tt], z[ii, tt], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                x1[ii, tt] ~ normal(x1[ii, tt-1] + dt*df, sig); // T[x1_lo, x1_hi];
                df = EpileptorDP2D_fun_z_lin(x1[ii, tt], z[ii, tt], x0[active_regions[ii]], K*coupling[ii, tt], tau0, tau1); // tau0[ii]
                z[ii, tt] ~ normal(z[ii, tt-1] + dt*df, sig); // T[z_lo, z_hi];
            }
        }

        /* Observation model  */
        if (observation_expression == 0) {
            observation = col(x1, tt);
        } else if (observation_expression == 1){
            observation = (col(x1, tt) - x1eq) / 2.0;
        } else {
            observation = (col(x1, tt) - x1eq + col(z, tt) - zeq[]) / 2.75;
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

}


generated quantities {

    vector[n_regions] ModelEpileptogenicity;
    vector[n_regions] PathologicalExcitability;
    matrix[n_times, n_signals] fit_signals;

    for (ii in 1:n_active_regions) {
        ModelEpileptogenicity[active_regions[ii]] = 3.0*x1eq[ii] + 5.0;
        PathologicalExcitability[active_regions[ii]] = rx0 * x0[active_regions[ii]] - x0cr;
    }

    for (ii in 1:n_nonactive_regions) {
        ModelEpileptogenicity[nonactive_regions[ii]] = 3.0*x1eq0[nonactive_regions[ii]] + 5.0;
        PathologicalExcitability[nonactive_regions[ii]] = rx0 * x0[nonactive_regions[ii]] - x0cr;
    }

    {
        vector[n_active_regions] observation;

        for (tt in 1:n_times) {
            /* Observation model  */
            /* Observation model  */
            if (observation_expression == 0) {
                observation = col(x1, tt);
            } else if (observation_expression == 1){
                observation = (col(x1, tt) - x1eq) / 2.0;
            } else {
                observation = (col(x1, tt) - x1eq + col(z, tt) - zeq) / 2.75;
            }
            if  (observation_model == 1) {
                // seeg log power
                fit_signals[tt] = (scale_signal * (log(mixing * exp(observation)) + offset_signal))';
            } else if (observation_model == 2){
                // seeg power: just x1 with some mixing, scaling and offset_signal
                fit_signals[tt] = (scale_signal * mixing * observation + offset_signal)';
            } else {
                // lfp power: just x1 with some mixing, scaling and offset_signal
                fit_signals[tt] = (scale_signal * observation + offset_signal)';
            }
        }
    }

}
