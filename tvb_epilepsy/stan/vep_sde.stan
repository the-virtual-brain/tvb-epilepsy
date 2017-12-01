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

    real sample_lpdf(real x, int pdf, real[] shape) {
        if (pdf == 1) {
            // normal
            return normal_lpdf(x | shape[1], shape[2]);
        } else if (pdf == 2) {
            // gamma: convert to rate from scale!
            return gamma_lpdf(x | shape[1], 1.0 / shape[2]);
        } else if (pdf == 3) {
            // lognormal
            return lognormal_lpdf(x | shape[1], shape[2]);
        } else if (pdf == 4) {
            // exponential
            return exponential_lpdf(x | shape[1]);
        /* Beta not used for now...
        } else if (pdf == 5) {
            // beta
            return beta_lpdf(x | shape[1], shape[2]);
        */
        } else {
            // uniform
            return uniform_lpdf(x | shape[1], shape[2]);
        }
    }

    real sample_from_stdnormal_lpdf(real x, int pdf, real[] shape) {
        real n01;
        if (pdf == 1) {
            // Normal(mean, sigma**2) =  mean + sigma*Normal(0,1)
            n01 = normal_lpdf(x | 0.0, 1.0);
            return shape[1] + shape[2] * n01;
        } else if (pdf == 2) {
        // not possible! requires sum of INDEPENDENT Gamma random variables!
            return gamma_lpdf(x | shape[1], 1.0 / shape[2]);
        } else if (pdf == 3) {
            // lognormal(mean, sigma**2) = exp(Normal(mean, sigma**2)) = exp(mean + sigma*Normal(0,1))
            n01 = normal_lpdf(x | 0.0, 1.0);
            return exp(shape[1] + shape[2] * n01);
        } else if (pdf == 4) {
        // not possible! requires sum of INDEPENDENT Gamma random variables!
            return exponential_lpdf(x | shape[1]);
        /* Beta not used for now...
        } else if (pdf == 5) {
            // Beta(alpha, beta) = Gamma(alpha, c) / (Gamma(alpha, c) + Gamma(beta, c))
            return beta_lpdf(x | shape[1], shape[2]);
        */
        } else {
            // Not possible!
            return uniform_lpdf(x | shape[1], shape[2]);
        }
    }
}


data {

    int SIMULATE;
    int DEBUG;

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
       _p[1] stands for a distribution's first parameter
       _p[2] stands for a distribution's second parameter, if any, otherwise, _p[2] = _p[1] */

    /* Generative model */
    /* Epileptor */
    // int zmode;
    real a;
    real b;
    real d;
    real yc;
    real Iext1;
    real slope;
    real x0cr;
    real rx0;
    // real x0_lo;
    // real x0_hi;
    /* x1 parameter (only normal distribution for autoregressive model) */
    real x1_lo;
    real x1_hi;
    /* z parameter (only normal distribution for autoregressive model) */
    real z_lo;
    real z_hi;
    /* x1eq parameter (only normal distribution, ignoring _pdf) */
    real x1eq_lo;
    real x1eq_hi;
    vector[n_regions] x1eq0;
    //real zeq_lo;
    //real zeq_hi;
    /* x1init parameter (only normal distribution, ignoring _pdf) */
    real x1init_lo;
    real x1init_hi;
    /* zinit parameter (only normal distribution, ignoring _pdf) */
    real zinit_lo;
    real zinit_hi;
    /* tau1 parameter (default: lognormal distribution) */
    real<lower=0.0> tau1_lo;
    real<lower=0.0> tau1_hi;
    real tau1_loc;
    real<lower=0.0> tau1_scale;
    real tau1_p[2];
    int<lower=0> tau1_pdf;
    /* tau0 parameter (default: lognormal distribution) */
    real<lower=0.0> tau0_lo;
    real<lower=0.0> tau0_hi;
    real tau0_loc;
    real<lower=0.0> tau0_scale;
    real tau0_p[2];
    int<lower=0> tau0_pdf;

    /* Coupling */
    /* K (global coupling) parameter (default: lognormal distribution) */
    real<lower=0.0> K_lo;
    real<lower=0.0> K_hi;
    real K_loc;
    real<lower=0.0> K_scale;
    real K_p[2];
    int<lower=0> K_pdf;
    /* MC (model connectivity) parameter (default: lognormal distribution) */
    real<lower=0.0> MC_lo;
    real<lower=0.0> MC_hi;
    matrix[n_regions, n_regions] MC_loc;
    matrix<lower=0.0>[n_regions, n_regions] MC_scale;
    real MC_p[2];
    int<lower=0> MC_pdf;


    /* Integration */
    int euler_method;
    real dt;
    /* Equilibrium point variability parameter (default: lognormal distribution) */
    real<lower=0.0> sig_eq_lo;
    real<lower=0.0> sig_eq_hi;
    real sig_eq_loc;
    real<lower=0.0> sig_eq_scale;
    real sig_eq_p[2];
    int<lower=0> sig_eq_pdf;
    /* Initial condition variability parameter (default: lognormal distribution) */
    real<lower=0.0> sig_init_lo;
    real<lower=0.0> sig_init_hi;
    real sig_init_loc;
    real<lower=0.0> sig_init_scale;
    real sig_init_p[2];
    int<lower=0> sig_init_pdf;
    /* Dynamic noise strength parameter (default: gamma distribution) */
    real<lower=0.0> sig_lo;
    real<lower=0.0> sig_hi;
    real sig_loc;
    real<lower=0.0> sig_scale;
    real sig_p[2];
    int<lower=0> sig_pdf;

    /* Observation model */
    int observation_expression;
    int observation_model;
    matrix[n_signals, n_active_regions] mixing;
    matrix[n_times, n_signals] signals;
    /* Observation variability parameter (default: lognormal distribution) */
    real<lower=0.0> eps_lo;
    real<lower=0.0> eps_hi;
    real eps_loc;
    real<lower=0.0> eps_scale;
    real eps_p[2];
    int<lower=0> eps_pdf;
    /* Observation signal scaling parameter (default: uniform distribution) */
    real<lower=0.0> scale_signal_lo;
    real<lower=0.0> scale_signal_hi;
    real scale_signal_loc;
    real<lower=0.0> scale_signal_scale;
    real scale_signal_p[2];
    int<lower=0> scale_signal_pdf;
    /* Observation signal offset parameter (default: uniform distribution) */
    real offset_signal_lo;
    real offset_signal_hi;
    real offset_signal_loc;
    real<lower=0.0> offset_signal_scale;
    real offset_signal_p[2];
    int<lower=0> offset_signal_pdf;
}


transformed data {
    real db;
    // Calculate db parameter, which corresponds to parameter b for the 2D reduced Epileptor (Proix etal 2014)
    db = d - b;
    if (DEBUG >= 0){
        print("DEBUG =", DEBUG, ", SIMULATE =", SIMULATE);
        print("n_regions =", n_regions, ", n_active_regions =", n_active_regions, ", n_nonactive_regions =", n_nonactive_regions);
        print("n_times =", n_times, ", n_signals =", n_signals);
        if (DEBUG >= 1){
            print("active_regions_flag =", active_regions_flag);
            print("active_regions =", active_regions);
            print("nonactive_regions =", nonactive_regions);
        }
        print("a =", a, ", b =", b, ", d =", d, ", yc =", yc, ", Iext1 =", Iext1, ", slope =", slope, ", x0cr =", x0cr, ", rx0 =", rx0);
        print("x1_lo =", x1_lo, ", x1_hi =", x1_hi);
        print("z_lo =", z_lo, ", z_hi =", z_hi);
        print("x1eq_lo =", x1eq_lo, ", x1eq_hi =", x1eq_hi);
        print("min(x1eq0)=", min(x1eq0), ", , max(x1eq0)=", max(x1eq0));
        if (DEBUG >= 1){
            print("x1eq0 =", x1eq0);
        }
        print("x1init_lo =", x1init_lo, ", x1init_hi =", x1init_hi);
        print("zinit_lo =", zinit_lo, ", zinit_hi =", zinit_hi);
        print("tau1_lo =", tau1_lo, ", tau1_hi =", tau1_hi, ", tau1_p =", tau1_p, ", tau1_loc =", tau1_loc, ", tau1_scale =", tau1_scale, ", tau1_pdf =", tau1_pdf);
        print("tau0_lo =", tau0_lo, ", tau0_hi =", tau0_hi, ", tau0_p =", tau0_p, ", tau0_loc =", tau0_loc, ", tau0_scale =", tau0_scale, ", tau0_pdf =", tau0_pdf);
        print("K_lo =", K_lo, ", K_hi =", K_hi, ", K_p =", K_p, ", K_loc =", K_loc, ", K_scale =", K_scale, ", K_pdf =", K_pdf);
        print("MC_lo =", MC_lo, ", MC_hi =", MC_hi, " MC_p =", MC_p, ", MC_pdf =", MC_pdf);
        if (DEBUG == 3){
        for (ii in 1:n_regions) {
                print("MC_loc[", ii, "] = ", MC_loc[ii]);
                print("MC_loc[", ii, "] = ", MC_loc[ii]);
            }
            for (ii in 1:n_regions) {
                print("MC_scale[", ii, "] = ", MC_scale[ii]);
                print("MC_scale[", ii, "] = ", MC_scale[ii]);
            }
        }
        print("euler_method =", euler_method);
        print("dt =", dt);
        print("sig_eq_scale =", sig_eq_scale, " sig_eq_lo =", sig_eq_lo, ", sig_eq_hi =", sig_eq_hi, ", sig_eq_p =", sig_eq_p, ", sig_eq_loc =", sig_eq_loc, ", sig_eq_scale =", sig_eq_scale, ", sig_eq_pdf =", sig_eq_pdf);
        print("sig_init_scale =", sig_init_scale, " sig_init_lo =", sig_init_lo, ", sig_init_hi =", sig_init_hi, ", sig_init_p =", sig_init_p, ", sig_init_loc =", sig_init_loc, ", sig_init_scale =", sig_init_scale, ", sig_init_pdf =", sig_init_pdf);
        print("sig_scale =", sig_scale, ", sig_lo =", sig_lo, ", sig_hi =", sig_hi, ", sig_p =", sig_p, ", sig_loc =", sig_loc, ", sig_scale =", sig_scale, ", sig_pdf =", sig_pdf);
        print("observation_expression =", observation_expression, ", observation_model =", observation_model);
        print("min(mixing) =",min(mixing), ", max(mixing) =", max(mixing));
        if (DEBUG == 1){
            vector[n_active_regions] maxmixing;
            vector[n_active_regions] minmixing;
            for (ii in 1:n_active_regions) {
                vector[n_signals] mixingcol;
                mixingcol = col(mixing, ii);
                maxmixing[ii] = max(mixingcol);
                minmixing[ii] = min(mixingcol);
            }
            print("min(mixing(active_regions))=", minmixing);
            print("max(mixing(active_regions))=", maxmixing);
        } else {
            for (ii in 1:n_active_regions) {
                print("mixing[", ii, "] = ", col(mixing, ii));
            }
        }
        print("min(signals) =", min(signals), ", max(signals) =", max(signals));
        if (DEBUG == 1){
            vector[n_times] maxsignals;
            vector[n_times] minsignals;
            for (tt in 1:n_times) {
                vector[n_signals] signalstimes;
                signalstimes = to_vector(signals[tt]);
                maxsignals[tt] = max(signalstimes);
                minsignals[tt] = min(signalstimes);
            }
            print("min(signals(cols))=", minsignals);
            print("max(signals(cols))=", maxsignals);
        } else if (DEBUG >= 3){
            for (tt in 1:n_times) {
                print("signals[", tt, "] = ", signals[tt]);
            }
        }
        print("eps_lo =", eps_lo, ", eps_hi =", eps_hi, ", eps_p =", eps_p, ", eps_loc =", eps_loc, ", eps_scale =", eps_scale, ", eps_pdf =", eps_pdf);
        print("scale_signal_lo =", scale_signal_lo, ", scale_signal_hi =", scale_signal_hi, ", scale_signal_p =", scale_signal_p, ", scale_signal_loc =", scale_signal_loc, ", scale_signal_scale =", scale_signal_scale, ", scale_signal_pdf =", scale_signal_pdf);
        print("offset_signal_lo =", offset_signal_lo, ", offset_signal_hi =", offset_signal_hi, ", offset_signal_p =", offset_signal_p, ", offset_signal_loc =", offset_signal_loc, ", offset_signal_scalec =", offset_signal_scale, ", offset_signal_pdf =", offset_signal_pdf);
    }
}


parameters {

    /* Generative model */
    /* Epileptor */
    vector<lower=x1eq_lo, upper=x1eq_hi>[n_regions] x1eq; // x1 equilibrium point coordinate
    vector<lower=x1init_lo, upper=x1init_hi>[n_active_regions] x1init; // x1 equilibrium point coordinate
    matrix<lower=x1_lo, upper=x1_hi>[n_times, n_active_regions] x1; // x1 state variable
    matrix<lower=z_lo, upper=z_hi>[n_times, n_active_regions] z; // z state variable
    real<lower=tau1_lo, upper=tau1_hi> tau1_star; // time scale [n_active_regions]
    real<lower=K_lo, upper=tau0_hi> tau0_star; // time scale separation [n_active_regions]
    /* Coupling */
    real<lower=K_lo, upper=K_hi> K_star; // global coupling scaling
    matrix<lower=0.0>[n_regions, n_regions] MC_star; // Model connectivity

    /* Integration */
    real<lower=sig_lo, upper=sig_hi> sig_star; // variance of phase flow, i.e., dynamic noise
    real<lower=sig_eq_lo, upper=sig_eq_hi> sig_eq_star; // variance of equilibrium point
    real<lower=sig_init_lo, upper=sig_init_hi> sig_init_star; // variance of initial condition

    /* Observation model */
    real<lower=eps_lo, upper=eps_hi> eps_star; // variance of observation noise
    real<lower=scale_signal_lo, upper=scale_signal_hi> scale_signal_star; // observation signal scaling
    real<lower=offset_signal_lo, upper=offset_signal_hi> offset_signal_star; // observation signal offset

}


transformed parameters {

    /* Generative model */
    /* Epileptor */
    vector[n_regions] zeq; // z equilibrium point coordinate
    vector[n_regions] x0; // excitability parameter
    real<lower=0.0> tau1; // time scale [n_active_regions]
    real<lower=0.0> tau0; // time scale separation [n_active_regions]
    /* Coupling */
    real<lower=0.0> K; // global coupling scaling
    matrix<lower=0.0>[n_regions, n_regions] MC; // Model connectivity
    vector[n_regions] coupling_eq; // coupling at equilibrium
    matrix[n_times, n_active_regions] coupling; // actual effective coupling per time point

    /* Integration */
    real<lower=0.0> sig; // variance of phase flow, i.e., dynamic noise
    real<lower=0.0> sig_eq; // variance of equilibrium point
    real<lower=0.0> sig_init; // variance of initial condition

    /* Observation model */
    real<lower=0.0> eps; // variance of observation noise
    real<lower=0.0> scale_signal; // observation signal scaling
    real offset_signal; // observation signal offset
        
    tau1 = tau1_star * tau1_scale + tau1_loc;
    if (DEBUG >= 0){
        print("tau1=", tau1);
    }
    tau0 = tau0_star * tau0_scale + tau0_loc;
    if (DEBUG >= 0){
        print("tau0=", tau0);
    }
    K = K_star * K_scale + K_loc;
    if (DEBUG >= 0){
        print("K=", K);
    }
    MC = MC_star * MC_scale + MC_loc;
    if (DEBUG >= 0){
        print("min(MC)=", min(MC), ", max(MC)=", max(MC));
        if (DEBUG == 1){
            vector[n_regions] maxMC;
            vector[n_regions] minMC;
            print("x1eq=", x1eq);
            print("x1eq=", x1eq);
            for (ii in 1:n_regions) {
                vector[n_regions] colMC;
                colMC = col(MC, ii);
                maxMC[ii] = max(colMC);
                minMC[ii] = min(colMC);
            }
            print("min(MC(cols))=", minMC);
            print("max(MC(cols))=", maxMC);
        } else if (DEBUG >= 2) {
            for (ii in 1:n_regions) {
                print("MC[", ii, "] = ", MC[ii]);
            }
        }
    }
    sig_eq = sig_eq_star * sig_eq_scale + sig_eq_loc;
    if (DEBUG >= 0){
        print("sig_eq=", sig_eq);
    }
    sig_init = sig_init_star * sig_init_scale + sig_init_loc;
    if (DEBUG >= 0){
        print("sig_init=", sig_init);
    }
    sig = sig_star * sig_scale + sig_loc;
    if (DEBUG >= 0){
        print("sig=", sig);
    }
    eps = eps_star * eps_scale + eps_loc;
    if (DEBUG >= 0){
        print("eps=", eps);
    }
    scale_signal = scale_signal_star * scale_signal_scale + scale_signal_loc;
    if (DEBUG >= 0){
        print("scale_signal=", scale_signal);
    }
    offset_signal = offset_signal_star * offset_signal_scale + offset_signal_loc;
    if (DEBUG >= 0){
        print("offset_signal=", offset_signal);
    }

    /* zeq */
    for (ii in 1:n_regions) {
        // solving for zeq (z->0, tau1->1)
        zeq[ii] = EpileptorDP2D_fun_x1(x1eq[ii], 0.0, yc, Iext1, a, db, d, slope, 1.0);
    }
    if (DEBUG >= 0){
        print("min(zeq)=", min(zeq), ", max(zeq)=", max(zeq));
        if (DEBUG >= 1){
            print("zeq=", zeq);
        }
    }

    /* Coupling
    We place it here for the moment because it has a high diagnostic value */
    for (ii in 1:n_regions) {
        coupling_eq[ii] = 0.0;
        for (jj in 1:n_regions) {
            if (ii!=jj) {
                    coupling_eq[ii] = coupling_eq[ii] + MC[ii, jj] * (x1eq[jj] - x1eq[ii]);
            }
        }
    }
    if (DEBUG >= 0){
        print("min(coupling_eq)=", min(coupling_eq), ", max(coupling_eq)=", max(coupling_eq));
        if (DEBUG >= 1){
            print("coupling_eq=", coupling_eq);
        }
    }

    for (tt in 1:n_times) {
        for (ii in 1:n_active_regions) {
            coupling[tt, ii] = 0.0;
            // coupling active -> active regions excluding self-coupling
            for (jj in 1:n_active_regions) {
               if (ii!=jj) {
                   coupling[tt, ii] =
                              coupling[tt, ii] + MC[active_regions[ii], active_regions[jj]] * (x1[tt, jj] - x1[tt, ii]);
               }
            }
            // coupling nonactive -> active regions
            for (jj in 1:n_nonactive_regions) {
               // non active regions are assumed to always stay close to their equilibrium point
               coupling[tt, ii] = coupling[tt, ii] +
                             MC[active_regions[ii], nonactive_regions[jj]] * (x1eq[nonactive_regions[jj]] - x1[tt, ii]);
            }
        }
        if (DEBUG == 1) {
            print("min(coupling[", tt, "]) = ", min(coupling[tt]), ", max(coupling[", tt, "]) = ", max(coupling[tt]));
        }
        else if (DEBUG >= 2){
            print("coupling[", tt, "] = ", coupling[tt]);
        }
    }
    if (DEBUG >= 0){
        print("min(coupling)=", min(coupling), ", max(coupling)=", max(coupling));
    }

    /* x0 */
    for (ii in 1:n_regions) {
        // solving for x0 (x0->0, tau0->1, tau1->1)
        x0[ii] = EpileptorDP2D_fun_z_lin(x1eq[ii], zeq[ii], 0.0, K * coupling_eq[ii], 1.0, 1.0) / 4.0;
    }
    if (DEBUG >= 0){
        print("min(x0)=", min(x0), ", max(x0)=", max(x0));
        if (DEBUG >= 1){
            print("x0=", x0);
        }
    }
}


model {

    vector[n_active_regions] observation;
    real df;

    /* Sampling of global coupling scaling */
    K_star ~ sample(K_pdf, K_p);
    if (DEBUG >= 0){
        print("K_star=", K_star);
    }

    /* Sampling of the various variances */
    sig_eq_star ~ sample(sig_eq_pdf, sig_eq_p);
    if (DEBUG >= 0){
        print("sig_eq_star=", sig_eq_star);
    }
    sig_init_star ~ sample(sig_init_pdf, sig_init_p);
    if (DEBUG >= 0){
        print("sig_init_star=", sig_p);
    }
    sig_star ~ sample(sig_pdf, sig_p);
    if (DEBUG >= 0){
        print("sig_star=", sig_star);
    }

    /* Sampling of x1 equilibrium point coordinate and effective connectivity */
    for (ii in 1:n_regions) {
        x1eq[ii] ~ normal(x1eq0[ii], sig_eq);
        for (jj in 1:n_regions) {
            MC_star[ii, jj] ~ sample(MC_pdf, MC_p);
        }
    }
    if (DEBUG >= 0){
        print("min(x1eq)=", min(x1eq), ", max(x1eq)=", max(x1eq));
        print("min(MC_star)=", min(MC_star), ", max(MC_star)=", max(MC_star));
        if (DEBUG == 1){
            vector[n_regions] maxMC_star;
            vector[n_regions] minMC_star;
            print("x1eq=", x1eq);
            print("x1eq=", x1eq);
            for (ii in 1:n_regions) {
                vector[n_regions] colMC_star;
                colMC_star = col(MC, ii);
                maxMC_star[ii] = max(colMC_star);
                minMC_star[ii] = min(colMC_star);
            }
            print("min(MC_star(cols))=", minMC_star);
            print("max(MC_star(cols))=", maxMC_star);
        } else if (DEBUG >= 2) {
            for (ii in 1:n_regions) {
                print("MC_star[", ii, "] = ", MC_star[ii]);
            }
        }
    }

    /* Sampling of initial condition*/
    for (ii in 1:n_active_regions) {
        x1[ii, 1] ~ normal(x1eq[active_regions[ii]], sig_init);
        z[ii, 1] ~ normal(zeq[active_regions[ii]], sig_init);
//        tau0_star[ii] ~ sample(tau1_pdf, tau1_p);
//        tau0_star[ii] ~ sample(tau0_pdf, tau0_shap);
    }
    if (DEBUG >= 0){
        print("min(x1init)=", min(x1[,1]), ", max(x1init)=", max(x1[,1]));
        print("min(zinit)=", min(z[,1]), ", max(zinit)=", max(z[,1]));
        if (DEBUG >= 1){
            print("x1init=", x1[,1]);
            print("zinit=", z[,1]);
        }
    }

    /* Sampling of time scales */
    tau1_star ~ sample(tau1_pdf, tau1_p);
    if (DEBUG >= 0){
        print("tau1_star=", tau1_star);
    }
    tau0_star ~ sample(tau1_pdf, tau0_p);
    if (DEBUG >= 0){
        print("tau0_star=", tau0_star);
    }

    /* Sampling of observation scaling and offset */
    scale_signal_star ~ sample(scale_signal_pdf, scale_signal_p);
    if (DEBUG >= 0){
        print("scale_signal_star=", scale_signal_star);
    }
    offset_signal_star ~ sample(offset_signal_pdf, offset_signal_p);
    if (DEBUG >= 0){
        print("offset_signal_star=", offset_signal_star);
    }

    /* Integrate & predict  */
    for (tt in 2:n_times) {
        /* Auto-regressive generative model  */
        if (euler_method==-1){ // backward euler method
            for (ii in 1:n_active_regions) {
                df = EpileptorDP2D_fun_x1(x1[tt, ii], z[tt, ii], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                x1[tt, ii] ~ normal(x1[tt-1, ii] + dt*df, sig); // T[x1_lo, x1_hi];
                df = EpileptorDP2D_fun_z_lin(x1[tt, ii], z[tt, ii], x0[active_regions[ii]], K*coupling[tt, ii], tau0, tau1); // tau0
                z[tt, ii] ~ normal(z[tt-1, ii] + dt*df, sig); // T[z_lo, z_hi];
            }
            // TODO: code for midpoint euler method
        } else {// forward euler method
            for (ii in 1:n_active_regions) {
                df = EpileptorDP2D_fun_x1(x1[tt-1, ii], z[tt-1, ii], yc, Iext1, a, db, d, slope, tau1); //tau1[ii]
                x1[tt, ii] ~ normal(x1[tt-1, ii] + dt*df, sig); // T[x1_lo, x1_hi];
                df = EpileptorDP2D_fun_z_lin(x1[tt-1, ii], z[tt-1, ii], x0[active_regions[ii]], K*coupling[tt-1, ii], tau0, tau1); // tau0[ii]
                z[tt, ii] ~ normal(z[tt-1, ii] + dt*df, sig); // T[z_lo, z_hi];
            }
        }
        if (DEBUG == 1) {
            print("min(x1[", tt, "]) = ", min(x1[tt]), ", max(x1[", tt, "]) = ", max(x1[tt]));
            print("min(z[", tt, "]) = ", min(z[tt]), ", max(z[", tt, "]) = ", max(z[tt]));
        }
        else if (DEBUG >= 2){
            print("x1[", tt, "] = ", x1[tt]);
            print("z[", tt, "] = ", z[tt]);
        }
        if (SIMULATE > 0){
            /* Observation model  */
            if (observation_expression == 0) {
                observation = to_vector(x1[tt]);
            } else if (observation_expression == 1){
                observation = (to_vector(x1[tt]) - x1eq) / 2.0;
            } else {
                observation = (to_vector(x1[tt]) - x1eq + to_vector(z[tt]) - zeq) / 2.75;
            }
            if (DEBUG == 1) {
                print("min(observation[", tt, "]) = ", min(observation), ", max(observation[", tt, "]) = ", max(observation));
            }
            else if (DEBUG >= 2){
                print("observation[", tt, "] = ", observation);
            }
            if  (observation_model == 0) {
                // seeg log power: observation with some log mixing, scaling and offset_signal
                signals[tt] ~ normal(scale_signal * log(mixing * exp(observation)) + offset_signal, eps);
            } else if (observation_model == 1){
                // observation with some linear mixing, scaling and offset_signal
                signals[tt] ~ normal(scale_signal * mixing * observation + offset_signal, eps);
            } else {
                // observation with some scaling and offset_signal, without mixing
                signals[tt] ~ normal(scale_signal * observation + offset_signal, eps);
            }
        }
    }
    if (DEBUG >= 0){
        print("min(x1)=", min(x1), ", max(x1)=", max(x1));
        print("min(z)=", min(z), ", max(z)=", max(z));
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
    if (DEBUG >= 0) {
        print("min(ModelEpileptogenicity) = ", min(ModelEpileptogenicity), ", max(ModelEpileptogenicity) = ", max(ModelEpileptogenicity));
        print("min(PathologicalExcitability) = ", min(PathologicalExcitability), ", max(PathologicalExcitability) = ", max(PathologicalExcitability));
        if (DEBUG >= 1){
            print("ModelEpileptogenicity=", ModelEpileptogenicity);
            print("PathologicalExcitability=", PathologicalExcitability);
        }
    }

    {
        vector[n_active_regions] observation;

        for (tt in 1:n_times) {
            /* Observation model  */
            if (observation_expression == 0) {
                observation = to_vector(x1[tt]);
            } else if (observation_expression == 1){
                observation = (to_vector(x1[tt]) - x1eq) / 2.0;
            } else {
                observation = (to_vector(x1[tt]) - x1eq + to_vector(z[tt])- zeq) / 2.75;
            }
            if  (observation_model == 0) {
                // seeg log power
                fit_signals[tt] = to_row_vector((scale_signal * (log(mixing * exp(observation)) + offset_signal)));
            } else if (observation_model == 1){
                // seeg power: just x1 with some mixing, scaling and offset_signal
                fit_signals[tt] = to_row_vector(scale_signal * mixing * observation + offset_signal);
            } else {
                // lfp power: just with some x1 scaling and offset_signal
                fit_signals[tt] = to_row_vector(scale_signal * observation + offset_signal);
            }
            if (DEBUG == 1) {
                print("min(fit_signals[", tt, "]) = ", min(fit_signals[tt]), ", max(fit_signals[", tt, "]) = ", max(fit_signals[tt]));
            }
            else if (DEBUG >= 3){
                print("fit_signals[", tt, "] = ", fit_signals[tt]);
            }
        }
    }
    if (DEBUG >= 0) {
        print("min(fit_signals) = ", min(fit_signals), ", max(fit_signals) = ", max(fit_signals));
    }
}
