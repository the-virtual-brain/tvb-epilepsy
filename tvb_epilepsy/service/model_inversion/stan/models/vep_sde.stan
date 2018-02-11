functions {

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

    row_vector ode_step(int nx, row_vector x, row_vector df, real dt) {
        row_vector[nx] x_next = x + df * dt;
        return x_next;
    }

    row_vector sde_step(int nx, row_vector x, row_vector df, real dt, row_vector dWtsqrtdt) {
        row_vector[nx] x_next = ode_step(nx, x, df, dt) + dWtsqrtdt;
        return x_next;
    }

    matrix vector_differencing(int ni, int nj, row_vector xi, row_vector xj) {
        matrix[nj, ni] Dji;
        for (i in 1:ni) {
            Dji[i] = xj - xi[i];
        }
        return Dji;
    }

    row_vector calc_coupling(int ni, int nj, row_vector xi, row_vector xj, matrix MCij) {
        matrix[ni, nj] Dji = vector_differencing(ni, nj, xi, xj);
        row_vector[ni] coupling = to_row_vector(rows_dot_product(MCij, Dji));
        return coupling;
    }

    row_vector EpileptorDP2D_fun_x1(int nn, row_vector x1, row_vector z, real yc, real Iext1,
                                    real a, real db, real d, real slope, real tau1) {
        row_vector[nn] constants = rep_row_vector(Iext1 + yc, nn);
        row_vector[nn] fx1;
        for (ii in 1:nn) {
            // population 1
            if (x1[ii] <= 0.0) {
                // if_ydot0 = a * x1 ** 2 + (d - b) * y[0],
                fx1[ii] = a * x1[ii] * x1[ii] + db * x1[ii];
            } else {
                // d * y[0] - 0.6 * (y[1] - 4.0) ** 2 - slope,
                fx1[ii] =  z[ii] - 4.0;
                fx1[ii] = d * x1[ii] - 0.6 * fx1[ii] * fx1[ii] - slope;
            }
        }
        // ydot[0] = tau1 * (yc - y[1] + Iext1 - where(y[0] < 0.0, if_ydot0, else_ydot0) * y[0])
        fx1 = tau1 * (constants - z - fx1 .* x1);
        return fx1;
    }

    row_vector EpileptorDP2D_fun_z_lin(int nn, row_vector x1, row_vector z, row_vector x0, row_vector coupling,
                                       real tau0, real tau1) {
        // slow energy variable with a linear form (original Epileptor)
        // ydot[1] = tau1 * (4 * (y[0] - x0) + where(y[1] < 0.0, if_ydot1, else_ydot1) - y[1] + K * c_pop1) / tau0
        row_vector[nn] fz = 4.0 * (x1 - x0) - z - coupling;
        /*
        for (ii in 1:nn) {
            if (z[ii] < 0.0) {
                // if_ydot1 = - 0.1 * y[1] ** 7
                fz[ii] = fz[ii] - 0.1 * z[ii] * z[ii] * z[ii] * z[ii] * z[ii] * z[ii] * z[ii];
            }
        }
        */
        fz =  tau1 *  fz / tau0;
        return fz;
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
    int n_connections;

    /* Integer flags and indices for (non)active regions */
    int active_regions[n_active_regions];
    int nonactive_regions[n_nonactive_regions];

    /* _lo parameters' lower limits
       _hi parameters' higher limits
       _pdf integer index of the distribution to be used for sampling, for the moment among:
            0. uniform
            1. normal
            2. lognormal
            3. gamma
            4. exponential
            5. beta
       _p[1] distribution's first parameter
       _p[2] distribution's second parameter, if any, otherwise, _p[2] = _p[1] */

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
    /* x1eq_star parameter  */
    real x1eq_max;
    real<upper=0.0> x1eq_min;
    real<lower=0.0> x1eq_star_lo;
    real<lower=0.0> x1eq_star_hi;
    row_vector<upper=0.0>[n_regions] x1eq_star_loc;
    row_vector<lower=0.0>[n_regions] x1eq_star_scale;
    real x1eq_star_p[n_regions, 2];
    int<lower=0> x1eq_star_pdf;
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
    /* SC symmetric connectivity data */
    row_vector<lower=0.0>[n_connections] SC;
    /* MCsplit (model connectivity direction split) parameter (only normal distribution) */
    real<lower=0.0> MCsplit_lo;
    real<lower=0.0> MCsplit_hi;
    row_vector<lower=0.0, upper=1.0>[n_connections] MCsplit_loc;
    row_vector<lower=0.0>[n_connections] MCsplit_scale;
    /* MC_scale (model connectivity scale factor (multiplying standard deviation) */
    real<lower=0.0> MC_scale;

    /* Integration */
    real dt;
    /* Initial condition variability */
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
    int observation_model;
    matrix[n_signals, n_regions] mixing;
    row_vector[n_signals] signals[n_times];
    /* Observation variability parameter (default: lognormal distribution) */
    real<lower=0.0> eps_lo;
    real<lower=0.0> eps_hi;
    real eps_loc;
    real<lower=0.0> eps_scale;
    real eps_p[2];
    int<lower=0> eps_pdf;
    /* Observation signal scaling parameter (defaul: lognormal distribution) */
    real<lower=0.0> scale_signal_lo;
    real<lower=0.0> scale_signal_hi;
    real scale_signal_loc;
    real<lower=0.0> scale_signal_scale;
    real scale_signal_p[2];
    int<lower=0> scale_signal_pdf;
    /* Observation signal offset parameter (only normal distribution) */
    // real offset_signal_lo;
    // real offset_signal_hi;
    real offset_signal_p[2];
}


transformed data {
    // Calculate db parameter, which corresponds to parameter b for the 2D reduced Epileptor (Proix etal 2014)
    real db = d - b;
    real sqrtdt = sqrt(dt);
    row_vector[n_regions] zeros = rep_row_vector(0.0, n_regions);
    /* Transformation of low and high bounds for star parameters
     * following (x-loc) / scale transformation of pdf support */
    real tau1_star_lo = (tau1_lo - tau1_loc) / tau1_scale;
    real tau1_star_hi = (tau1_hi - tau1_loc) / tau1_scale;
    real tau0_star_lo = (tau0_lo - tau0_loc) / tau0_scale;
    real tau0_star_hi = (tau0_hi - tau0_loc) / tau0_scale;
    real K_star_lo = (K_lo - K_loc) / K_scale;
    real K_star_hi = (K_hi - K_loc) / K_scale;
    real sig_init_star_lo = (sig_init_lo - sig_init_loc) / sig_init_scale;
    real sig_init_star_hi = (sig_init_hi - sig_init_loc) / sig_init_scale;
    real sig_star_lo = (sig_lo - sig_loc) / sig_scale;
    real sig_star_hi = (sig_hi - sig_loc) / sig_scale;
    real eps_star_lo = (eps_lo - eps_loc) / eps_scale;
    real eps_star_hi = (eps_hi - eps_loc) / eps_scale;
    real scale_signal_star_lo = (scale_signal_lo - scale_signal_loc) / scale_signal_scale;
    real scale_signal_star_hi = (scale_signal_hi - scale_signal_loc) / scale_signal_scale;
    if (DEBUG > 0)
        print("tau1_star_lo=", tau1_star_lo, " tau0_star_lo=", tau0_star_lo, " K_star_lo=", K_star_lo,
              " sig_init_star_lo=", sig_init_star_lo, " sig_star_lo=", sig_star_lo,
              " eps_star_lo=", eps_star_lo, " scale_signal_star_lo=", scale_signal_star_lo);
}


parameters {

    /* Generative model */
    /* Epileptor */
    row_vector<lower=0.0, upper=-x1eq_max-x1eq_min>[n_regions] x1eq_star; //star of x1 equilibrium point coordinate, <lower=x1eq_2star_lo, upper=x1eq_2star_hi>
    row_vector<lower=x1init_lo, upper=x1init_hi>[n_regions] x1init; // x1 initial condition coordinate
    row_vector<lower=zinit_lo, upper=zinit_hi>[n_regions] zinit; // x1 initial condition coordinate
    real<lower=sig_init_star_lo, upper=sig_init_star_hi> sig_init_star; // variance of initial condition
    row_vector[n_active_regions] dX1t[n_times-1]; // x1 dWt only for active nodes
    row_vector[n_active_regions] dZt[n_times-1]; // z dWt     >>   >>    >>
    real<lower=tau1_star_lo, upper=tau1_star_hi> tau1_star; // time scale [n_active_regions]
    real<lower=tau0_star_lo, upper=tau0_star_hi> tau0_star; // time scale separation [n_active_regions]
    /* Coupling */
    real<lower=K_star_lo, upper=K_star_hi> K_star; // global coupling scaling
    row_vector<lower=MCsplit_lo, upper=MCsplit_hi>[n_connections] MCsplit; // Model connectivity direction split
    matrix<lower=0.0, upper=1.0>[n_connections, 2] MC_star; // Non-symmetric model connectivity
    /* SDE Integration */
    real<lower=sig_star_lo, upper=sig_star_hi> sig_star; // variance of phase flow, i.e., dynamic noise
    /* Observation model */
    real<lower=eps_star_lo, upper=eps_star_hi> eps_star; // variance of observation noise
    real<lower=scale_signal_star_lo, upper=scale_signal_star_hi> scale_signal_star; // observation signal scaling
    real offset_signal; // observation signal offset
}


transformed parameters {

    /* Observation model */
    row_vector[n_signals] fit_signals[n_times]; // expected output signal
    real<lower=0.0> eps = eps_star * eps_scale + eps_loc; // variance of observation noise
    // observation signal scaling:
    real<lower=0.0> scale_signal = scale_signal_star * scale_signal_scale + scale_signal_loc;

    /* Generative model */

    /* Epileptor */
    row_vector[n_regions] x1[n_times]; // x1 state variable
    row_vector[n_regions] z[n_times]; // z state variable
    row_vector[n_regions] x0; // x0, excitability parameter
    real<lower=0.0> tau1 = tau1_star * tau1_scale + tau1_loc; // time scale
    real<lower=0.0> tau0 = tau0_star * tau0_scale + tau0_loc; // time scale separation
    /* x1eq, x1 equilibrium point coordinate */
    row_vector[n_regions] x1eq = x1eq_max - (x1eq_star .* x1eq_star_scale + x1eq_star_loc);
    /* zeq, z equilibrium point coordinate */
    row_vector[n_regions] zeq = EpileptorDP2D_fun_x1(n_regions, x1eq, zeros, yc, Iext1, a, db, d, slope, 1.0);

    /* SDE Integration of auto-regressive generative model  */
    real<lower=0.0> sig_init = sig_init_star * sig_init_scale + sig_init_loc; // variance of initial condition
    real<lower=0.0> sig = sig_star * sig_scale + sig_loc; // variance of phase flow, i.e., dynamic noise

    /* Coupling */
    real<lower=0.0> K = K_star * K_scale + K_loc; // global coupling scaling
    row_vector[n_regions] coupling[n_times]; // actual effective coupling per time point
    row_vector[n_regions] coupling_eq; // coupling at equilibrium
    matrix[n_regions, n_regions] MC;
    {   int icon = 0;
        for (jj in 1:n_regions) {
            for (ii in 1:jj) {
                if (ii == jj)
                    MC[ii, jj] = 0.0; // diagonal (self-) MC
                else {
                    icon += 1;
                    MC[ii, jj] = MC_star[icon, 1]; // upper triangular MC
                    MC[jj, ii] = MC_star[icon, 2]; // lower triangular MC
                }
            }
        }
    }

    if (DEBUG > 0)
        print("tau1=", tau1, " tau0=", tau0, " K=", K,  " sig_init=", sig_init, " sig=", sig,
              " eps=", eps, " scale_signal=", scale_signal);
    if (DEBUG > 2)
        print("MC", MC);

    coupling_eq = calc_coupling(n_regions, n_regions, x1eq, x1eq, MC);

    /* x0, excitability parameter */
    x0 = EpileptorDP2D_fun_z_lin(n_regions, x1eq, zeq, zeros, K * coupling_eq, 1.0, 1.0) / 4.0;

    if (DEBUG > 1) {
        print("x0=", x0);
        print("x1eq=", x1eq);
        print("zeq=", zeq);
        print("coupling_eq=", coupling_eq);
    }

    /* Initial condition */
    x1[1] = x1init;
    z[1] = zinit;
    coupling[1] = calc_coupling(n_regions, n_regions, x1[1], x1[1], MC);
    if (DEBUG > 1) {
        print("x1init=", x1init);
        print("zinit=", zinit);
        print("coupling_init=", coupling[1]);
    }

    /* Integration of auto-regressive generative model  */
    {   row_vector[n_regions] df;
        row_vector[n_regions] observation;

        for (tt in 2:n_times) {
            df = EpileptorDP2D_fun_x1(n_regions, x1[tt-1], z[tt-1], yc, Iext1, a, db, d, slope, tau1);
            if (DEBUG > 2)
                print("tt=", tt, "dfx=", df);
            x1[tt] = ode_step(n_regions, x1[tt-1], df, dt);
            x1[tt, active_regions] = x1[tt, active_regions] + dX1t[tt-1] * sqrtdt;
            if (DEBUG > 2)
                print("x1[tt]", x1[tt]);
            coupling[tt] = calc_coupling(n_regions, n_regions, x1[tt], x1[tt], MC);
            df = EpileptorDP2D_fun_z_lin(n_regions, x1[tt-1], z[tt-1], x0, K*coupling[tt-1], tau0, tau1);
            if (DEBUG > 2)
                print("dfz=", df);
            z[tt] = ode_step(n_regions, z[tt-1], df, dt);
            z[tt, active_regions] = z[tt, active_regions] + dZt[tt-1] * sqrtdt;

        }
    }

    for (tt in 1:n_times) {
        if  (observation_model == 0) {
            // seeg log power: observation with some log mixing, scaling and offset_signal
            fit_signals[tt] = scale_signal * (log(mixing * exp(x1[tt]')) + offset_signal)';
        } else if (observation_model == 1){
            // observation with some linear mixing, scaling and offset_signal
            fit_signals[tt] = scale_signal * (mixing * x1[tt]' + offset_signal)';
        } else {
            // observation with some scaling and offset_signal, without mixing
            fit_signals[tt] = scale_signal * (x1[tt] + offset_signal);
        }
    }
}


model {

    row_vector[n_connections] MC_loc;

    /* Sampling of time scales */
    tau1_star ~ sample(tau1_pdf, tau1_p);
    tau0_star ~ sample(tau0_pdf, tau0_p);
    /* Sampling of global coupling scaling */
    K_star ~ sample(K_pdf, K_p);
    /* Sampling of model connectivity and its split parameter */
    MCsplit ~ normal(MCsplit_loc, MCsplit_scale);
    MC_loc = SC .* MCsplit;
    MC_star[:, 1] ~ normal(MC_loc, fabs(MC_loc) * MC_scale); // upper triangular MC
    MC_loc = SC .* (1-MCsplit);
    MC_star[:, 2] ~ normal(MC_loc, fabs(MC_loc) * MC_scale); // lower triangular MC

    /* Sampling of initial condition variance as well as dynamical noise strength */
    sig_init_star ~ sample(sig_init_pdf, sig_init_p);
    sig_star ~ sample(sig_pdf, sig_p);
    /* Sampling of x1 equilibrium point coordinate and initial condition */
    // x1eq ~ normal(x1eq_loc, sig_eq);
    for (ii in 1:n_regions) {
        x1eq_star[ii] ~ sample(x1eq_star_pdf, x1eq_star_p[ii]);
    }
    x1init ~ normal(x1eq, sig_init);
    zinit ~ normal(zeq, sig_init/2);

    /* Sampling of observation scaling and offset */
    eps_star ~ sample(eps_pdf, eps_p);
    scale_signal_star ~ sample(scale_signal_pdf, scale_signal_p);
    offset_signal ~ normal(offset_signal_p[1], offset_signal_p[2]);
    if (DEBUG > 0)
        print("offset_signal=", offset_signal);
    /* Integrate & predict  */
    for (tt in 1:(n_times-1)) {
        /* Auto-regressive generative model  */
        to_vector(dX1t[tt]) ~ normal(0, sig);
        to_vector(dZt[tt]) ~ normal(0, sig);
    }
    /* Observation model  */
    if (SIMULATE <= 0){
        // signals ~ normal(fit_signals, eps);
        for (tt in 1:n_times)
            signals[tt] ~ normal(fit_signals[tt], eps);
    }
}
