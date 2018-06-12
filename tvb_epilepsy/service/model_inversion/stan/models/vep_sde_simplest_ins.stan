functions {

    matrix vector_differencing(row_vector x) {
        matrix[num_elements(x), num_elements(x)] D;
        for (i in 1:num_elements(x)) {
            D[i] = x - x[i];
        }
        return D;
    }

    row_vector x_step(row_vector x, row_vector z, real I1, real time_scale) { //, row_vector x_eta, real sigma
        int nn = num_elements(x);
        row_vector[nn] x_next;
        row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
        row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
        x_next = x + (time_scale * dx); # + x_eta * sigma;
        return x_next;
    }

    row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, vector Ic, real x_eq_def, real time_scale,
                      row_vector z_eta, real sigma, real tau0) {
        int nn = num_elements(z);
        row_vector[nn] z_next;
        matrix[nn, nn] D = vector_differencing(x);
        // Ic = Ic_i = sum_{j in nonactive regions} [w_ij]
        // gx(nonactive->active) = Ic * (x_j - x_i) = Ic * (x_eq_def - x)
        row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D) + Ic .* to_vector(x_eq_def - x));
        row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
        z_next = z + (time_scale * dz) + z_eta * sigma;
        return z_next;
    }
}

data {
    int DEBUG;
    int SIMULATE;
    int nn;
    int nt; // 1012
    int ns;
    real I1; //=3.1
    real tau0; //=10, [3, 30]
    real dt; //~= 0.1 (used 0.0976562)
    real x0_hi; // -4.0, [-3.0, -4.0]
    // real x0_std; // ~0.5
    // real x0_lo;  // 0.0
    row_vector [nn] x0_star_mu; // x0_hi - x0_mean = 0.0 - ~[2.2 to 2.5] = ~[2.2 to 2.5]
    row_vector [nn] x0_star_std; // minimum((x0_hi - x0_lo)/8.0, x0_star_mu/5.0) ~= 0.5
    // row_vector [nn] x0_mu;  // healthy: -2.5, sick ~=-2.0, max = [-3.0, -4.0], min = -1.0
    real x_eq_def; // = -5.0/3 the value of all healhty non-active node
    row_vector [nn] x_init_mu; // in [-2.0, -1.0], used -1.566
    row_vector [nn] z_init_mu; // in [2.9, 4.5], used 3.060
    real x_init_std; // 0.0333
    real z_init_std; // 0.0333/2
    real time_scale_mu; // 0.5
    real time_scale_std; // 0.0667
    real k_mu; // 3.448 = 3 * 100 / n_regions(=87)
    real k_std; // 0.575 = k_mu/6
    real sigma_mu; // =0.01
    real sigma_std; // =0.01/3
    real epsilon_mu; //=0.1
    real epsilon_std; //=0.1/3
    real offset_mu;  //=0.0
    real offset_std; //=1.0
    real amplitude_mu; //=1.0
    real amplitude_std; //=1.0/6
    real amplitude_lo; // =0.3
    matrix[ns, nn] gain;
    row_vector[ns] seeg_log_power[nt];
    vector[nn] Ic;
    matrix<lower=0.0, upper=1.0>[nn, nn] SC;
}

transformed data {
    real sqrtdt = sqrt(dt);
    real amplitude_zscore = amplitude_std/amplitude_mu;
    real epsilon_zscore = epsilon_std/epsilon_mu;
    real time_scale = time_scale_mu;
    real k = k_mu;
    real sigma_zscore = sigma_std/sigma_mu;
    row_vector[nn] x0_star_zscore = x0_star_std ./ x0_star_mu;
    //matrix[ns, nn] log_gain = log(gain);
    matrix [nn, nn] SC_ = SC;
    for (i in 1:nn) SC_[i, i] = 0;
    SC_ = SC_ / max(SC_) * rows(SC_);
}

parameters {
    // integrate and predict
    real<lower=-1.0, upper=1.0> offset_star;
    real<lower=-3.0, upper=3.0> amplitude_star;
    real<upper=3.0> epsilon_star;
    real<upper=3.0> sigma_star;
    row_vector<upper=3.0> [nn] x0_star;

    // time-series state non-centering:
    row_vector[nn] x_init_star;
    row_vector[nn] z_init_star;
    // row_vector[nn] x_eta_star[nt - 1];
    row_vector[nn] z_eta_star[nt - 1];
}

transformed parameters {
    real offset = offset_mu + offset_star * offset_std;
    real amplitude = amplitude_mu * exp(amplitude_zscore * amplitude_star);
    real epsilon = epsilon_mu * exp(epsilon_zscore * epsilon_star);
    real sigma = sigma_mu * exp(sigma_zscore * sigma_star);
    row_vector[nn] x0 = x0_hi - (x0_star_mu .* exp(x0_star_zscore .* x0_star));
    row_vector[nn] x_init = x_init_mu + x_init_star * x_init_std;
    row_vector[nn] z_init = z_init_mu + z_init_star * z_init_std;
    row_vector[nn] x[nt];
    row_vector[nn] z[nt];
    row_vector[ns] mu_seeg_log_power[nt];

    x[1] = x_init; // - 1.5;
    z[1] = z_init; // 3.0;
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, dt*time_scale); //, x_eta_star[t], sqrtdt*sigma
        z[t+1] = z_step(x[t], z[t], x0, k*SC, Ic, x_eq_def, dt*time_scale, z_eta_star[t], sqrtdt*sigma, tau0);
    }

    for (t in 1:nt)
        mu_seeg_log_power[t] = amplitude * (log(gain * exp(x[t]'-x_eq_def)) + offset)';
}

model {
    offset_star ~ normal(0.0, 1.0);
    amplitude_star ~ normal(0.0, 1.0);
    epsilon_star ~ normal(0.0, 1.0);
    sigma_star ~ normal(0.0, 1.0);
    to_row_vector(x0_star) ~ normal(0.0, 1.0);
    x_init_star ~ normal(0.0, 1.0);
    z_init_star ~ normal(0.0, 1.0);

    for (t in 1:(nt - 1)) {
        // to_vector(x_eta_star[t]) ~ normal(0.0, 1.0);
        to_vector(z_eta_star[t]) ~ normal(0.0, 1.0);
    }

    if (SIMULATE<1)
        for (t in 1:nt)
            seeg_log_power[t] ~ normal(mu_seeg_log_power[t], epsilon);
}

generated quantities {
    row_vector[ns] log_likelihood[nt];
    for (t in 1:nt) {
        for (s in 1:ns) {
            log_likelihood[t][s] = normal_lpdf(seeg_log_power[t][s] |  mu_seeg_log_power[t][s], epsilon);
        }
    }
}
