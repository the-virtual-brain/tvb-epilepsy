functions {

    matrix vector_differencing(row_vector x) {
        matrix[num_elements(x), num_elements(x)] D;
        for (i in 1:num_elements(x)) {
            D[i] = x - x[i];
        }
        return D;
    }

    row_vector x_step(row_vector x, row_vector z, real I1, real time_scale, row_vector x_eta, real sigma) {
        int nn = num_elements(x);
        row_vector[nn] x_next;
        row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
        row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
        x_next = x + (time_scale * dx) + x_eta * sigma;
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
    row_vector [nn] x0_star_mu;
    row_vector [nn] x0_star_std; // 1/3 of the smaller distance of mu to boundary (either max or min value) for each x0
    // row_vector [nn] x0_mu;  // healthy: -2.5, sick ~=-2.0, max = [-3.0, -4.0], min = -1.0
    // real x0_std;
    // real x0_lo;
    real x0_hi; // [-3.0, -4.0]
    real x_eq_def; // = -5.0/3 the value of all healhty non-active node
    row_vector [nn] x_init_mu; // in [-2.0, -1.0], used -1.566
    row_vector [nn] z_init_mu; // in [2.9, 4.5], used 3.060
    real init_std; // 0.0333
    real time_scale_mu; // 0.5
    real time_scale_std; // 0.0667
    real k_mu; // 0.575 = 50 / n_regions(=87)
    real k_std; // 0.096 = k_mu/6
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
    real time_scale_zscore = time_scale_std/time_scale_mu;
    real k_zscore = k_std/k_mu;
    real amplitude_zscore = amplitude_std/amplitude_mu;
    real epsilon_zscore = epsilon_std/epsilon_mu;
    real sigma_zscore = sigma_std/sigma_mu;
    row_vector[nn] x0_star_zscore = x0_star_std ./ x0_star_mu;
    //matrix[ns, nn] log_gain = log(gain);
    matrix [nn, nn] SC_ = SC;
    for (i in 1:nn) SC_[i, i] = 0;
        SC_ /= max(SC_) * rows(SC_);
}

parameters {
    // integrate and predict
    row_vector<upper=3.0> [nn] x0_star;
    real epsilon_star;
    real<lower=-3.0, upper=3.0> amplitude_star;
    real offset;
    real<lower=-3.0, upper=3.0> time_scale_star;

    // time-series state non-centering:
    row_vector[nn] x_init;
    row_vector[nn] z_init;
    row_vector[nn] x_eta[nt - 1];
    row_vector[nn] z_eta[nt - 1];
    real<lower=0.0> sigma_star;
    real k_star;
}

transformed parameters {
    real amplitude = amplitude_mu * exp(amplitude_zscore * amplitude_star);
    real epsilon = epsilon_mu * exp(epsilon_zscore * epsilon_star); //0.05
    real sigma = sigma_mu * exp(sigma_zscore * sigma_star); //0.053 * exp(0.1 * sigma_star);
    real time_scale = time_scale_mu * exp(time_scale_zscore * time_scale_star); //0.15 * exp(0.4 * time_scale_star - 1.0);
    real k = k_mu * exp(k_zscore * k_star); //1e-3 * exp(0.5 * k_star);
    row_vector[nn] x0 = x0_hi - (x0_star_mu .* exp(x0_star_zscore .* x0_star));
    row_vector[nn] x[nt];
    row_vector[nn] z[nt];
    row_vector[ns] mu_seeg_log_power[nt];

    x[1] = x_init; // - 1.5;
    z[1] = z_init; // 3.0;
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, dt*time_scale, x_eta[t], sqrtdt*sigma);
        z[t+1] = z_step(x[t], z[t], x0, k*SC, Ic, x_eq_def, dt*time_scale, z_eta[t], sqrtdt*sigma, tau0);
    }
    for (t in 1:nt)
        mu_seeg_log_power[t] = amplitude * (log(gain * exp(x[t]')) + offset)';
}

model {
    to_row_vector(x0_star) ~ normal(0, 1.0);
    k_star ~ normal(0, 1);
    x_init ~ normal(x_init_mu, init_std); // 0.0, 1.0
    z_init ~ normal(z_init_mu, init_std); // 0.0, 1.0
    sigma_star ~ normal(0, 1.0);
    time_scale_star ~ normal(0, 1.0);

    amplitude_star ~ normal(0.0, 1.0);
    offset ~ normal(offset_mu, offset_std);
    epsilon_star ~ normal(0, 1.0);

    for (t in 1:(nt - 1)) {
        to_vector(x_eta[t]) ~ normal(0, 1);
        to_vector(z_eta[t]) ~ normal(0, 1);
    }

    if (SIMULATE<1)
        for (t in 1:nt)
            seeg_log_power[t] ~ normal(mu_seeg_log_power[t], epsilon);
}

/*
generated quantities {
    row_vector[ns] gq_seeg_log_power[nt];
    for (t in 1:nt)
        for (i in 1:ns)
            gq_seeg_log_power[t][i] = normal_rng(mu_seeg_log_power[t][i], epsilon);
}
*/
