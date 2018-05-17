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

    real[] normal_mean_std_to_lognorm_mu_sigma(real mean, real std) {
        real logsm21 = log((std/mean) ** 2 + 1);
        real mustd[2];
        mu_sigma[1] = log(mean) - 0.5*logsm21;
        mu_sigma[2] = sqrt(logsm21);
        return mu_sigma
    }

    row_vector normal_mean_std_to_lognorm_mu(row_vector mean, row_vector std) {
        int nn = num_elements(mean);
        row_vector[nn] logsm21 = std./mean
        logsm21 = log(logsm21 .* logsm21 + 1);
        return log(mean) - 0.5*logsm21;
    }

    row_vector normal_mean_std_to_lognorm_sigma(row_vector mean, row_vector std) {
        int nn = num_elements(mean);
        row_vector[nn] logsm21 = std./mean
        logsm21 = log(logsm21 .* logsm21 + 1);
        return sqrt(logsm21);
    }

    real standard_normal_to_lognormal(real standard_normal, real mu, real sigma){
        return (exp(mu + sigma*standard_normal);
    }


    row_vector standard_normal_to_lognormal_row(row_vector standard_normal, row_vector mu, row_vector sigma){
        return exp(mu + sigma.*standard_normal;
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
    real[2] amplitude_mu_sigma = normal_mean_std_to_lognorm_mu_sigma(amplitude_mu, amplitude_std);
    real[2] epsilon_mu_sigma = normal_mean_std_to_lognorm_mu_sigma(epsilon_mu, epsilon_std);
    real[2] time_scale_mu_sigma = normal_mean_std_to_lognorm_mu_sigma(time_scale_mu, time_scale_std);
    // real[2] k_mu_sigma = normal_mean_std_to_lognorm_mu_sigma(k_mu, k_std);
    real k = k_mu;
    real[2] sigma_mu_sigma = normal_mean_std_to_lognorm_mu_sigma(sigma_mu, sigma_std);
    row_vector[nn] x0_logmu = normal_mean_std_to_lognorm_mu(x0_hi-x0_star_mu, x0_star_std);
    row_vector[nn] x0_sigma = normal_mean_std_to_lognorm_sigma(x0_hi-x0_star_mu, x0_star_std);
    //matrix[ns, nn] log_gain = log(gain);
    matrix [nn, nn] SC_ = SC;
    for (i in 1:nn) SC_[i, i] = 0;
    SC_ = SC_ / max(SC_) * rows(SC_);
}

parameters {
    // integrate and predict
    row_vector [nn] x0_star;
    real epsilon_star;
    real amplitude_star;
    real offset_star;
    real sigma_star;
    real time_scale_star;
    // real k_star;

    // time-series state non-centering:
    row_vector[nn] x_init_star;
    row_vector[nn] z_init_star;
    // row_vector[nn] x_eta[nt - 1];
    row_vector[nn] z_eta[nt - 1];

}

transformed parameters {
    real offset = offset_mu + offset_star * offset_std;
    real amplitude = standard_normal_to_lognormal(amplitude_star, amplitude_mu_sigma[1], amplitude_mu_sigma[2]);
    real epsilon = standard_normal_to_lognormal(epsilon_star, epsilon_mu_sigma[1], epsilon_mu_sigma[2]);
    real sigma = standard_normal_to_lognormal(sigma_star, sigma_mu_sigma[1], sigma_mu_sigma[2]);
    real time_scale = standard_normal_to_lognormal(time_scale_star, time_scale_mu_sigma[1], time_scale_mu_sigma[2]);
    // real k = standard_normal_to_lognormal(k_star, k_mu_sigma[1], k_mu_sigma[2]);
    row_vector[nn] x0 = x0_hi - standard_normal_to_lognormal_row(x0_star, x0_logmu, x0_sigma);
    row_vector[nn] x_init = x_init_mu + x_init_star * x_init_std;
    row_vector[nn] z_init = z_init_mu + z_init_star * z_init_std;
    row_vector[nn] x[nt];
    row_vector[nn] z[nt];
    row_vector[ns] mu_seeg_log_power[nt];

    x[1] = x_init; // - 1.5;
    z[1] = z_init; // 3.0;
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, dt*time_scale); //, x_eta[t], sqrtdt*sigma
        z[t+1] = z_step(x[t], z[t], x0, k*SC, Ic, x_eq_def, dt*time_scale, z_eta[t], sqrtdt*sigma, tau0);
    }

    for (t in 1:nt)
        mu_seeg_log_power[t] = amplitude * (log(gain * exp(x[t]'-x_eq_def)) + offset)';
}

model {
    offset_star ~ normal(0.0, 1.0);
    amplitude_star ~ normal(0.0, 1.0);
    epsilon_star ~ normal(0.0, 1.0);
    sigma_star ~ normal(0.0, 1.0);
    time_scale_star ~ normal(0.0, 1.0);
    // k_star ~ normal(0.0, 1.0);
    to_row_vector(x0_star) ~ normal(0.0, 1.0);
    x_init_star ~ normal(0.0, 1.0);
    z_init_star ~ normal(0.0, 1.0);

    for (t in 1:(nt - 1)) {
        // to_vector(x_eta[t]) ~ normal(0.0, 1.0);
        to_vector(z_eta[t]) ~ normal(0.0, 1.0);
    }

    if (SIMULATE<1)
        for (t in 1:nt)
            seeg_log_power[t] ~ normal(mu_seeg_log_power[t], epsilon);
}


generated quantities {
    row_vector[ns] log_likelihood[nt];
    for (t in 1:nt)
        log_likelihood[t] = ﻿normal_lpdf(seeg_log_power[t] | mu_seeg_log_power[t], epsilon)
}
