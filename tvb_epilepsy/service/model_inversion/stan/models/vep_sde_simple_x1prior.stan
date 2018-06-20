functions {

    matrix vector_differencing(row_vector x1) {
        matrix[num_elements(x1), num_elements(x1)] D;
        for (i in 1:num_elements(x1)) {
            D[i] = x1 - x1[i];
        }
        return D;
    }

    row_vector x1_step(row_vector x1, row_vector z, real Iext1, real tau1) { //, row_vector dX1t, real sigma
        int n_active_regions = num_elements(x1);
        row_vector[n_active_regions] x1_next;
        row_vector[n_active_regions] Iext1_vec = rep_row_vector(Iext1 + 1.0, n_active_regions);
        row_vector[n_active_regions] dx1 = Iext1_vec - (x1 .* x1 .* x1) - 2.0 * (x1 .* x1) - z;
        x1_next = x1 + (tau1 * dx1); # + dX1t * sigma;
        return x1_next;
    }

    row_vector z_step(row_vector x1, row_vector z, row_vector x0, matrix FC, vector Ic, real x1_eq_def, real tau1,
                      row_vector dZt, real sigma, real tau0) {
        int n_active_regions = num_elements(z);
        row_vector[n_active_regions] z_next;
        matrix[n_active_regions, n_active_regions] D = vector_differencing(x1);
        // Ic = Ic_i = sum_{j in nonactive regions} [w_ij]
        // gx(nonactive->active) = Ic * (x1_j - x1_i) = Ic * (x1_eq_def - x1)
        row_vector[n_active_regions] gx = to_row_vector(rows_dot_product(FC, D) + Ic .* to_vector(x1_eq_def - x1));
        row_vector[n_active_regions] dz = inv(tau0) * (4 * (x1 - x0) - z - gx);
        z_next = z + (tau1 * dz) + dZt * sigma;
        return z_next;
    }

    real[] normal_mean_std_to_lognorm_mu_sigma(real mean_, real std_) {
        real mu_sigma[2];
        real logsm21 = std_/mean_;
        logsm21 = log(logsm21 * logsm21 + 1.0);
        mu_sigma[1] = log(mean_) - 0.5 * logsm21;
        mu_sigma[2] = sqrt(logsm21);
        return mu_sigma;
    }

    row_vector normal_mean_std_to_lognorm_mu(row_vector mean_, row_vector std_) {
        int n_active_regions = num_elements(mean_);
        row_vector[n_active_regions] logsm21 = std_ ./ mean_;
        logsm21 = log(logsm21 .* logsm21 + 1.0);
        return log(mean_) - 0.5 * logsm21;
    }

    row_vector normal_mean_std_to_lognorm_sigma(row_vector mean_, row_vector std_) {
        int n_active_regions = num_elements(mean_);
        row_vector[n_active_regions] logsm21 = std_ ./ mean_;
        logsm21 = log(logsm21 .* logsm21 + 1.0);
        return sqrt(logsm21);
    }

    real standard_normal_to_lognormal(real standard_normal, real mu, real sigma){
        return exp(mu + sigma * standard_normal);
    }


    row_vector standard_normal_to_lognormal_row(row_vector standard_normal, row_vector mu, row_vector sigma){
        return exp(mu + sigma .* standard_normal);
    }
}

data {
    int DEBUG;
    int SIMULATE;
    int n_active_regions;
    int n_times; // 1012
    int n_target_data;
    real Iext1; //=3.1
    real tau0; //=10, [3, 30]
    real dt; //~= 0.1 (used 0.0976562)
    real x0_hi; // 2.5
    // real x0_std; // ~0.5
    // real x0_lo;  // 0.0
    row_vector [n_active_regions] x0_star_mu; // x0_hi - x0_mean = 2.5 - ~[-2.5 to 0.0] = ~[2.5 to 5]
    row_vector [n_active_regions] x0_star_std; // minimum((x0_mean - x0_lo)/2.0) ~= 1.0
    // row_vector [n_active_regions] x0_mu;  // healthy: -2.5, sick ~=-2.0, max = [-3.0, -4.0], min = -1.0
    real x1_eq_def; // = -5.0/3 the value of all healhty non-active node
    real x1_min;
    real x1_max;
    int X1_PRIOR;
    row_vector x1_loc;
    row_vector x1_mu;
    row_vector x1_sigma;
    row_vector [n_active_regions] x1_init_mu; // in [-2.0, -1.0], used -1.566
    row_vector [n_active_regions] z_init_mu; // in [2.9, 4.5], used 3.060
    real x1_init_min;
    real x1_init_max;
    real x1_init_std; // 0.0333
    real z_init_std; // 0.0333/2
    real tau1_mu; // 0.5
    real tau1_std; // 0.0667
    real K_mu; // 3.448 = 3 * 100 / n_regions(=87)
    real K_std; // 0.575 = K_mu/6
    real sigma_mu; // =0.01
    real sigma_std; // =0.01/3
    real epsilon_mu; //=0.1
    real epsilon_std; //=0.1/3
    real offset_mu;  //=0.0
    real offset_std; //=1.0
    real scale_mu; //=1.0
    real scale_std; //=1.0/6
    real scale_lo; // =0.3
    int log_target_data; // 1 for log, 0 for linear obervation function
    matrix[n_target_data, n_active_regions] gain;
    row_vector[n_target_data] target_data[n_times];
    vector[n_active_regions] Ic;
    matrix<lower=0.0, upper=1.0>[n_active_regions, n_active_regions] SC;
}

transformed data {
    real sqrtdt = sqrt(dt);
    real scale_mu_sigma[2] = normal_mean_std_to_lognorm_mu_sigma(scale_mu, scale_std);
    real epsilon_mu_sigma[2] = normal_mean_std_to_lognorm_mu_sigma(epsilon_mu, epsilon_std);
    real tau1_mu_sigma[2] = normal_mean_std_to_lognorm_mu_sigma(tau1_mu, tau1_std);
    // real K_mu_sigma[2] = normal_mean_std_to_lognorm_mu_sigma(K_mu, K_std);
    real K = K_mu;
    real sigma_mu_sigma[2] = normal_mean_std_to_lognorm_mu_sigma(sigma_mu, sigma_std);
    row_vector[n_active_regions] x0_logmu = normal_mean_std_to_lognorm_mu(x0_star_mu, x0_star_std);
    row_vector[n_active_regions] x0_sigma = normal_mean_std_to_lognorm_sigma(x0_star_mu, x0_star_std);
    matrix [n_active_regions, n_active_regions] SC_ = SC;
    for (i in 1:n_active_regions) SC_[i, i] = 0;
    SC_ = SC_ / max(SC_) * rows(SC_);

    /*
    print("scale_mu_sigma=", scale_mu_sigma,
           ", scale=", standard_normal_to_lognormal(0.0, scale_mu_sigma[1], scale_mu_sigma[2]));
    print("epsilon_mu_sigma=", epsilon_mu_sigma,
          ", epsilon=", standard_normal_to_lognormal(0.0, epsilon_mu_sigma[1], epsilon_mu_sigma[2]));
    print("tau1_mu_sigma=", tau1_mu_sigma,
          ", tau1=", standard_normal_to_lognormal(0.0, tau1_mu_sigma[1], tau1_mu_sigma[2]));
    print("K_mu_sigma=", K_mu_sigma,
          ", k=", standard_normal_to_lognormal(0.0, K_mu_sigma[1], K_mu_sigma[2]));
    print("sigma_mu_sigma=", sigma_mu_sigma,
          ", sigma=", standard_normal_to_lognormal(0.0, sigma_mu_sigma[1], sigma_mu_sigma[2]));
    print("x0_logmu=", x0_logmu, ", x0_sigma=", x0_sigma);
    */
}

parameters {
    // integrate and predict
    row_vector [n_active_regions] x0_star;
    real epsilon_star;
    real scale_star;
    real offset_star;
    real sigma_star;
    real tau1_star;
    // real K_star;

    // time-series state non-centering:
    row_vector<lower=x1_init_min-x1_init_mu, upper=x1_init_max-x1_init_mu>[n_active_regions] x1_init_star;
    row_vector[n_active_regions] z_init_star;
    // row_vector[n_active_regions] dX1t_star[n_times - 1];
    row_vector[n_active_regions] dZt_star[n_times - 1];

}

transformed parameters {
    real offset = offset_mu + offset_star * offset_std;
    real scale = standard_normal_to_lognormal(scale_star, scale_mu_sigma[1], scale_mu_sigma[2]);
    real epsilon = standard_normal_to_lognormal(epsilon_star, epsilon_mu_sigma[1], epsilon_mu_sigma[2]);
    real sigma = standard_normal_to_lognormal(sigma_star, sigma_mu_sigma[1], sigma_mu_sigma[2]);
    real tau1 = standard_normal_to_lognormal(tau1_star, tau1_mu_sigma[1], tau1_mu_sigma[2]);
    // real K = standard_normal_to_lognormal(K_star, K_mu_sigma[1], K_mu_sigma[2]);
    row_vector[n_active_regions] x0 = x0_hi - standard_normal_to_lognormal_row(x0_star, x0_logmu, x0_sigma);
    row_vector[n_active_regions]<lower=x1_min, upper=x1_max> x1_init = x1_init_mu + x1_init_star * x1_init_std;
    row_vector[n_active_regions] z_init = z_init_mu + z_init_star * z_init_std;
    row_vector<lower=x1_min, upper=x1_max>[n_active_regions] x1[n_times];
    row_vector[n_active_regions] z[n_times];
    row_vector[n_target_data] fit_target_data[n_times];

    x1[1] = x1_init; // - 1.5;
    z[1] = z_init; // 3.0;
    for (t in 1:(n_times-1)) {
        x1[t+1] = x1_step(x1[t], z[t], Iext1, dt*tau1); //, dX1t_star[t], sqrtdt*sigma
        z[t+1] = z_step(x1[t], z[t], x0, K*SC, Ic, x1_eq_def, dt*tau1, dZt_star[t], sqrtdt*sigma, tau0);
    }

    if (log_target_data>0) {
        for (t in 1:n_times)
            fit_target_data[t] = scale * (log(gain * exp(x1[t]'-x1_eq_def)) + offset)';
    } else {
        for (t in 1:n_times)
            fit_target_data[t] = scale * (gain * (x1[t]'-x1_eq_def) + offset)';
    }

    /*
    print("offset=", offset);
    print("scale=", scale);
    print("epsilon=", epsilon);
    print("k=", k);
    print("sigma=", sigma);
    print("tau1=", tau1);
    print("x0=", x0);
    */
}

model {
    offset_star ~ normal(0.0, 1.0);
    scale_star ~ normal(0.0, 1.0);
    epsilon_star ~ normal(0.0, 1.0);
    sigma_star ~ normal(0.0, 1.0);
    tau1_star ~ normal(0.0, 1.0);
    // K_star ~ normal(0.0, 1.0);
    to_row_vector(x0_star) ~ normal(0.0, 1.0);
    x1_init_star ~ normal(0.0, 1.0);
    z_init_star ~ normal(0.0, 1.0);

    for (t in 1:(n_times - 1)) {
        // to_vector(dX1t_star[t]) ~ normal(0.0, 1.0);
        to_vector(dZt_star[t]) ~ normal(0.0, 1.0);
    }

    if (X1_PRIOR > 0) {
         for (t in 1:(n_times - 1))
            to_vector(x1[t] - x1_min - x1_loc) ~ lognormal(x1_mu, x1_sigma);
    }

    if (SIMULATE<1)
        for (t in 1:n_times)
            target_data[t] ~ normal(fit_target_data[t], epsilon);
}


generated quantities {
    row_vector[n_target_data] log_likelihood[n_times];
    for (t in 1:n_times) {
        for (s in 1:n_target_data) {
            log_likelihood[t][s] = normal_lpdf(target_data[t][s] |  fit_target_data[t][s], epsilon);
        }
    }
}
