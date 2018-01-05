functions {

    matrix vector_differencing(row_vector x) {
        matrix[num_elements(x), num_elements(x)] D;
        for (i in 1:num_elements(x)) {
            D[i] = x - x[i];
        }
        return D';
    }

    row_vector x_step(row_vector x, row_vector z, real I1, real time_scale, real sigma) {
        int nn = num_elements(x);
        row_vector[nn] x_next;
        row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
        row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
        x_next = x + (time_scale * dx);
        return x_next;
    }

    row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, vector Ic,
            real time_scale, row_vector z_eta, real sigma, real tau0) {
        int nn = num_elements(z);
        row_vector[nn] z_next;
        matrix[nn, nn] D = vector_differencing(x);
        row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D) - Ic .* to_vector(1.8 + x));
        row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
        z_next = z + (time_scale * dz) + z_eta * sigma;
        return z_next;
    }
}

data {
    int nn;
    int nt;
    int ns;
    real I1;
    real tau0;
    real dt;
    matrix[ns, nn] gain;
    row_vector[ns] seeg_log_power[nt];
    vector[nn] Ic;
    matrix<lower=0.0, upper=1.0>[nn, nn] SC;
    int predict_data;
}

transformed data {
    matrix[ns, nn] log_gain = log(gain);
    matrix [nn, nn] SC_ = SC;
    for (i in 1:nn) SC_[i, i] = 0;
    SC_ /= max(SC_) * rows(SC_);
}

parameters {
    // integrate and predict
    row_vector[nn] x0;
    real epsilon_star;
    real<lower=0.0> amplitude;
    real offset;
    real time_scale_star;

    // time-series state non-centering:
    row_vector[nn] x_init;
    row_vector[nn] z_init;
    row_vector[nn] z_eta[nt - 1];
    real<lower=0.0> sigma_star;
    real k_star;
}

transformed parameters {
    real epsilon = 0.05 * exp(0.1 * epsilon_star);
    real sigma = 0.053 * exp(0.1 * sigma_star);
    real time_scale = 0.15 * exp(0.4 * time_scale_star - 1.0);
    real k = 1e-3 * exp(0.5 * k_star);
    row_vector[nn] x[nt];
    row_vector[nn] z[nt];
    row_vector[ns] mu_seeg_log_power[nt];

    x[1] = x_init - 1.5;
    z[1] = z_init + 2.0;
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, time_scale, sigma);
        z[t+1] = z_step(x[t], z[t], x0, k*SC, Ic, time_scale, z_eta[t], sigma, tau0);
    }
    for (t in 1:nt)
        mu_seeg_log_power[t] = amplitude * (log(gain * exp(x[t]')) + offset)';
}

model {
    x0 ~ normal(-3.0, 1.0);
    x_init ~ normal(0.0, 3);
    z_init ~ normal(0.0, 3);
    sigma_star ~ normal(0, 1.0);
    time_scale_star ~ normal(0, 1.0);

    amplitude ~ normal(1, 0.5);
    offset ~ normal(0.0, 0.5);
    epsilon_star ~ normal(0, 1.0);

    for (t in 1:(nt - 1))
        to_vector(z_eta[t]) ~ normal(0, 1);

    k_star ~ normal(0, 1);

    for (t in 1:predict_data)
        seeg_log_power[t] ~ normal(mu_seeg_log_power[t], epsilon);
}

generated quantities {
    row_vector[ns] gq_seeg_log_power[nt];
    for (t in 1:nt)
        for (i in 1:ns)
            gq_seeg_log_power[t][i] = normal_rng(mu_seeg_log_power[t][i], epsilon);
}