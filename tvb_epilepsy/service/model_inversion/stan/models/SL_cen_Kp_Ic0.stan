data {
    int nn;
    int nt;
    real I1;
    real tau0;
    real dt;
    matrix[nt, nn] seeg_log_power;
    matrix<lower=0.0>[nn, nn] Con;
    real x0_lo;
    real x0_hi;
    real eps_hi;
    real sig_hi;
    real zlim[2]; 
    real xlim[2]; 
}

transformed data {
}

parameters {
    matrix<lower=xlim[1], upper=xlim[2]>[nn, nt] x;
    matrix<lower=zlim[1], upper=zlim[2]>[nn, nt] z;

    vector[nn] x0;
    
    real<lower=0> K;
    real<lower=0.0> amp;
    real<lower=0.0> offset;
    
    real<lower=0.0> eps;
    real<lower=0.0> sig;
    real<lower=0.0,  upper=1.> tt;
    


}

transformed parameters {
}

model {
    real dx;
    real dz;
    real gx;

                    
    x0 ~ normal(-3.0, .75);                
    amp ~ normal(1,10);
    offset ~ normal(0, 2);
                                      
    eps ~ normal(0, 1);
    sig ~ normal(0, 1);
    tt ~ normal(.5, .1); 
      

    K ~ normal(10, 1.); 
    

    /* integrate & predict */
    
    for (i in 1:nn) {
     x[i, 1] ~ normal(-1.5, 1);
     z[i, 1] ~ normal(+2.0, 1);
   } 
    
    for (t in 1:(nt-1)) {
        for (i in 1:nn) {
             gx = 0;
             for (j in 1:nn)
                 if (i!=j)
                    gx = gx + Con[i, j]*(x[j, t] - x[i, t]);
            dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
            dz = (1/tau0)*(4*(x[i, t] - x0[i]) - z[i, t] - K*gx);
            x[i, t+1] ~ normal(x[i, t] + dt*tt*dx, sig); 
            z[i, t+1] ~ normal(z[i, t] + dt*tt*dz, sig); 
                         }
                     }
       
    
    to_vector(seeg_log_power) ~ normal(amp * (to_vector(x') + offset), eps);

}


