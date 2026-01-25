import jax.numpy as jnp
import jax
import equinox as eqx
import diffrax
import jax.random as jr
from jax_networks import Encoder
from typing import Optional

class Antibody_ode(eqx.Module):

    def __call__(self, time, y0, args):
        death_rate_S_cell, vaccine_autigen_decline_rate, Ab_production_init_acc, F2, F3, Ab_degrad_rate, injection_times = args

        S, Ab = y0
        fold_change = 1.0  # Assuming fold_change is constant here
        second_dose = injection_times[0]
        third_dose = injection_times[1]
        
        values = jnp.select(
                 [time < second_dose, (time >= second_dose) & (time < third_dose), time >= third_dose],
                 [jnp.array([fold_change, time]),  jnp.array([F2, time-second_dose]), jnp.array([F3, time-third_dose])],
                default=jnp.array([1.0 ,time]))

        dSdt =  1 * (values[0] * jnp.exp(-vaccine_autigen_decline_rate *  values[1]) - death_rate_S_cell * S)
        dAbdt = 1 * (Ab_production_init_acc * S - Ab_degrad_rate * Ab)

        return jnp.array([dSdt, dAbdt])

class Antibody(eqx.Module):
    encoder: Encoder
    vector_field: Antibody_ode = eqx.static_field()
    logstd_theta:float
    logstd_F2:float
    theta: float
    F2:float
    F3:float
    logvar_noise:float
    has_deltaS : bool
    deltaS: Optional[float]
    deltaAB: Optional[float]
    n_z: int = eqx.static_field()
    time_scale:int = eqx.static_field()
    timepoints:int = eqx.static_field()
    T: int = eqx.static_field()
    latent_shape:int = eqx.static_field()

    def __init__(self, latent_shape, init_params, key, T=None, n_z=1, conv=False, use_mask=False, timepoints=17, input_dim=17, hidden_dim=32, out_channel=64, time_scale=1):

        self.encoder = Encoder(latent_shape, input_dim, hidden_dim, key, conv=conv, dropout=False, variance_init_bias=-10, 
                               input_channel=1, out_channel=out_channel, kernel_size=1, padding=0, use_mask=use_mask)
        self.vector_field = Antibody_ode()
        
        self.theta = init_params[0]
        self.F2 = init_params[1]
        self.F3 = init_params[2]
        self.logstd_theta = init_params[3]
        self.logstd_F2 = init_params[4]
        self.logvar_noise = init_params[5]
        self.has_deltaS = len(init_params) >= 7
        self.deltaS = jax.lax.cond(self.has_deltaS, lambda _: init_params[6], lambda _: 0.0, None)
        self.deltaAB = jax.lax.cond(len(init_params) == 8, lambda _: init_params[7], lambda _: 0.0, None)
        self.time_scale = time_scale
        self.n_z = n_z
        self.timepoints = timepoints #max sequence length
        self.latent_shape = latent_shape
        self.T = T

    def get_logstd_prior(self):
        return jnp.stack([self.logstd_theta, self.logstd_F2]) 

    def _latent(self, y, mask, key):
        key1, key2 = jr.split(key, 2)
        
        post_mean, logstd, _ = self.encoder(y, mask=mask, key=key1)
        std = jnp.exp(logstd)
        standard_mean = jnp.zeros(self.latent_shape)
        standard_cov = jnp.eye(self.latent_shape)  # Identity covariance ensures independence
        samples = jr.multivariate_normal(key2, standard_mean, standard_cov, shape=(self.n_z,))
        latent = post_mean + samples * std

        return latent, post_mean, logstd
    
    @staticmethod
    def _KLloss(post_mean, log_post_var, log_prior_var):
        # KL loss
        # we assum mu_prior is 0 for all the random effect
        klloss_per_dim = 0.5*(jnp.exp(log_post_var - log_prior_var) + post_mean **2 / (jnp.exp(log_prior_var)) - 1. + log_prior_var - log_post_var)
        
        return klloss_per_dim
    
    # Decoder of the VAE
    def _sample(self, latent, injection_times, ts, mask):
        
        latent = jnp.exp(latent).reshape(-1,self.latent_shape)
        theta_values = jnp.exp(self.theta) * latent[:, 0].reshape(-1, 1)
        F2_values = jnp.exp(self.F2) * latent[:, 1].reshape(-1, 1)
        
        def solve(individual_theta, individual_F2, injection_times, ts, mask):

            # Parameters
            # death_rate_S_cell = 0.01
            Ab_degrad_rate = jax.lax.cond(self.has_deltaS, lambda _: jnp.exp(self.deltaAB) + jnp.exp(self.deltaS), lambda _: 0.08, None)  # delta_Ab
            vaccine_autigen_decline_rate = 2.7  # delta_v
            death_rate_S_cell = jax.lax.cond(self.has_deltaS, lambda _: jnp.exp(self.deltaS), lambda _: 0.01, None) # delta_s
            
            term = diffrax.ODETerm(self.vector_field)
            solver = diffrax.Tsit5()

            # === 1. UNPAD a single subject's data using the mask ===
            # Calculate the true number of time points for this subject
            true_length = jnp.sum(mask).astype(jnp.int32)
            
            # Get the real start and end times
            t0 = 0.0
            t1 = ts[true_length - 1]

            y0 = jnp.array([0.01, 0.1])

            dt0 = 0.01
            saveat = diffrax.SaveAt(ts=ts)

            # The full set of arguments for the vector field
            ode_args = (death_rate_S_cell, vaccine_autigen_decline_rate, individual_theta, 
                    individual_F2, jnp.exp(self.F3), Ab_degrad_rate, injection_times)

            # Solve the ODE
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8),
                args=ode_args,
                saveat=saveat,
                max_steps = 100000
                # adjoint = diffrax.ForwardMode()
            )

            processed_result = jnp.log10(sol.ys[:, 1:2]) # Or any other processing
            # Use the mask to zero-out the results for the padded time steps
            final_output = processed_result * mask[:, None] # Broadcast mask

            return final_output

        Ab_trajectories = jax.vmap(solve, in_axes=(0, 0, None, None, None))(theta_values.squeeze(1), F2_values.squeeze(1), injection_times, ts, mask)
      
        return Ab_trajectories
    

    def __call__(self, input_data, mask, key_i, ts, injection_times):

        latent, post_mean, post_logstd = self._latent(input_data, mask, key_i)
        X = self._sample(latent, injection_times, ts, mask)
        post_logvar = jnp.log(jnp.exp(post_logstd)**2)
        logvars = jnp.array([jnp.log(jnp.exp(self.logstd_theta)**2),jnp.log(jnp.exp(self.logstd_F2)**2)])
        klloss = self._KLloss(post_mean, post_logvar, logvars)
        return klloss, X, post_mean, post_logstd

    def inference(self, input_data, mask, key_i, ts, injection_times):
        
        samples, _, _, = self._latent(input_data, mask, key_i)
        X = self._sample(samples, injection_times, ts, mask)
        return X, samples
