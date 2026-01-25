import jax.numpy as jnp
import jax
import equinox as eqx
import diffrax
import jax.random as jr
from networks import Encoder
from typing import Optional

class Antibody_ode(eqx.Module):

    def __call__(self, time, y0, args):
        death_rate_S_cell, vaccine_antigen_decline_rate, Ab_production_init_acc, F2, F3, Ab_degrad_rate = args

        S, Ab = y0
        fold_change = 1.0  # Assuming fold_change is constant here
        second_dose = 30
        third_dose = 250
        
        values = jnp.select(
                 [time < second_dose, (time >= second_dose) & (time < third_dose), time >= third_dose],
                 [jnp.array([fold_change, time]),  jnp.array([F2, time-second_dose]), jnp.array([F3, time-third_dose])],
                default=jnp.array([1.0 ,time]))

        dSdt =  1 * (values[0] * jnp.exp(-vaccine_antigen_decline_rate *  values[1]) - death_rate_S_cell * S)
        dAbdt = 1 * (Ab_production_init_acc * S - Ab_degrad_rate * Ab)

        return jnp.array([dSdt, dAbdt])

class Antibody(eqx.Module):

    encoder: Encoder
    vector_field: Antibody_ode = eqx.static_field()
    logstd_theta:float
    logstd_F2:float
    log_noise_var:float
    theta: float
    F2:float
    F3:float
    has_deltaS : bool
    deltaS: Optional[float]
    deltaAB: Optional[float]
    n_z: int = eqx.static_field()
    time_scale:int = eqx.static_field()
    timepoints:int = eqx.static_field()
    T: int = eqx.static_field()
    latent_shape:int = eqx.static_field()

    def __init__(self, latent_shape, init_params, key, n_z=1, T=100, timepoints=10, kernel_size=3,
                 hidden_dim=4, input_channel=1, out_channel=1, time_scale=1, conv=True, rnn=False, data_aug=False, data_aug_method=None, irregular=False):
        
        self.encoder = Encoder(latent_shape, timepoints, hidden_dim,  key, conv=conv, input_channel=input_channel,
                               rnn=rnn, data_aug=data_aug, data_aug_method=data_aug_method, irregular=irregular, out_channel=out_channel, kernel_size=kernel_size)
        self.vector_field = Antibody_ode()
        
        self.log_noise_var = init_params[-1]
        self.theta = init_params[0]
        self.F2 = init_params[1]
        self.F3 = init_params[2]
        self.logstd_theta = init_params[3]
        self.logstd_F2 = init_params[4]
        self.has_deltaS = len(init_params) >= 7
        self.deltaS = jax.lax.cond(self.has_deltaS, lambda _: init_params[5], lambda _: jnp.nan, None)
        self.deltaAB = jax.lax.cond(len(init_params) == 8, lambda _: init_params[6], lambda _: jnp.nan, None)
        self.time_scale = time_scale
        self.n_z = n_z
        self.timepoints = timepoints
        self.T = T
        self.latent_shape = latent_shape

    def _latent(self, y, state, key, iaf=False):
        key1, key2 = jr.split(key, 2)
        
        post_mean, logstd, state = self.encoder(y, key=key2, state=state)
        std = jnp.exp(logstd)
        standard_mean = jnp.zeros(self.latent_shape)
        standard_cov = jnp.eye(self.latent_shape)  # Identity covariance ensures independence
        samples = jr.multivariate_normal(key1, standard_mean, standard_cov, shape=(self.n_z,))
        latent = post_mean + samples * std
        if iaf:
            latent, log_det = self.iaf(latent)
            return latent, log_det, post_mean, logstd, state
        return latent, post_mean, logstd, state
    
    @staticmethod
    def _loss(post_mean, log_post_var, log_prior_var, iaf=False, log_det=0):
        # KL loss
        # we assum mu_prior is 0 for all the random effect
        klloss = jnp.sum(0.5*(jnp.exp(log_post_var - log_prior_var) + post_mean **2 / (jnp.exp(log_prior_var)) - 1. + log_prior_var - log_post_var))
        
        return klloss
    
    # Decoder of the VAE
    def _sample(self, latent, times):
        
        latent = jnp.exp(latent).reshape(-1,self.latent_shape)
        theta_values = jnp.exp(self.theta) * latent[:, 0].reshape(-1, 1)
        F2_values = jnp.exp(self.F2) * latent[:, 1].reshape(-1, 1)
        
        def solve(individual_theta, individual_F2):
            # Parameters
            # Ab_degrad_rate = jax.lax.cond(self.has_deltaS, lambda _: jnp.exp(self.deltaAB) + jnp.exp(self.deltaS), lambda _: 0.08, None)  # delta_Ab
            # Ab_degrad_rate = 0.08
            deltaAB_is_present = jnp.isfinite(self.deltaAB)
            Ab_degrad_rate = jnp.where(deltaAB_is_present, jnp.exp(self.deltaAB) + jnp.exp(self.deltaS), 0.08)
            vaccine_autigen_decline_rate = 2.7  # delta_v
            death_rate_S_cell = jax.lax.cond(self.has_deltaS, lambda _: jnp.exp(self.deltaS), lambda _: 0.01, None) # delta_s

            term = diffrax.ODETerm(self.vector_field)

            solver = diffrax.Tsit5()
            t0 = 0
            t1 = self.T

            # times = jnp.linspace(t0, t1, self.timepoints)
            # times = times / self.time_scale

            y0 = jnp.array([0.01, 0.1])

            dt0 = 0.01
            saveat = diffrax.SaveAt(ts=times)

            # Solve the ODE
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8),
                args=(death_rate_S_cell, vaccine_autigen_decline_rate, individual_theta, individual_F2, jnp.exp(self.F3), Ab_degrad_rate),
                saveat=saveat,
                max_steps = 500000
                # adjoint = diffrax.ForwardMode()
            )

            return sol.ys

        X = jax.vmap(solve)(theta_values.squeeze(1), F2_values.squeeze(1))
        Ab_trajectories = jnp.log10(X[:, :, 1])

        return Ab_trajectories

    def __call__(self, y, ts, key_i, state, ):

        latent, post_mean, post_logstd, state = self._latent(y, state, key_i)
        X = self._sample(latent, ts)
        post_logvar = 2*post_logstd
        logvars = jnp.array([2*self.logstd_theta, 2*self.logstd_F2])
        klloss = self._loss(post_mean, post_logvar, logvars) 

        return klloss, X, state, latent, post_mean, post_logstd

    def decoder(self, y, times, key_i, state):
        samples, _, _, _ = self._latent(y, state, key_i)
        X = self._sample(samples, times)
        return X, samples


