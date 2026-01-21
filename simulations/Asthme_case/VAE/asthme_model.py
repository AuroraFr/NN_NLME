import jax.numpy as jnp
import jax
import equinox as eqx
import diffrax
import jax.random as jr
from jax_networks import Encoder
from typing import Optional

class asthme_vector_field(eqx.Module):

    def __call__(self, t, y0, params):
        
        p, c, a, m = y0
        p_max = params["p_max"]
        k_ap = params["k_ap"]
        eta_ap = params["eta_ap"]
        k_cp = params["k_cp"]
        gamma_c = params["gamma_c"]
        gamma_p = params["gamma_p"]
        k_pc = params["k_pc"]
        phi_c = params["phi_c"]
        k_s = params["k_s"]
        nu = params["nu"]
        t_is = params["t_i"]
        k_ac = params["k_ac"]
        eta_ac = params["eta_ac"]
        gamma_a = params["gamma_a"]
        k_b = params["k_b"]
        k_pm = params["k_pm"]
        k_apm = params["k_apm"]
        eta_apm = params["eta_apm"]
        phi_m = params["phi_m"]
        k_p = params['k_p']

        stimulus = sum((k_s / nu) * jnp.exp(-((t - t_i) ** 2) / nu ** 2) for t_i in t_is)

        dpdt = k_p * p * (1 - p / p_max) * (1 + (k_ap * a * p) / (eta_ap + a * p)) + k_cp * gamma_c / gamma_p * c - k_pc * p
        dcdt = k_pc * gamma_p / gamma_c * p - (k_cp + phi_c) * c
        dadt = (stimulus + (k_ac * a * c) / (eta_ac + a * c)) * c * m / gamma_a - k_b * (gamma_p * p / gamma_c + c) * a - a
        dmdt = (k_pm + (k_apm * a * p) / (eta_apm + a * p)) * gamma_p * p - phi_m * m

        return jnp.array([dpdt, dcdt, dadt, dmdt])


class Asthme(eqx.Module):

    encoder: Encoder
    vector_field: asthme_vector_field = eqx.static_field()
    Kp: float
    # Ks: float
    Kb: float
    # phiC: float
    Kac: float
    Kp_logstd: float
    # Ks_logstd: float
    # phiC_logstd: float
    logvar_noise: float
    n_z: int = eqx.static_field()
    timepoints:int = eqx.static_field()
    T: int = eqx.static_field()
    latent_shape:int = eqx.static_field()

    def __init__(self, latent_shape, init_params, key, n_z=1, T=100, timepoints=10, batchnorm=False, input_channel=1, 
                 irregular=False, use_mask=False,
                 hidden_dim=4, time_scale=1, conv=True, kernel_size=3, rnn=False, data_aug=False, data_aug_method=None):
        self.encoder = Encoder(latent_shape, timepoints, hidden_dim, key, variance_init_bias=-5, batchnorm=batchnorm, 
                               input_channel=input_channel, irregular=irregular, use_mask=use_mask,
                               conv=conv, kernel_size=kernel_size, rnn=rnn, data_aug=data_aug, data_aug_method=data_aug_method)
        self.vector_field = asthme_vector_field()
        self.Kp = init_params[0]
        self.Kb = init_params[1]
        self.Kac = init_params[2]
        # self.phiC = init_params[3]
        self.Kp_logstd =  init_params[3]
        self.logvar_noise = init_params[-1]
        
        # self.phiC_logstd =  init_params[3]
        # self.Ks =  init_params[1]
        # self.Ks_logstd =  init_params[3]
        
        self.n_z = n_z
        self.timepoints = timepoints
        self.T = T
        self.latent_shape = latent_shape

    def _latent(self, y, state, key):
        key1, key2 = jr.split(key, 2)
        
        post_mean, logstd, state = self.encoder(y, key=key2, state=state)
        std = jnp.exp(logstd)
        standard_mean = jnp.zeros(self.latent_shape)
        standard_cov = jnp.eye(self.latent_shape)
        samples = jr.multivariate_normal(key1, standard_mean, standard_cov, shape=(self.n_z,))
        latent = post_mean + samples * std
        return latent, post_mean, logstd, state
    
    @staticmethod
    def _loss(post_mean, log_post_var, log_prior_var):
        
        klloss = jnp.sum(0.5*(jnp.exp(log_post_var - log_prior_var) + post_mean **2 / (jnp.exp(log_prior_var)) - 1. + log_prior_var - log_post_var))
        return klloss
    
    # Decoder of the VAE
    def _sample(self, latent, times):
        log_kp = self.Kp + latent[:, 0]
        # log_ks = self.Ks + latent[:, 1]
        Kp_values = jnp.exp(log_kp)
        # Ks_values = jnp.exp(log_ks)
        # phiC_values = jnp.exp(log_phiC)

        def solve(param1):
            params = {
                "k_p": param1,
                "p_max": 1.0,
                "k_ap": 1.0,
                "eta_ap": 1.0,
                "k_cp": 0.01,
                "gamma_c": 1.0,
                "gamma_p": 1.0,
                "k_pc": 1.0,
                "phi_c": 0.1,
                "k_s": 0.2,
                "nu": 30.0,
                "t_i": list(range(50, 250, 40)),
                "k_ac": jnp.exp(self.Kac),
                "eta_ac": 1.0,
                "gamma_a": 0.01,
                "k_b": jnp.exp(self.Kb),
                "k_pm": 0.1,
                "k_apm": 0.1,
                "eta_apm": 10.0,
                "phi_m": 0.01
            }

            term = diffrax.ODETerm(self.vector_field)

            solver = diffrax.Kvaerno5()

            t0 = 0
            t1 = self.T

            # times = jnp.linspace(t0, t1, self.timepoints)
            # times = times / self.time_scale

            y0 = jnp.array([0.1, 0.8, 0.01, 0.9])

            dt0 = 0.1
            saveat = diffrax.SaveAt(ts=times)

            # Solve the ODE
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6),
                args=params,
                saveat=saveat,
                # max_steps = 100000
                # adjoint = diffrax.ForwardMode()
            )

            return sol.ys

        X = jax.vmap(solve)(Kp_values)
        epsilon = 1e-8
        Y_trajectories = jnp.log10(jax.nn.relu(X[:, :, 2]) + epsilon)
        print('A(t) trajectories', Y_trajectories.shape)
        
        return Y_trajectories

    def __call__(self, y, times, key_i, state, return_latent=True):

        latent, post_mean, post_logstd, state = self._latent(y, state, key_i)
        X = self._sample(latent, times)
        post_logvar = 2 * post_logstd
        logvars = jnp.array([2*self.Kp_logstd])
        klloss = self._loss(post_mean, post_logvar, logvars)
        
        if return_latent:
            return klloss, X, state, post_mean, post_logstd
        
        return klloss, X, state

    def decoder(self, y, key_i, state):
        samples, _, _, _ = self._latent(y, state, key_i)
        X = self._sample(samples)
        return X, samples


