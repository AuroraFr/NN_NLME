import pandas as pd
import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, Float
import diffrax
from jax import jit
import time
import jax.random as jr
import numpy as np
from jax_networks import *

class PKPD_ode(eqx.Module):
    # @jit
    def __call__(self, t, x, args): 
        phi1, phi2, time_scale = args
        param = jnp.multiply(jnp.array([[-phi1, phi2], [0, -phi2]]), time_scale)
        return param @ x

class PKPD(eqx.Module):
    encoder: Encoder
    vector_field: PKPD_ode
    theta1_logstd:float
    log_noise_var:float
    theta1: float
    theta2: float
    n_z: int
    mu_prior:int
    time_scale:int

    def __init__(self, latent_shape, init_params, input_dim, hidden_dim, key, conv=True, rnn=False, n_z=1, 
                 mu_prior=0, time_scale=1, irregular=False, input_channel=1):
        self.encoder = Encoder(latent_shape, input_dim, hidden_dim, key, conv=conv, rnn=rnn, irregular=irregular, input_channel=input_channel)
        self.vector_field = PKPD_ode()
        
        self.theta1 = init_params[0]
        self.theta2 = init_params[1]
        self.theta1_logstd = init_params[2]
        self.log_noise_var = init_params[-1]
        
        self.time_scale = time_scale
        self.n_z = n_z
        self.mu_prior = mu_prior

    def _latent(self, y, state, key):
        mean, logstd, state = self.encoder(y, state)
        std = jnp.exp(logstd)
        print(mean.shape, std.shape, jr.normal(key, (self.n_z,)).shape)
        latent = mean + jr.normal(key, (self.n_z,)) * std
        return latent, mean, logstd, state
    
    @staticmethod
    def _loss(post_mu, mu_prior, log_post_var, log_prior_var):        
        # KL loss
        klloss = 0.5*(jnp.exp(log_post_var - log_prior_var) + (post_mu - mu_prior)**2 / (jnp.exp(log_prior_var)) - 1. + log_prior_var - log_post_var)
        return klloss
    
    # Decoder of the VAE
    def _sample(self, latent, times):
        y0 = jnp.repeat(jnp.array([[2.0, 3.0]]), repeats=self.n_z, axis=0)
        phi1 = jnp.exp(self.theta1) *jnp.exp(latent).reshape(-1,1)
        
        t0 = 0
        t1 = 11
        # ts=jnp.linspace(t0, t1, 11)

        def solve(phi1, y0):
            dt0 = 0.01
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.vector_field),
                diffrax.Dopri5(scan_kind="bounded"),
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8),
                args=(phi1, jnp.exp(self.theta2), 1),
                saveat=diffrax.SaveAt(ts=times),
                max_steps=10000
            )
            return sol.ys

        start_time = time.time()

        X = jax.vmap(solve)(phi1.squeeze(), y0)
        print("solver time", time.time() - start_time, X.shape)
    
        return X[:,:,0]

    def __call__(self, y, ts, key_i, state):
        latent, post_mu, post_logstd, state = self._latent(y, state, key_i)
        X = self._sample(latent, ts)
        post_logvar = 2*post_logstd
        klloss = self._loss(post_mu, self.mu_prior, post_logvar, 2*self.theta1_logstd)
        return klloss, X, state



