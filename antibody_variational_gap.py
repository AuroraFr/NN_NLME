import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import equinox as eqx
from antibody_multidoses_model import Antibody
from os import listdir
from os.path import isfile, join
import jax.random as jr

def log_normal_density(x, mean, logvar):
    """Log-density of a diagonal Gaussian."""
    return -0.5 * (jnp.log(2 * jnp.pi) + logvar + (x - mean)**2 / jnp.exp(logvar))

def importance_weighted_log_likelihood(key, Y_i, model, state):
    """
    Monte Carlo importance-sampling estimate of log p_Phi(Y)
    using q_psi(b|Y) as proposal.

    Args:
        key: PRNG key
        Y: (T, D) observed trajectory for one subject
        model:
        state:

    Returns:
        log_p_hat: estimated log marginal likelihood (log p_Phi(Y))
        elbo: estimated ELBO
        gap: log_p_hat - elbo (variational gap)
    """

    key1, key2 = jr.split(key, 2)
    
    klloss, X, state, b_samples, post_mean, post_logstd = model(Y_i, key2, state)
    X = X.reshape(model.n_z, -1)

    Y_i = jnp.repeat(Y_i, repeats=model.n_z, axis=0).reshape(model.n_z, -1)
    const = 2. * jnp.pi
    log_noise_var = jnp.log(0.1 ** 2)
    log_p_Y_given_b = (-0.5*jnp.log(const) - 0.5*log_noise_var - 0.5 * jnp.exp(-log_noise_var) * (X - Y_i)**2).mean(axis=-1)
    
    # Compute log densities
    log_q = log_normal_density(b_samples, post_mean, 2*post_logstd).sum(axis=-1)
    logvar = jnp.array([2*model.logstd_theta, 2*model.logstd_F2])
    log_p_b = log_normal_density(b_samples, jnp.array([0.0, 0.0]), logvar).sum(axis=-1)  # prior

    # Importance weights
    log_w = log_p_Y_given_b + log_p_b - log_q
    log_p_hat = jax.scipy.special.logsumexp(log_w) - jnp.log(log_w.shape[0])
    
    # Effective sample size (ESS)
    log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
    log_ess = -jax.scipy.special.logsumexp(2.0 * log_w_norm)
    ess = jnp.exp(log_ess)             # in [1, K]
    ess_frac = ess / log_w.shape[0]    # should be, say, > 0.01 ideally
    print(ess_frac)

    # ELBO estimate
    reconstruction_loss = jnp.mean(log_p_Y_given_b)
    elbo = reconstruction_loss - klloss
    elbo_from_terms = jnp.mean(log_p_Y_given_b + log_p_b - log_q)
    
    gap = log_p_hat - elbo
    return log_p_hat, elbo, gap


jax.config.update("jax_enable_x64", True)
latent_shape = 2
n_z = 10000

init_theta = jnp.log(30.0)  # theta
init_theta_logstd = jnp.log(0.8) # random effect std on theta
init_F2 = jnp.log(8.0)
init_F3 = jnp.log(18.5)
init_F2_logstd = jnp.log(1)
init_deltaS= jnp.log(0.01)
init_deltaAb= jnp.log(0.01)
init_noise_var = jnp.log(0.1 ** 2)

init_params = jnp.array([init_theta, init_F2, init_F3, init_theta_logstd, init_F2_logstd, init_deltaS, init_deltaAb])
data = np.load('antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_0.npy')[:,:,1]


file_path = 'EXPs/exp_antibody_3dose_2latent/cnn_enc_15p_400d_50s_deltaAB_2latent_theta_F2/'
onlyfiles = sorted([f for f in listdir(file_path) if isfile(join(file_path, f))])
gaps = []
i = 0
for filename in onlyfiles:
    key = random.PRNGKey(5)
    key1, key2 =  jr.split(key, 2)
    model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, key1, n_z=n_z, timepoints=15, hidden_dim=4, rnn=False, conv=True)
    if i > 0:
        break
    model = eqx.tree_deserialise_leaves(file_path+filename, model)
    print(model.theta, jnp.exp(model.F2), jnp.exp(model.F3), jnp.exp(model.logstd_theta), jnp.exp(model.logstd_F2), model.deltaS, jnp.exp(model.deltaAB))

    keys = jr.split(key2, data.shape[0])
    log_p_hat, elbo, gap = jax.vmap(importance_weighted_log_likelihood, in_axes=(0, 0, None, None))(
        keys, data, model, state
    )

    print(f"Log marginal likelihood ≈ {jnp.mean(log_p_hat):.3f}")
    print(f"ELBO ≈ {jnp.mean(elbo):.3f}")
    print(f"Variational gap ≈ {jnp.mean(gap):.3f}")
    i += 1