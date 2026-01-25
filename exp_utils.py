import jax.numpy as jnp
def gaussian_log_prob(model, X, targets, const=2.0 * jnp.pi):
    
    log_p_y_given_b = -0.5 * jnp.log(const) - 0.5 * model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
    
    return log_p_y_given_b