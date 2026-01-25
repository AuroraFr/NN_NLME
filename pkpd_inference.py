import os
os.environ['JAX_PLATFORMS']='cpu'
from jax_pkpd_model import PKPD
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import numpy as np

#Training
latent_shape = 1
init_conditions = [2.0, 3.0]
n_z = 1
input_dim=5
hidden_dim=16
seed = 42
key = jr.PRNGKey(seed)

init_theta1 = jnp.log(0.3)
init_theta2 = jnp.log(1)
init_prior_logstd = jnp.log(0.8)
init_noise_logvar = jnp.log(0.3 ** 2)
init_params = jnp.array([init_theta1, init_theta2, init_prior_logstd, init_noise_logvar])

model, state = eqx.nn.make_with_state(PKPD)(latent_shape, init_params, input_dim,  hidden_dim, key, n_z=n_z, conv=False, rnn=True,
                                            input_channel=3)

parameters = []

for i in range(1,100):
    try:
        model_loaded = eqx.tree_deserialise_leaves("EXPs/PKPD_exp/irregular_5p_100_10d_noise/PKPD_"+str(i)+".eqx", model)
        print(model_loaded.theta1, model_loaded.theta2, model_loaded.theta1_logstd, model_loaded.log_noise_var)
        parameters.append([model_loaded.theta1, model_loaded.theta2, model_loaded.theta1_logstd, model_loaded.log_noise_var])
    except:
        continue

np.savetxt('results/PKPD_irregular_5p_100_10d_noise.txt', parameters)





