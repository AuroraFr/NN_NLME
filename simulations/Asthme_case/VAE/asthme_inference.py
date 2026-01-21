import os
os.environ['JAX_PLATFORMS']='cpu'
from asthme_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from os import listdir
from os.path import isfile, join

jax.config.update("jax_enable_x64", True)
latent_shape = 1
n_z = 1

seed = 42
key = jr.PRNGKey(seed)

param1 = jnp.log(30.0)
param2 = jnp.log(0.8)

init_params = jnp.array([param1, param1, param2, param2])

model, state = eqx.nn.make_with_state(Asthme)(latent_shape, init_params, key, n_z=n_z, T=400, batchnorm=False,
                                                  timepoints=20, time_scale=1, conv=True, kernel_size=7, rnn=True, input_channel=1, hidden_dim=4)

from os import listdir
from os.path import isfile, join

file_path = 'EXPs/exp_asthme_1latent_noise_20_convergence_kp_kb_kac/'

onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]

parameters=[]

for filename in sorted(onlyfiles):    
    try:
        model_loaded = eqx.tree_deserialise_leaves(file_path+filename, model)
        parameters.append([jnp.exp(model_loaded.Kp), jnp.exp(model_loaded.Kb), jnp.exp(model_loaded.Kac), jnp.exp(model_loaded.Kp_logstd)])
        
        print(filename, jnp.exp(model_loaded.Kp), jnp.exp(model_loaded.Kb), jnp.exp(model_loaded.Kac), 
              jnp.exp(model_loaded.Kp_logstd), jnp.sqrt(jnp.exp(model_loaded.logvar_noise)))
    except Exception as error:
        print(error)

# np.savetxt('EXPs/exp_asthme_1latent_convergence_3/Asthme_convergence_3.txt',parameters, fmt='%.4f')