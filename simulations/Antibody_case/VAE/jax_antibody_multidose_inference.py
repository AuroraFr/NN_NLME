import os
os.environ['JAX_PLATFORMS']='cpu'
from jax_antibody_multidoses_joint_model import *
import jax.numpy as jnp
import equinox as eqx

from os import listdir
from os.path import isfile, join
import numpy as np

jax.config.update("jax_enable_x64", True)
latent_shape = 2
n_z = 1

init_theta = jnp.log(30.0)
init_logstd_theta = jnp.log(0.8**2)
init_delta_S = 0.05
init_noise_var = jnp.log(0.1 ** 2)
init_F2 = jnp.log(8.0)
init_F3 = jnp.log(18.5)
init_logstd_F2 = jnp.log(1)

seed = 42
key = jr.PRNGKey(seed)
irregular = True

init_params = jnp.array([init_theta, init_F2, init_F3, init_logstd_theta, init_logstd_F2, 0.01, 0.08, 0.2**2])

model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, key, n_z=n_z, irregular=True, 
                                                timepoints=10, hidden_dim=32, rnn=True, conv=False, out_channel=1, input_channel=3)

from os import listdir
from os.path import isfile, join

#irregular_10p_400d_50s_deltaAB_noise_2latent_theta_F2
#irregular_identifiability_10p_400d_50s_deltaAB_noise_2latent_theta_F2
if irregular:
    file_path = 'EXPs/exp_antibody_3dose_2latent/irregular_identifiability_10p_400d_50s_deltaS_noise_2latent_theta_F2/'
else:
    file_path = 'EXPs/exp_antibody_3dose_2latent/10p_400d_50s_2latent_theta_F2/'
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]

parameters=[]

for filename in sorted(onlyfiles):
    print(filename)
    try:
        model_loaded = eqx.tree_deserialise_leaves(file_path+filename, model)
        print(model_loaded.theta, model_loaded.F2, model_loaded.F3, 
              jnp.exp(model_loaded.logstd_theta), jnp.exp(model_loaded.logstd_F2), jnp.sqrt(jnp.exp(model_loaded.log_noise_var)),
              model_loaded.deltaS, model_loaded.deltaAB)
        parameters.append([model_loaded.theta,model_loaded.F2, model_loaded.F3, model_loaded.deltaS,  
                           jnp.exp(model_loaded.logstd_theta), jnp.exp(model_loaded.logstd_F2), jnp.sqrt(jnp.exp(model_loaded.log_noise_var))])
    except Exception as error:
        print(error)
np.savetxt('results/antibody_3doses_400d_10p_50s_deltaS_irregular_identifiability_2latents.txt', np.array(parameters), fmt='%.4f')
