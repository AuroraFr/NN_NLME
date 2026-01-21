import os
os.environ['JAX_PLATFORMS']='cpu'
from jax_antibody_multidoses_joint_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from os import listdir
from os.path import isfile, join
import pandas as pd
jax.config.update("jax_enable_x64", True)
latent_shape = 2
n_z = 1

init_Ab_production_init_acc = jnp.log(30.0)
init_Ab_production_init_acc_logvar = jnp.log(0.8**2)
init_delta_S = 0.05
init_noise_var = jnp.log(0.1 ** 2)
init_F2 = jnp.log(8.0)
init_F3 = jnp.log(18.5)
init_fold_change_2nd_dose_logvar = jnp.log(1)

seed = 42
key = jr.PRNGKey(seed)
irregular = False

init_params = jnp.array([init_Ab_production_init_acc, init_F2, init_F3, init_Ab_production_init_acc_logvar, init_fold_change_2nd_dose_logvar, jnp.log(0.2**2)])

model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, key, n_z=n_z, irregular=irregular, timepoints=15, hidden_dim=4, rnn=False, conv=True, out_channel=1, input_channel=1)

from os import listdir
from os.path import isfile, join

#15p_400d_50s_2latent_deltaAB_theta_F2_noise
#10p_400d_50s_2latent_theta_F2_noise
#10p_400d_50s_2latent_theta_F2_deltaAB_noise
if irregular:
    file_path = 'EXPs/exp_antibody_3dose_2latent/irregular_10p_400d_50s_2latent_theta_F2/'
else:
    file_path = 'EXPs/exp_antibody_3dose_2latent/15p_400d_50s_2latent_theta_F2_deltaAB_noise/'
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]

parameters=[]

for filename in sorted(onlyfiles):
    print(filename)
    try:
        model_loaded = eqx.tree_deserialise_leaves(file_path+filename, model)
        print(model_loaded.theta,model_loaded.F2, model_loaded.F3, jnp.exp(model_loaded.logstd_theta), jnp.exp(model_loaded.logstd_F2), model_loaded.deltaAB, model_loaded.deltaS, jnp.sqrt(jnp.exp(model_loaded.log_noise_var)))
        parameters.append([model_loaded.theta,model_loaded.F2, model_loaded.F3, model_loaded.deltaAB, model_loaded.deltaS,jnp.exp(model_loaded.logstd_theta), jnp.exp(model_loaded.logstd_F2)])
    except Exception as error:
        print(error)
# np.savetxt('results/antibody_3doses_400d_15p_50s_deltaAb_identi_2latents.txt', np.array(parameters), fmt='%.4f')
