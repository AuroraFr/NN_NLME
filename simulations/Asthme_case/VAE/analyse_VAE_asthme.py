import os
os.environ['JAX_PLATFORMS']='cpu'
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import numpy as np
from asthme_model import Asthme
import jax.random as jr
latent_shape = 1
n_z = 1
seed = 42
key = jr.PRNGKey(seed)

param1 = jnp.log(30.0)  # theta
param2 = jnp.log(0.8) # random effect std on theta

parameters =[]

from os import listdir
from os.path import isfile, join

file_path = 'EXPs/exp_asthme_1latent_quality_kp/'
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]

init_params = jnp.array([param1, param2])

for i in range(1, 100):
    filename = 'Asthme_1latent_Kp_'+str(i)+'.eqx'

    model, state = eqx.nn.make_with_state(Asthme)(latent_shape, init_params, key, n_z=n_z, T=400, batchnorm=False,
                                                  timepoints=20, time_scale=1, conv=True, kernel_size=7, rnn=False)
    try:
        model_loaded = eqx.tree_deserialise_leaves(file_path+"/"+filename, model)
        # print(i, model_loaded.Kp, jnp.exp(model_loaded.Kb), jnp.exp(model_loaded.Kp_logstd))
        parameters.append([jnp.exp(model_loaded.Kp), jnp.exp(model_loaded.Kp_logstd)])
    except Exception as error:
        # print(error)
        continue

parameters = np.array(parameters)
# np.savetxt('results/asthme_1latent_convergence_kp.txt', parameters)

df = pd.DataFrame(columns=['parameter', 'Mean', 'RRMSE', 'Bias','Variance'])

true_std_Kp = 0.05
true_pop_Kp = np.log(1.15)
true_kb = 1
true_kac = np.log(0.01)

predicted_pop_Kp= np.log(parameters[:, 0])
# predicted_kb = parameters[:, 1]
# predicted_kac = np.log(parameters[:, 2])
predicted_std_Kp = parameters[:, 1]

pop_Kp_mse = np.sqrt(np.mean((predicted_pop_Kp - true_pop_Kp)**2)) / np.abs(true_pop_Kp)
pop_Kp_bias = (np.mean(predicted_pop_Kp) - true_pop_Kp)  / np.abs(true_pop_Kp)
pop_Kp_variance = np.var(predicted_pop_Kp)

# pop_Kb_mse = np.round(np.sqrt(np.mean((predicted_kb - true_kb)**2)) / np.abs(true_kb), 4)
# pop_Kb_bias = np.round((np.mean(predicted_kb) - true_kb)  / np.abs(true_kb), 4)
# pop_Kb_variance = np.round(np.var(np.log(predicted_kb)),4)

# pop_Kac_mse = np.round(np.sqrt(np.mean((predicted_kac - true_kac)**2)) / np.abs(true_kac), 4)
# pop_Kac_bias = np.round((np.mean(predicted_kac) - true_kac)  / np.abs(true_kac), 4)
# pop_Kac_variance = np.round(np.var(predicted_kac),4)

std_Kp_mse = np.sqrt(np.mean((predicted_std_Kp - true_std_Kp)**2)) / np.abs(true_std_Kp)
std_Kp_bias = (np.mean(predicted_std_Kp)-true_std_Kp) / np.abs(true_std_Kp)
std_Kp_variance = np.var(predicted_std_Kp)

df.loc[len(df)] = {'parameter':'pop_Kp', 'Mean':np.mean(predicted_pop_Kp), 'RRMSE':pop_Kp_mse, 'Bias': pop_Kp_bias, 'Variance':pop_Kp_variance}
# df.loc[len(df)] = {'parameter':'pop_Kb', 'Mean':np.mean(predicted_kb), 'RRMSE':pop_Kb_mse, 'Bias': pop_Kb_bias, 'Variance':pop_Kp_variance}
# df.loc[len(df)] = {'parameter':'pop_Kac', 'Mean':np.mean(predicted_kac), 'RRMSE':pop_Kac_mse, 'Bias': pop_Kac_bias, 'Variance':pop_Kac_variance}
df.loc[len(df)] = {'parameter':'Kp_std', 'Mean':np.mean(predicted_std_Kp),'RRMSE':std_Kp_mse, 'Bias': std_Kp_bias, 'Variance':std_Kp_variance}

print(df)


