import os
os.environ['JAX_PLATFORMS']='cpu'
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import numpy as np
import jax.random as jr
from jax_antibody_multidoses_joint_model import Antibody

latent_shape = 2
n_z = 1
basic = False
deltaS = True
deltaAb = False

init_theta = jnp.log(30.0)  # theta
init_theta_logstd = jnp.log(0.8) # random effect std on theta
init_delta_S = 0.05
init_noise_var = jnp.log(0.1 ** 2)
init_F2 = jnp.log(8.0)
init_F3 = jnp.log(18.5)
init_F2_logstd = jnp.log(1)

seed = 42
key = jr.PRNGKey(seed)

init_params = jnp.array([init_theta, init_F2, init_F3, init_theta_logstd, init_F2_logstd, 0.01, 0.1**2])

parameters =[]

from os import listdir
from os.path import isfile, join

file_path = 'EXPs/exp_antibody_3dose_2latent/irregular_10p_400d_50s_deltaAB_noise_2latent_theta_F2'
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]

for i in range(1, 100):
    filename = "Antibody_"+str(i)+"_rnn.eqx"
    model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, key, n_z=n_z, irregular=True, 
                                                timepoints=10, hidden_dim=32, rnn=True, conv=False, out_channel=1, input_channel=3)

    try:
        model_loaded = eqx.tree_deserialise_leaves(file_path+"/"+filename, model)
        if basic:
            parameters.append([model_loaded.theta, model_loaded.F2, model_loaded.F3, 
                               model_loaded.logstd_theta, model_loaded.logstd_F2, model_loaded.log_noise_var])
        elif deltaS:
            parameters.append([model_loaded.theta, model_loaded.F2, model_loaded.F3, model_loaded.logstd_theta, 
                               model_loaded.logstd_F2, model_loaded.deltaS,model_loaded.log_noise_var])
        else:
            deltaAb = jnp.exp(model_loaded.deltaS) + jnp.exp(model_loaded.deltaAB)
            parameters.append([model_loaded.theta, model_loaded.F2, model_loaded.F3, model_loaded.logstd_theta, 
                               model_loaded.logstd_F2, model_loaded.deltaS, jnp.log(deltaAb), model_loaded.log_noise_var])
    except Exception as error:
        print(error)

np.savetxt('results/antibody_400d_10p_50s_irregular_deltaAb_noise_2latents.txt', parameters)

parameters = pd.read_csv('results/antibody_400d_10p_50s_irregular_deltaAb_noise_2latents.txt', header=None, sep=' ').values

df = pd.DataFrame(columns=['parameter', 'Mean', 'RRMSE', 'Bias','Variance'])
print(parameters.shape)

true_std_theta = 0.5
true_pop_theta = np.log(24.5)
true_F2 = np.log(7.1)
true_F3 = np.log(18.5)
true_std_F2 = 0.9
true_deltaS = np.log(0.01)
true_deltaAB = np.log(0.08)
true_log_noisevar = np.log(0.1**2)

predicted_pop_theta= parameters[:, 0]
predicted_F2 = parameters[:, 1]
predicted_F3 = parameters[:, 2]

predicted_std_theta = np.exp(parameters[:,3])
predicted_std_F2 = np.exp(parameters[:, 4])

if deltaS or deltaAb:
    predicted_deltaS = parameters[:,5]

if deltaAb:
    predicted_deltaAB = parameters[:,6]

predicted_noisevar = parameters[:,-1]

pop_theta_mse = np.round(np.sqrt(np.mean((predicted_pop_theta - true_pop_theta)**2)) / np.abs(true_pop_theta), 4)
pop_theta_bias = np.round((np.mean(predicted_pop_theta) - true_pop_theta)  / np.abs(true_pop_theta), 4)
pop_theta_variance = np.round(np.var(predicted_pop_theta),4)

pop_F2_mse = np.round(np.sqrt(np.mean((predicted_F2 - true_F2)**2)) / np.abs(true_F2),4)
pop_F2_bias = np.round((np.mean(predicted_F2) - true_F2)  / np.abs(true_F2),4)
pop_F2_variance = np.round(np.var(predicted_F2),4)

pop_F3_mse = np.round(np.sqrt(np.mean((predicted_F3 - true_F3)**2)) / np.abs(true_F3),4)
pop_F3_bias = np.round((np.mean(predicted_F3) - true_F3)  / np.abs(true_F3),4)
pop_F3_variance = np.round(np.var(predicted_F3),4)

std_theta_mse = np.round(np.sqrt(np.mean((predicted_std_theta - true_std_theta)**2)) / np.abs(true_std_theta),4)
std_theta_bias = np.round((np.mean(predicted_std_theta)-true_std_theta) / np.abs(true_std_theta),4)
std_theta_variance = np.var(predicted_std_theta)

std_F2_mse = np.round(np.sqrt(np.mean((predicted_std_F2 - true_std_F2)**2)) / np.abs(true_std_F2), 4)
std_F2_bias = np.round((np.mean(predicted_std_F2) - true_std_F2)  / np.abs(true_std_F2),4)
std_F2_variance = np.round(np.var(predicted_std_F2),4)

noisevar_mse = np.round(np.sqrt(np.mean((predicted_noisevar - true_log_noisevar)**2)) / np.abs(true_log_noisevar), 4)
noisevar_bias = np.round((np.mean(predicted_noisevar) - true_log_noisevar)  / np.abs(true_log_noisevar),4)
noisevar_variance = np.round(np.var(predicted_noisevar),4)

if deltaS or deltaAb:
    deltaS_rrmse = np.round(np.sqrt(np.mean((predicted_deltaS - true_deltaS)**2)) / np.abs(true_deltaS),4)
    deltaS_bias = np.round((np.mean(predicted_deltaS-true_deltaS)) / np.abs(true_deltaS),4)
    deltaS_variance = np.var(predicted_deltaS)

if deltaAb:
    deltaAB_rrmse = np.round(np.sqrt(np.mean((predicted_deltaAB - true_deltaAB)**2)) / np.abs(true_deltaAB),4)
    deltaAB_bias = np.round((np.mean(predicted_deltaAB - true_deltaAB)) / np.abs(true_deltaAB),4)
    deltaAB_variance = np.var(predicted_deltaAB)


df.loc[len(df)] = {'parameter':'pop_theta', 'Mean':np.mean(predicted_pop_theta), 'RRMSE':pop_theta_mse, 'Bias': pop_theta_bias, 'Variance':pop_theta_variance}
df.loc[len(df)] = {'parameter':'pop_F2', 'Mean':np.mean(predicted_F2),'RRMSE':pop_F2_mse, 'Bias': pop_F2_bias, 'Variance':pop_F2_variance}
df.loc[len(df)] = {'parameter':'pop_F3', 'Mean':np.mean(predicted_F3),'RRMSE':pop_F3_mse, 'Bias': pop_F3_bias, 'Variance':pop_F3_variance}
df.loc[len(df)] = {'parameter':'theta_rm_std', 'Mean':np.mean(predicted_std_theta),'RRMSE':std_theta_mse, 'Bias': std_theta_bias, 'Variance':std_theta_variance}
df.loc[len(df)] = {'parameter':'F2_rm_std', 'Mean':np.mean(predicted_std_F2),'RRMSE':std_F2_mse, 'Bias': std_F2_bias, 'Variance':std_F2_variance}
df.loc[len(df)] = {'parameter':'noise_var', 'Mean':np.mean(predicted_noisevar),'RRMSE':noisevar_mse, 'Bias': noisevar_bias, 'Variance':noisevar_variance}

if deltaS or deltaAb:
    df.loc[len(df)] = {'parameter':'deltaS', 'Mean':np.mean(predicted_deltaS),'RRMSE':deltaS_rrmse, 'Bias': deltaS_bias, 'Variance':deltaS_variance}

if deltaAb:
    df.loc[len(df)] = {'parameter':'deltaAB', 'Mean':np.mean(predicted_deltaAB),'RRMSE':deltaAB_rrmse, 'Bias': deltaAB_bias, 'Variance':deltaAB_variance}
    
print(df)
df.to_csv('results/VAE_antibody_irregular_10_deltaAb_noise.csv')


