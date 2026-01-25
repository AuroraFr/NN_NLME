import pandas as pd
import os
import numpy as np

#lambda_value = torch.tensor(0.2).repeat(10)
#init_condition = torch.tensor(2.0).repeat(10)
#random_effect_sigma = torch.log(torch.tensor(0.9**2)).repeat(10)
#measure_error_sigma = 0.2

#modalities_dim = 1
#modalities={'lambda':11}
#latent_dim = 1

#sigma0 = [1**2]
L_values = [100]
#prior_variance=[0.5**2]

init_conditions = [2.0, 3.0]
modalities={'phi1':11}
latent_dim = 1
integration_time = np.linspace(0,10,11)
sigma0 = [0.5**2]
prior_phi1_variance = [0.8**2]
start_values = {'theta1': 0.3, 'theta2': 1.0, 'b': prior_phi1_variance}
results = {}

# for i in range(100):
#     model_name = 'PKPD_Quality_'+str(i)+'_L50'
#     new_modelname = 'PKPD_Quality_'+str(i)+'_L100'
#     checkpoint_path="./exp_quality_pkpd_nonoise_0.2_11_100_advanced/"
#     os.rename(checkpoint_path + model_name, checkpoint_path + new_modelname)

df = pd.DataFrame(columns=['parameter', 'RRMSE', 'Bias','Variance'])
variances_list = []

# calculate MSE
parameters = np.loadtxt("results/PKPD_irregular_5p_100_10d_noise.txt")

true_noise_variance = np.log(0.2 ** 2)
true_prior_sigma = np.log(0.5)
true_pop_theta1 = np.log(0.5)
true_pop_theta2 = np.log(2)

predicted_pop_theta1 = parameters[:, 0]
predicted_pop_theta2 = parameters[:, 1]
predicted_noise_variance = parameters[:,3]
predicted_prior_sigma = parameters[:,2]

noise_variance_mse = np.sqrt(np.mean((predicted_noise_variance - true_noise_variance)**2)) / np.abs(true_noise_variance)
noise_variance_bias = (np.mean(predicted_noise_variance)-true_noise_variance)/true_noise_variance
noise_variance_variance = np.var(predicted_noise_variance)

prior_variance_mse = np.sqrt(np.mean((predicted_prior_sigma - true_prior_sigma)**2)) / np.abs(true_prior_sigma)
prior_variance_bias = (np.mean(predicted_prior_sigma)-true_prior_sigma) / true_prior_sigma
prior_variance_variance = np.var(predicted_prior_sigma)

pop_theta1_mse = np.sqrt(np.mean((predicted_pop_theta1 - true_pop_theta1)**2)) / true_pop_theta1
pop_theta1_bias = (np.mean(predicted_pop_theta1) - true_pop_theta1)  / true_pop_theta1
pop_theta1_variance = np.var(parameters[:, 0])
# pop_theta1_variance = np.var(predicted_pop_theta1)

pop_theta2_mse = np.sqrt(np.mean((predicted_pop_theta2 - true_pop_theta2)**2)) / np.abs(true_pop_theta2)
pop_theta2_bias = (np.mean(predicted_pop_theta2) - true_pop_theta2)/ true_pop_theta2
pop_theta2_variance = np.var(parameters[:, 1])
# pop_theta2_variance = np.var(predicted_pop_theta2)

df.loc[len(df)] = {'parameter':'pop_theta1', 'RRMSE':pop_theta1_mse, 'Bias': pop_theta1_bias, 'Variance':pop_theta1_variance}
df.loc[len(df)] = {'parameter':'pop_theta2', 'RRMSE':pop_theta2_mse, 'Bias': pop_theta2_bias, 'Variance':pop_theta2_variance}
df.loc[len(df)] = {'parameter':'theta1_prior_variance', 'RRMSE':prior_variance_mse, 'Bias': prior_variance_bias, 'Variance':prior_variance_variance}
df.loc[len(df)] = {'parameter':'noise_variance', 'RRMSE':noise_variance_mse, 'Bias': noise_variance_bias, 'Variance':noise_variance_variance}

df.to_csv('PKPD_irregular_5_noise.csv')

