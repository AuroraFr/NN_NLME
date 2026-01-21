import numpy as np
import pandas as pd

theta1_list=[]
theta2_list=[]
theta1_std_list=[]
noise_std_list=[]

folder = '/beegfs/zli/workspace/monolix/PKPD_results/irregular_5p_10d_100s_noise_results/'
for i in range(1, 100):
    params = pd.read_csv(folder+"/pkpd_param_"+str(i)+".csv")
    theta1_list.append(params.iloc[0])
    theta2_list.append(params.iloc[1])
    theta1_std_list.append(params.iloc[2])
    noise_std_list.append(params.iloc[3])


theta1_list = np.array(theta1_list)
theta2_list = np.array(theta2_list)
theta1_std_list = np.array(theta1_std_list)
noise_std_list = np.array(noise_std_list)

df = pd.DataFrame(columns=['parameter', 'RRMSE', 'RBias','Variance'])
true_theta1 = np.log(0.5)
true_theta2 = np.log(2)
true_omega_theta1 = np.log(0.5)
true_noise_var = np.log(0.2 ** 2)

mse_theta1 = np.sqrt(np.mean((theta1_list - true_theta1)**2)) / np.abs(true_theta1)
mse_theta2 = np.sqrt(np.mean((theta2_list - true_theta2)**2)) / np.abs(true_theta2)
mse_omega_theta1 = np.sqrt(np.mean((np.log(theta1_std_list) - true_omega_theta1)**2)) / true_omega_theta1
mse_noise = np.sqrt(np.mean((np.log(noise_std_list**2) - true_noise_var)**2)) / true_noise_var

bias_theta1 = np.mean(theta1_list - true_theta1)/true_theta1 
bias_theta2 = np.mean(theta2_list - true_theta2)/true_theta2
bias_omega_theta1 = np.mean(np.log(theta1_std_list) - true_omega_theta1)/true_omega_theta1
bias_noise = np.mean(np.log(noise_std_list**2) - true_noise_var)/true_noise_var

variance_theta1 = np.var(theta1_list)
variance_theta2 = np.var(theta2_list)
variance_omega_theta1 = np.var(np.log(theta1_std_list))
variance_noise = np.var(np.log(noise_std_list**2))


df.loc[len(df)] = {'parameter':'pop_theta1', 'RRMSE':mse_theta1, 'RBias': bias_theta1, 'Variance':variance_theta1}
df.loc[len(df)] = {'parameter':'pop_theta2', 'RRMSE':mse_theta2, 'RBias': bias_theta2, 'Variance':variance_theta2}
df.loc[len(df)] = {'parameter':'omega_theta1', 'RRMSE':mse_omega_theta1, 'RBias': bias_omega_theta1, 'Variance':variance_omega_theta1}
df.loc[len(df)] = {'parameter':'noise', 'RRMSE':mse_noise, 'RBias': bias_noise, 'Variance':variance_noise}

print(df)
df.to_csv('results/monolix_pkpd_irregular_5_result_noise.csv')
