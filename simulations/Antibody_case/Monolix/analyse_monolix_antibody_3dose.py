import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['parameter', 'true_value','estimated_mean_value','RRMSE', 'RBias','Variance'])
folder = '/beegfs/zli/workspace/monolix/antibody_results/irregular_10p_400d_50s_results/'
true_theta = np.log(24.5)
true_F2 = np.log(7.1)
true_F3 = np.log(18.5)
true_omega_theta = 0.5
true_omega_F2 = 0.9
true_noise_var = 0.1 ** 2
true_deltaS = np.log(0.01)

thetas = []
omega_thetas = []
omega_F2s = []
F2s = []
F3s = []
deltaS = []
deltaAb = []

for i in range(2, 101):
    data = pd.read_csv(folder+'antibody_param_'+str(i)+'.csv')
    # print(data)
    theta = data.iloc[0]
    F2 = data.iloc[1]
    F3 = data.iloc[2]
    omega_theta = data.iloc[6]
    omega_F2 = data.iloc[7]
    
    thetas.append(theta)
    F2s.append(F2)
    F3s.append(F3)
    # deltaAb.append(data.iloc[5])
    # deltaS.append(data.iloc[4])
    omega_thetas.append(omega_theta)
    omega_F2s.append(omega_F2)
    

thetas = np.array(thetas).squeeze()
omega_thetas = np.array(omega_thetas).squeeze()
omega_F2s = np.array(omega_F2s).squeeze()
F2s = np.array(F2s).squeeze()
F3s = np.array(F3s).squeeze()
# deltaS = np.array(deltaS).squeeze()
# deltaAb = np.array(deltaAb).squeeze()

# result_df = pd.DataFrame(columns=['theta', 'F2','F3','deltaS', 'deltaAb','omega_theta', 'omega_F2'])
result_df = pd.DataFrame(columns=['theta', 'F2','F3', 'omega_theta', 'omega_F2'])
result_df['theta'] = thetas
result_df['F2'] = F2s
result_df['F3'] = F3s
# result_df['deltaS'] = deltaS
# result_df['deltaAb'] = deltaAb
result_df['omega_theta'] = omega_thetas
result_df['omega_F2'] = omega_F2s

print(result_df)
result_df.to_csv('results/monolix_irregular_10_400_50.csv', index=False, float_format='%.4f')
mean_theta = np.mean(thetas)
mean_omega_theta = np.mean(omega_thetas)
mean_F2 = np.mean(F2s)
mean_omega_F2 = np.mean(omega_F2s)
mean_F3 = np.mean(F3s)
# mean_deltaS = np.mean(deltaS)

rrmse_theta = np.sqrt(np.mean((thetas - true_theta)**2)) / np.abs(true_theta)
rrmse_omega_theta = np.sqrt(np.mean((omega_thetas - true_omega_theta)**2)) / np.abs(true_omega_theta)
rrmse_F2 = np.sqrt(np.mean((F2s - true_F2)**2)) / np.abs(true_F2)
rrmse_omega_F2 = np.sqrt(np.mean((omega_F2s - true_omega_F2)**2)) / np.abs(true_omega_F2)
rrmse_F3 = np.sqrt(np.mean((F3s - true_F3)**2)) / np.abs(true_F3)
# rrmse_deltaS = np.sqrt(np.mean((deltaS - true_deltaS)**2)) / np.abs(true_deltaS)


rbias_theta = np.mean(thetas - true_theta)/np.abs(true_theta) 
rbias_omega_theta = np.mean(omega_thetas - true_omega_theta)/ np.abs(true_omega_theta)
rbias_F2 = np.mean(F2s - true_F2)/np.abs(true_F2) 
rbias_F3 = np.mean(F3s - true_F3)/np.abs(true_F3) 
rbias_omega_F2 = np.mean(omega_F2s - true_omega_F2)/ np.abs(true_omega_F2)
# rbias_deltaS = np.mean(deltaS - true_deltaS)/ np.abs(true_deltaS)

variance_theta = np.var(thetas)
variance_omega_theta = np.var(omega_thetas)
variance_F2 = np.var(F2s)
variance_F3 = np.var(F3s)
variance_omega_F2 = np.var(omega_F2s)
# variance_deltaS = np.var(deltaS)

df.loc[len(df)] = {'parameter':'pop_theta', 'true_value':true_theta,'estimated_mean_value':mean_theta,'RRMSE':rrmse_theta, 'RBias': rbias_theta, 'Variance':variance_theta}
df.loc[len(df)] = {'parameter':'pop_F2', 'true_value':true_F2,'estimated_mean_value':mean_F2,'RRMSE':rrmse_F2, 'RBias': rbias_F2, 'Variance':variance_F2}
df.loc[len(df)] = {'parameter':'pop_F3', 'true_value':true_F3,'estimated_mean_value':mean_F3,'RRMSE':rrmse_F3, 'RBias': rbias_F3, 'Variance':variance_F3}
# df.loc[len(df)] = {'parameter':'pop_deltaS', 'true_value':true_deltaS,'estimated_mean_value':mean_deltaS,'RRMSE':rrmse_deltaS, 'RBias': rbias_deltaS, 'Variance':variance_deltaS}
df.loc[len(df)] = {'parameter':'omega_theta', 'true_value':true_omega_theta,'estimated_mean_value':mean_omega_theta,'RRMSE':rrmse_omega_theta, 'RBias': rbias_omega_theta, 'Variance':variance_omega_theta}
df.loc[len(df)] = {'parameter':'omega_F2', 'true_value':true_omega_F2,'estimated_mean_value':mean_omega_F2,'RRMSE':rrmse_omega_F2, 'RBias': rbias_omega_F2, 'Variance':variance_omega_F2}

print(df.shape)
variances = np.array([variance_theta, variance_F2,variance_F3, variance_omega_theta, variance_omega_F2])
intervals = 1.96 * np.sqrt(variances)
print(variances, intervals)
print(np.mean((omega_F2s - true_omega_F2)**2), np.mean(omega_F2s - true_omega_F2) ** 2 + variance_omega_F2)
theta_count = 0
F2_count = 0
F3_count = 0
std_theta = 0
std_F2 = 0
deltaS_count = 0

for i in range(99):

    if thetas[i] - intervals[0] <= np.log(24.5) <= thetas[0] + intervals[0]:
        theta_count += 1
    if F2s[i] - intervals[1] <= np.log(7.1) <= F2s[i] + intervals[1]:
        F2_count += 1
    if F3s[i] - intervals[2] <= np.log(18.5) <= F3s[i] + intervals[2]:
        F3_count += 1
    
    # if deltaS[i] - intervals[3] <= np.log(0.01) <= deltaS[i] + intervals[3]:
    #     deltaS_count += 1
    
    if omega_thetas[i] - intervals[3] <= 0.5 <= omega_thetas[i] + intervals[3]:
        std_theta += 1

    if omega_F2s[i] - intervals[4] <= 0.9 <= omega_F2s[i] + intervals[4]:
        std_F2 += 1


print(theta_count/99, F2_count/99, F3_count/99, deltaS_count/99, std_theta/99, std_F2/99)
