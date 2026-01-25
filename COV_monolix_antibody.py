import numpy as np
import pandas as pd

folder = '/beegfs/zli/workspace/monolix/antibody_3doses_15p_400d_50s_deltaS_results/'
parameters = []
std_dfs = []
omega_theta_count = 0
theta_count = 0
omega_F2_count=0
F2_count = 0
F3_count = 0
deltaS_count = 0

for i in range(2,101):

    std_df = pd.read_csv(folder+'antibody_sd_'+str(i)+'.csv')['stochasticApproximation.se']
    param_df = pd.read_csv(folder+'antibody_param_'+str(i)+'.csv')
    parameter = param_df.values
    theta = parameter[0]
    F2 = parameter[1]
    F3 = parameter[2]
    deltaS = parameter[4]
    omega_theta = parameter[6]
    omega_F2 = parameter[7]
    std = std_df.values
    if np.isnan(std).any():
        print("NaN detected", std, i)
        std[3] = 0
    else:
        std_dfs.append(std)

    if theta - 1.96 * std[0] <= np.log(24.5) <= theta + 1.96 * std[0]:
        theta_count += 1
    if F2 - 1.96 * std[1] <= np.log(7.1) <= F2 + 1.96 * std[1]:
        F2_count += 1
    if F3 - 1.96 * std[2] <= np.log(18.5) <= F3 + 1.96 * std[2]:
        F3_count += 1
    else:
        print('F3 outside', F3, F3 - 1.96 * std[2], F3 + 1.96 * std[2], np.log(18.5), std[2]**2)
    
    if not np.isnan(std[3]):
        if deltaS - 1.96 * std[3] <= np.log(0.01) <= deltaS + 1.96 * std[3]:
            deltaS_count += 1
        else:
            print('deltaS outside', deltaS, deltaS - 1.96 * std[3], deltaS + 1.96 * std[3])
    
    if omega_theta - 1.96 * std[4] <= 0.5 <= omega_theta + 1.96 * std[4]:
        omega_theta_count += 1

    if omega_F2 - 1.96 * std[5] <= 0.9 <= omega_F2 + 1.96 * std[5]:
        omega_F2_count += 1

print(theta_count/99, F2_count/99, F3_count/99, deltaS_count/97, omega_theta_count/99, omega_F2_count/99)

std_dfs = np.array(std_dfs)
print(np.mean(std_dfs ** 2, axis=0))
