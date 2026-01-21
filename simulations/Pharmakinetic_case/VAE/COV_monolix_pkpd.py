import numpy as np
from utils import calculate_CI
import pandas as pd

# True_params = [0.5, -0.6931,  0.6931, 0.2]


df = pd.read_csv('./exp_results_reports/monolix_nosigma_pkpd_estimations.txt', sep="    ")
# df['a'] = df['a'] ** 2
print(df.describe())
parameters = df.values
intervals = []
std_dfs = []

for i, parameter in enumerate(parameters):
    std_df = pd.read_csv('./EXPs/monolix_pkpd_nonoise_sd_results/pkpd_'+str(i)+'.csv')['stochasticApproximation.se']
    variances = std_df.values
    std_dfs.append(variances)
    interval = calculate_CI(parameter, variances)
    intervals.append(interval)
   
   
prior_sd_count = 0
theta1_count = 0
theta2_count = 0
noise_var_count = 0

for i, interval in enumerate(intervals):
    if interval[0][0] <= -0.6931 <= interval[0][1]:
        theta1_count += 1
    else:
        print("theta1", parameters[i][0],interval[0])
    
    if interval[1][0] <= 0.6931 <= interval[1][1]:
        theta2_count += 1
    else:
        print("theta2", parameters[i][1],interval[1])
    
    if interval[2][0] <= 0.5 <= interval[2][1]:
        prior_sd_count += 1
    else:
        print("prior", parameters[i][2],interval[2])
    
    # if interval[3][0] <= 0.2 <= interval[3][1]:
    #     noise_var_count += 1
    # else:
    #     print(parameters[i])


print(prior_sd_count, theta1_count, theta2_count, noise_var_count)

std_dfs = np.array(std_dfs)
print(np.mean(std_dfs, axis=0)**2)