import numpy as np
import pandas as pd

df = pd.read_csv('results/antibody_3doses_400d_10p_50s_irregular_2latents.txt', sep=" ", header=None)
parameters = df.values
print("parameters shape", parameters.shape)

theta_count = 0
F2_count = 0
F3_count = 0
std_theta = 0
std_F2 = 0
deltaS = 0
deltaAb = 0

empirical_variances = np.var(parameters, axis=0)
print(empirical_variances)

interval = 1.96 * np.sqrt(empirical_variances)

total_count = 0
for i in range(99):
    if parameters[i, 0]- interval[0] <= np.log(24.5) <= parameters[i, 0] + interval[0]:
        theta_count += 1

    if parameters[i, 1]- interval[1] <= np.log(7.1) <= parameters[i, 1] + interval[1]:
        F2_count += 1

    if parameters[i, 2]- interval[2] <= np.log(18.5) <= parameters[i, 2] + interval[2]:
        F3_count += 1

    # if parameters[i, 4]- interval[4] <= np.log(0.01) <= parameters[i, 4] + interval[4]:
    #     deltaS += 1
    
    # if parameters[i, 3]- interval[3] <= np.log(0.08) <= parameters[i, 3] + interval[3]:
    #     deltaAb += 1
    
    if parameters[i, -2]- interval[-2] <= np.log(0.5) <= parameters[i, -2] + interval[-2]:
        std_theta += 1
   
    
    if parameters[i, -1]- interval[-1] <= np.log(0.9) <= parameters[i, -1] + interval[-1]:
        std_F2 += 1

    total_count += 1

print(total_count)
print(theta_count/total_count, F2_count/total_count, F3_count/total_count, std_theta/total_count, std_F2/total_count)