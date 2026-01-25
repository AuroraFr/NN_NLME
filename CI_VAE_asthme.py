import numpy as np
import pandas as pd

df = pd.read_csv('results/asthme_1latent_Kp_50s_kp.txt', sep=" ", header=None)
parameters = df.values
print("parameters shape", parameters.shape)

Kp_count = 0
kb_count = 0
kac_count = 0
std_Kp_count = 0

empirical_variances = np.array([0.000047, 0.000015])

interval = 1.96 * np.sqrt(empirical_variances)

total_count = 0
for i in range(99):
    if np.log(parameters[i, 0])- interval[0] <= np.log(1.15) <= np.log(parameters[i, 0]) + interval[0]:
        Kp_count += 1
    else:
        print(parameters[i, 0], parameters[i, 0]- interval[0], parameters[i, 0] + interval[0])

    # if np.exp(parameters[i, 1]) - interval[1] <= 1 <= np.exp(parameters[i, 1]) + interval[1]:
    #     kb_count += 1
    
    # if parameters[i, 2]- interval[2] <= np.log(0.01) <= parameters[i, 2] + interval[2]:
    #     kac_count += 1

    if parameters[i, 1]- interval[1] <= 0.05 <= parameters[i, 1] + interval[1]:
        std_Kp_count += 1


    total_count += 1

print(Kp_count, std_Kp_count)
