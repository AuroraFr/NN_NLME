import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

df = pd.read_csv('results/asthme_1latent_Kp_50s_kp.txt', sep=" ", header=None)
parameters = np.log(df.values)

Kp_count = 0
Kb_count = 0
Kac_count = 0
std_kp_count = 0

Kp_variances = []
Kb_variances = []
Kac_variances = []
std_Kp_variances = []
total_count = 0

file_path = 'EXPs/asthme_nohessian_kp_50s_20d_5000/'

for i in range(1, 100): 
    filename = 'variance_matrix_'+str(i)+'.npy'
    variance_matrix = np.load(file_path+filename)
    variance_matrix = np.diagonal(variance_matrix)

    if np.any(variance_matrix < 0):
        print('negative values', i)
        continue
    
    total_count += 1
    
    interval = 1.96 * np.sqrt(variance_matrix[0])
    min = parameters[i-1, 0] - interval
    max = parameters[i-1, 0] + interval
    Kp_variances.append(variance_matrix[0])

    if min <= np.log(1.15) <= max:
        Kp_count += 1
    # else:
    #     print("Kp outside", i+1, parameters[i,:], variance_matrix, min, max)

    # interval = 1.96 * np.sqrt(variance_matrix[1])
    # min = parameters[i-1, 1] - interval
    # max = parameters[i-1, 1] + interval
    # if min <= np.log(1) <= max:
    #     Kb_count += 1
    # # else:
    # #     print("Kb outside", i+1, parameters[i,:], variance_matrix, min, max)
    # Kb_variances.append(variance_matrix[1])

    # interval = 1.96 * np.sqrt(variance_matrix[2])
    # min = parameters[i-1, 2] - interval
    # max = parameters[i-1, 2] + interval
    # if min <= np.log(0.01) <= max:
    #     Kac_count += 1
    # # else:
    # #     print("Kac outside", i+1, parameters[i,:], variance_matrix, min, max)
    # Kac_variances.append(variance_matrix[2])

    interval = 1.96 * np.sqrt(variance_matrix[-1] * np.exp(2*parameters[i-1, -1]))
    min = np.exp(parameters[i-1, -1]) - interval
    max = np.exp(parameters[i-1, -1]) + interval
    std_Kp_variances.append(variance_matrix[-1] * np.exp(2*parameters[i-1, -1]))

    if min <= 0.05 <= max:
        std_kp_count += 1
        # else:
        #     print("std kp outside", i+1, parameters[i, -1], variance_matrix, min, max)
   

mean_Kp_variance = np.mean(np.array(Kp_variances))
mean_Kb_variance = np.mean(np.array(Kb_variances))
# mean_Kac_variance = np.mean(np.array(Kac_variances))
mean_std_Kp_variance = np.mean(np.array(std_Kp_variances))

print("mean_variance ", mean_Kp_variance,  mean_std_Kp_variance)

print(Kp_count/total_count,  std_kp_count/total_count)

print(total_count)
