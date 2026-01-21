import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

df = pd.read_csv('results/antibody_400d_10p_50s_irregular_deltaAb_noise_2latents.txt', sep=" ", header=None)
deltaS_index = 5
deltaAb_index = 6
logstd_theta_index = 3
logstd_F2_index = 4

parameters = df.values
empirical_theta_variance = np.var(parameters[:,0])
empirical_F2_variance = np.var(parameters[:,1])
empirical_F3_variance = np.var(parameters[:,2])

empirical_deltaAb = np.var(parameters[:,deltaAb_index])
empirical_deltaS = np.var(parameters[:,deltaS_index])
empirical_logstd_theta_variance = np.var(np.exp(parameters[:,logstd_theta_index]))
empirical_logstd_F2_variance = np.var(np.exp(parameters[:,logstd_F2_index]))

logstd_theta_count = 0
logstd_F2_count = 0
theta_count = 0
F2_count = 0
F3_count = 0
deltaS_count = 0
deltaAb_count = 0

theta_variances = []
F2_variances = []
F3_variances = []
deltaS_variances = []
deltaAb_variances = []
logstd_theta_variances = []
logstd_F2_variances = []
total_count = 0

file_path = 'EXPs/antibody_3dose_irregular_deltaAB_nohessian_variance_400d_10p_50s_10000/'

for i in range(1, 100):
    try:
        filename = 'variance_matrix_'+str(i)+'.npy'        
        variance_matrix = np.load(file_path+filename)
        variance_matrix = np.diagonal(variance_matrix)

        if np.any(variance_matrix < 0):
            print('negative values', i)
            continue
        
        total_count += 1
        
        theta_variances.append(variance_matrix[0])
            
        interval = 1.96 * np.sqrt(variance_matrix[0])
        min = parameters[i-1, 0] - interval
        max = parameters[i-1, 0] + interval
        if min <= np.log(24.5) <= max:
            theta_count += 1
        # else:
        #     print("theta outside", i+1, parameters[i,:], variance_matrix, min, max)

        logstd_theta_variances.append(variance_matrix[logstd_theta_index] * np.exp(2*parameters[i-1, logstd_theta_index]))
        logstd_F2_variances.append(variance_matrix[logstd_F2_index] * np.exp(2*parameters[i-1, logstd_F2_index]))
        # deltaS_variances.append(variance_matrix[3])
        # logstd_theta_variances.append(variance_matrix[logstd_theta_index])
        # logstd_F2_variances.append(variance_matrix[logstd_F2_index])

        interval = 1.96 * np.sqrt(variance_matrix[1])
        min = parameters[i-1, 1] - interval
        max = parameters[i-1, 1] + interval
        if min <= np.log(7.1) <= max:
            F2_count += 1
        # else:
        #     print("F2 outside", i+1, parameters[i,:], variance_matrix, min, max)
        F2_variances.append(variance_matrix[1])

        interval = 1.96 * np.sqrt(variance_matrix[2])
        min = parameters[i-1, 2] - interval
        max = parameters[i-1, 2] + interval
        if min <= np.log(18.5) <= max:
            F3_count += 1
        # else:
        #     print("F3 outside", i+1, parameters[i,:], variance_matrix, min, max)
        F3_variances.append(variance_matrix[2])
        deltaS_variances.append(variance_matrix[deltaS_index])
        deltaAb_variances.append(variance_matrix[deltaAb_index])

        interval = 1.96 * np.sqrt(variance_matrix[deltaS_index])
        min = parameters[i-1, deltaS_index] - interval
        max = parameters[i-1, deltaS_index] + interval
        if min <= np.log(0.01) <= max:
            deltaS_count += 1

        interval = 1.96 * np.sqrt(variance_matrix[deltaAb_index])
        min = parameters[i-1, deltaAb_index] - interval
        max = parameters[i-1, deltaAb_index] + interval
        if min <= np.log(0.08) <= max:
            deltaAb_count += 1

        interval = 1.96 * np.sqrt(variance_matrix[logstd_theta_index] * np.exp(2*parameters[i-1, logstd_theta_index]))
        min = np.exp(parameters[i-1, logstd_theta_index]) - interval
        max = np.exp(parameters[i-1, logstd_theta_index]) + interval

        # interval = 1.96 * np.sqrt(variance_matrix[logstd_theta_index])
        # min = parameters[i-1, logstd_theta_index] - interval
        # max = parameters[i-1, logstd_theta_index] + interval
        if min <= 0.5 <= max:
            logstd_theta_count += 1
        # else:
        #     print("std theta outside", i, np.exp(parameters[i-1, logstd_theta_index]), variance_matrix, min, max)
        
        interval = 1.96 * np.sqrt(variance_matrix[logstd_F2_index] * np.exp(2*parameters[i-1, logstd_F2_index]))
        min = np.exp(parameters[i-1, logstd_F2_index]) - interval
        max = np.exp(parameters[i-1, logstd_F2_index]) + interval

        # interval = 1.96 * np.sqrt(variance_matrix[logstd_F2_index])
        # min = parameters[i-1, logstd_F2_index]- interval
        # max = parameters[i-1, logstd_F2_index] + interval
        if min <= 0.9 <= max:
            logstd_F2_count += 1
        # else:
        #     print("std F2 outside", i, np.exp(parameters[i-1, logstd_F2_index]), min, max)
   
    except Exception as error :
        print(error)
        continue


mean_theta_variance = np.mean(np.array(theta_variances))
mean_logstd_theta_variance = np.mean(np.array(logstd_theta_variances))
mean_logstd_F2_variance = np.mean(np.array(logstd_F2_variances))
mean_F2_variance = np.mean(np.array(F2_variances))
mean_F3_variance = np.mean(np.array(F3_variances))
mean_deltaS_variance = np.mean(np.array(deltaS_variances))
mean_deltaAb_variance = np.mean(np.array(deltaAb_variances))

print("mean_variance ", mean_theta_variance, mean_F2_variance, mean_F3_variance, mean_deltaS_variance, mean_deltaAb_variance, mean_logstd_theta_variance, mean_logstd_F2_variance)

print("empirical variance :", empirical_theta_variance, empirical_F2_variance, empirical_F3_variance, empirical_deltaS, empirical_deltaAb, empirical_logstd_theta_variance, empirical_logstd_F2_variance)

print(theta_count/total_count, F2_count/total_count, F3_count/total_count, deltaS_count/total_count, deltaAb_count/total_count, logstd_theta_count/total_count, logstd_F2_count/total_count)

print(total_count)