import numpy as np
import pandas as pd

df = pd.read_csv('results/PKPD_irregular_5p_100_10d_noise.txt', sep=" ", header=None)
parameters = df.values

prior_sd_count = 0
theta1_count = 0
theta2_count = 0
noise_var_count = 0

prior_logstd = parameters[:,2]
noise_vars = parameters[:,-1]

total_count_theta1 = 0
total_count_theta2 = 0
total_count_prior = 0
total_count_noise = 0
total_count = 0

theta1_variances = []
theta2_variances = []
omega_variances = []
noise_variances = []
for i in range(1, 100):
    try:
        result = np.load('EXPs/pkpd_variance_nohessian_5000/variance_matrix_'+str(i)+'.npy')
        
        variance_matrix = np.diagonal(result)
        if variance_matrix[0] < 0 or variance_matrix[1] <0 or variance_matrix[2] <0 :
            print('negative values', i)
            continue
        
        theta1_variances.append(variance_matrix[0])
        theta2_variances.append(variance_matrix[1])
        omega_variances.append(variance_matrix[2])
        noise_variances.append(variance_matrix[3])

        total_count_theta1 += 1
        interval = 1.96 * np.sqrt(variance_matrix[0])
        min = parameters[i-1, 0] - interval
        max = parameters[i-1, 0] + interval
        if min <= np.log(0.5) <= max:
            theta1_count += 1
        # else:
        #     print("theta1 outside", i, parameters[i,:], variance_matrix, min, max)
        
        total_count_theta2 += 1
        interval = 1.96 * np.sqrt(variance_matrix[1])
        min = parameters[i-1, 1] - interval
        max = parameters[i-1, 1] + interval
        if min <= np.log(2) <= max:
            theta2_count += 1
        # else:
        #     print("theta2 outside", i, parameters[i,:], variance_matrix, min, max)

       
        total_count_prior += 1
        interval = 1.96 * np.sqrt(variance_matrix[2])
        min = prior_logstd[i-1] - interval
        max = prior_logstd[i-1] + interval
        # min = parameter[2] - interval
        # max = parameter[2] + interval
        if min <= np.log(0.5) <= max:
            prior_sd_count += 1
        # else:
        #     print("prior outside", i, parameters[i,:], variance_matrix, min, max)
        
        interval = 1.96 * np.sqrt(variance_matrix[3])
        min = noise_vars[i-1] - interval
        max = noise_vars[i-1] + interval
        if min <= np.log(0.2 ** 2) <= max:
            noise_var_count += 1
        else:
            print("noise outside", i, parameters[i-1, -1], variance_matrix[3])
        
        total_count += 1

    except Exception as error :
        print(error)
        continue

mean_theta1_variance = np.mean(np.array(theta1_variances))
mean_theta2_variance = np.mean(np.array(theta2_variances))
mean_omega_variance = np.mean(np.array(omega_variances))
mean_noise_variance = np.mean(np.array(noise_variances))
print(prior_sd_count/total_count_prior, theta1_count/total_count_theta1, theta2_count/total_count_theta2, noise_var_count/total_count)
print(prior_sd_count, total_count_prior, theta1_count, total_count_theta1, theta2_count, total_count_theta2, noise_var_count, total_count)
print(mean_theta1_variance, mean_theta2_variance, mean_omega_variance, mean_noise_variance)