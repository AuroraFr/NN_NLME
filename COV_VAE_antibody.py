import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def coverage_from_estimated_variance(summary_df, params_file, variance_folder = 'EXPs/antibody_3dose_irregular_deltaAB_nohessian_variance_400d_10p_50s_8000/'):

    df = pd.read_csv(params_file, sep=" ", header=None)
    deltaS_index = 6
    deltaAb_index = 7
    logstd_theta_index = 3
    logstd_F2_index = 4

    parameters = df.values

    logstd_theta_count = 0
    logstd_F2_count = 0
    theta_count = 0
    F2_count = 0
    F3_count = 0
    deltaS_count = 0
    deltaAb_count = 0
    sigma_count = 0

    theta_variances = []
    F2_variances = []
    F3_variances = []
    deltaS_variances = []
    deltaAb_variances = []
    logstd_theta_variances = []
    logstd_F2_variances = []
    sigma_variances = []
    total_count = 0

    for i in range(1, 100):
        try:
            filename = 'variance_matrix_'+str(i)+'.npy'        
            variance_matrix = np.load(variance_folder+filename)
            variance_matrix = np.diagonal(variance_matrix)
            
            total_count += 1
            
            theta_variances.append(variance_matrix[0])
                
            interval = 1.96 * np.sqrt(variance_matrix[0])
            min = parameters[i-1, 0] - interval
            max = parameters[i-1, 0] + interval
            if min <= np.log(24.5) <= max:
                theta_count += 1
        
            logstd_theta_variances.append(variance_matrix[logstd_theta_index] * np.exp(2*parameters[i-1, logstd_theta_index]))
            logstd_F2_variances.append(variance_matrix[logstd_F2_index] * np.exp(2*parameters[i-1, logstd_F2_index]))

            interval = 1.96 * np.sqrt(variance_matrix[1])
            min = parameters[i-1, 1] - interval
            max = parameters[i-1, 1] + interval
            if min <= np.log(7.1) <= max:
                F2_count += 1
        
            F2_variances.append(variance_matrix[1])

            interval = 1.96 * np.sqrt(variance_matrix[2])
            min = parameters[i-1, 2] - interval
            max = parameters[i-1, 2] + interval
            if min <= np.log(18.5) <= max:
                F3_count += 1

            interval = 1.96 * np.sqrt(0.25 * variance_matrix[5] * np.exp(parameters[i-1, 5]))
            min = np.exp(0.5 * parameters[i-1, 5]) - interval
            max = np.exp(0.5 * parameters[i-1, 5]) + interval
            if min <= 0.1 <= max:
                sigma_count += 1

            F3_variances.append(variance_matrix[2])
            deltaS_variances.append(variance_matrix[deltaS_index])
            deltaAb_variances.append(variance_matrix[deltaAb_index])
            sigma_variances.append(variance_matrix[5])

            interval = 1.96 * np.sqrt(variance_matrix[deltaS_index])
            min = parameters[i-1, deltaS_index] - interval
            max = parameters[i-1, deltaS_index] + interval
            if min <= np.log(0.01) <= max:
                deltaS_count += 1

            interval = 1.96 * np.sqrt(variance_matrix[deltaAb_index])
            min = parameters[i-1, deltaAb_index] - interval
            max = parameters[i-1, deltaAb_index] + interval
            if min <= np.log(0.07) <= max:
                deltaAb_count += 1

            interval = 1.96 * np.sqrt(variance_matrix[logstd_theta_index] * np.exp(2*parameters[i-1, logstd_theta_index]))
            min = np.exp(parameters[i-1, logstd_theta_index]) - interval
            max = np.exp(parameters[i-1, logstd_theta_index]) + interval

            if min <= 0.5 <= max:
                logstd_theta_count += 1
            
            interval = 1.96 * np.sqrt(variance_matrix[logstd_F2_index] * np.exp(2*parameters[i-1, logstd_F2_index]))
            min = np.exp(parameters[i-1, logstd_F2_index]) - interval
            max = np.exp(parameters[i-1, logstd_F2_index]) + interval

            if min <= 0.9 <= max:
                logstd_F2_count += 1
    
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
    mean_sigma_variance = np.mean(np.array(sigma_variances))

    print("mean_variance ", mean_theta_variance, mean_F2_variance, mean_F3_variance, mean_logstd_theta_variance, mean_logstd_F2_variance, mean_sigma_variance, mean_deltaS_variance, mean_deltaAb_variance)
    print(theta_count/total_count, F2_count/total_count, F3_count/total_count, deltaS_count/total_count, deltaAb_count/total_count, logstd_theta_count/total_count, logstd_F2_count/total_count, sigma_count/total_count)

    #add information in summary_df
    df['estimated_variance'] = [mean_theta_variance, mean_F2_variance, mean_F3_variance, mean_logstd_theta_variance, mean_logstd_F2_variance, mean_sigma_variance, mean_deltaS_variance, mean_deltaAb_variance]
    df['estimated_coverage'] = [theta_count/total_count, F2_count/total_count, F3_count/total_count, logstd_theta_count/total_count, logstd_F2_count/total_count, sigma_count/total_count, deltaS_count/total_count, deltaAb_count/total_count]

    return summary_df