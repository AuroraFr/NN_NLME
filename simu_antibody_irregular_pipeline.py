#Pipeline to run VAE on regular simulated antibody kinetic data
import antibody_multidoses_train
from FIM_antibody_multidose_prior_reparam import variance_estimation
from analyse_VAE_antibody import analyse_poitwise_estimation
from COV_VAE_antibody import coverage_from_estimated_variance
from pathlib import Path
import numpy as np
from scipy.stats import norm
import os

b_sample_size = 8000
N_subjects = 50
#train the model and return the estimated params
estimated_parameters = []
data_folder = 'antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/'
model_folder='./EXPs/exp_antibody_3dose_2latent/irregular_10p_400d_50s_2latent_theta_F2_deltaAb_noise_pipeline/'
variance_folder = 'EXPs/antibody_irregular_deltaAB_variance_400d_10p_50s_'+str(b_sample_size)
estimated_params_file = 'results/antibody_poitwise_irregular_results.txt'

with open(estimated_params_file, "a", encoding="utf-8") as f:
    for i in range(1, 100):
        parameters = antibody_multidoses_train.main(i, model_folder=model_folder, data_folder=data_folder)
        estimated_parameters.append(parameters)
        f.write(parameters + "\n")
        f.flush()                 # push Python buffer to OS
        os.fsync(f.fileno()) 
        
        datas = np.load(data_folder+'dataset_'+str(i)+'.npy')
        timepoints = np.load(data_folder+'dataset_timepoints_'+str(i)+'.npy')
        variance_matrix = variance_estimation(datas, parameters, N_subjects, timepoints=timepoints, b_sample_size=b_sample_size, ni=10, T=400)
        Path(variance_folder).mkdir(parents=True, exist_ok=True)
        np.save(variance_folder+'/variance_matrix_'+str(i), variance_matrix)

#Analyse
pointwise_analyse_df = analyse_poitwise_estimation(estimated_params_file)

#Coverage
summary_df = coverage_from_estimated_variance(pointwise_analyse_df, estimated_params_file, variance_folder=variance_folder)
print(summary_df)



