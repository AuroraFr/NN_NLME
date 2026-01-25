#Pipeline to run VAE on irregular simulated antibody kinetic data
import antibody_multidoses_train_regular as antibody_multidoses_train_regular
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
data_folder = 'antibody_datasets/scipy_antibody_3dose_15_400_50/'
model_folder='./EXPs/exp_antibody_3dose_2latent/15p_400d_50s_2latent_theta_F2_deltaAb_noise_pipeline/'
variance_folder = 'EXPs/antibody_regular_deltaAB_variance_400d_15p_50s_'+str(b_sample_size)
estimated_params_file = 'results/antibody_poitwise_regular_results.txt'

with open(estimated_params_file, "a", encoding="utf-8") as f:
    for i in range(1, 3):
        parameters = antibody_multidoses_train_regular.main(i, model_folder=model_folder, data_folder=data_folder)
        estimated_parameters.append(parameters)
        f.write(parameters + "\n")
        f.flush()                 # push Python buffer to OS
        os.fsync(f.fileno()) 
        
        datas = np.load(data_folder+'dataset'+str(i)+'.npy')
        variance_matrix = variance_estimation(datas, parameters, N_subjects, b_sample_size=b_sample_size, ni=15, T=400)
        Path(variance_folder).mkdir(parents=True, exist_ok=True)
        np.save(variance_folder+'/variance_matrix_'+str(i), variance_matrix)

#Analyse
pointwise_analyse_df = analyse_poitwise_estimation(estimated_params_file)

#Coverage
summary_df = coverage_from_estimated_variance(pointwise_analyse_df, estimated_params_file, variance_folder=variance_folder)
print(summary_df)


