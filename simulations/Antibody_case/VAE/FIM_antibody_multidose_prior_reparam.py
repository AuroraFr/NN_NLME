import time
import jax
from jax import numpy as jnp
import numpy as np

import diffrax
from diffrax import Tsit5, diffeqsolve, ODETerm, SaveAt, PIDController
import jax.numpy as jnp
import pandas as pd
from scipy.stats import norm
from jax_networks import Antibody_3dose_ode

import warnings
warnings.filterwarnings('ignore')
jax.config.update("jax_enable_x64", True)

def log_likelihood(params, epsilons, y, times, ni, T):

        term = ODETerm(Antibody_3dose_ode())
        Ab_degrad_rate = jnp.exp(params[6])+jnp.exp(params[7]) # delta_Ab
        vaccine_autigen_decline_rate = 2.7  # delta_v
        death_rate_S_cell = jnp.exp(params[7])  # delta_s

        solver = Tsit5()

        t0 = 0
        t1 = T

        y0 = jnp.array([0.01, 0.1])

        dt0 = 0.01
        saveat = SaveAt(ts=times)

        b_i_theta = jnp.exp(params[3]) * epsilons[0]
        Ab_production_init_acc = jnp.exp(params[0]) * jnp.exp(b_i_theta)
        b_i_F2 = jnp.exp(params[4]) * epsilons[1]
        individual_F2 = jnp.exp(params[1]) * jnp.exp(b_i_F2)
        F3 = jnp.exp(params[2])

        # Solve the ODE
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller = PIDController(rtol=1e-8, atol=1e-8),
            max_steps = 10000,
            args=(death_rate_S_cell, vaccine_autigen_decline_rate, Ab_production_init_acc, individual_F2, F3, Ab_degrad_rate),
            saveat=saveat,
            adjoint=diffrax.ForwardMode()
        )

        noise_var = jnp.exp(params[5])
        const = 2. * jnp.pi
        output = jnp.log10(sol.ys)[:, 1]
        log_pxz = -0.5*ni*jnp.log(const) - 0.5 * ni *jnp.log(noise_var) - 0.5 *(1/noise_var) * jnp.sum((output - y)**2)

        return log_pxz

def variance_estimation(all_datas, model_params, N_subjects, times, hessian=True, b_sample_size=2000, T=400, ni=10, correct_param=False):

    observed_FIM_list = []
    num_params = len(model_params)
    all_datas = all_datas[:,:,1]
    latent_shape = 2

    for i in range(N_subjects):

        standard_mean = jnp.zeros(latent_shape)
        standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
        epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=b_sample_size)
        
        log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,0,None, None, None,None))(jnp.array(model_params), epsilons,  all_datas[i,:], times[i,:], ni, T)
        
        grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None,0,None, None,None,None)))(jnp.array(model_params), epsilons, all_datas[i,:], times[i,:], ni, T)

        likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        
        grad_marginal_Y = jnp.mean(jnp.einsum('ij,i ->ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
        marginal_Y = jnp.mean(likelihood_Y)

        # Stable weights w via softmax(loglik)
        w = jax.nn.softmax(log_likelihoods)  # (S,)

        # score s = grad log p(y_i)
        score = jnp.sum(w[:, None] * grad_log_likelihood_Y_b, axis=0)  # (P,)

        if hessian:
            hessian_log_likelihood_Y_b =  jax.jacfwd(jax.jacfwd(
                jax.vmap(log_likelihood, in_axes=(None,0,None,None))))(jnp.array(model_params), epsilons,  all_datas[i,:], times[i,:])
            pxz_hessian_part = hessian_log_likelihood_Y_b + np.expand_dims(grad_log_likelihood_Y_b, axis=2) * np.expand_dims(grad_log_likelihood_Y_b, axis=1)
            pxz_hessian = np.einsum('ijk, i-> ijk' ,pxz_hessian_part, likelihood_Y) # hessian of P(Y|b)
            hessian_marginal_Y = np.mean(pxz_hessian, axis=0)
        
        # if correct_param:
        # TODO
        #     correct_params = model_params - jnp.dot(inverse_H, score)
        
        if not np.isclose(np.squeeze(marginal_Y) ** 2, 0):
            if hessian:
                observed_information = - (hessian_marginal_Y * marginal_Y 
                                          - jnp.expand_dims(grad_marginal_Y, axis=1) * jnp.expand_dims(grad_marginal_Y, axis=0)) / (marginal_Y ** 2)
                observed_FIM_list.append(observed_information)
            
        if not hessian:
            observed_information = score[:, None] * score[None, :]
            observed_FIM_list.append(observed_information)

    FIM = np.sum(np.array(observed_FIM_list), axis=0)
    variance_matrix = np.linalg.inv(FIM)
    
    return variance_matrix

if __name__ == "__main__":

    from pathlib import Path
    timepoints = 10
    b_sample_size = 10000
    params = pd.read_csv('results/antibody_400d_10p_50s_irregular_deltaAb_noise_2latents.txt', sep=" ", header=None).values
    N_subjects = 50
    
    for i in range(1,2):
        data_file = 'antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_'+str(i)+'.npy'
        timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_timepoints_'+str(i)+'.npy')
        all_datas = np.load(data_file)

        variance_matrix = variance_estimation(all_datas, params[i-1, :], N_subjects, timepoints, hessian=False, b_sample_size=b_sample_size)

        result_folder = 'EXPs/antibody_3dose_irregular_deltaAB_nohessian_variance_400d_10p_50s_'+str(b_sample_size)
        
        Path(result_folder).mkdir(parents=True, exist_ok=True)
        np.save(result_folder+'/variance_matrix_'+str(i), variance_matrix)


    

    
    

    

