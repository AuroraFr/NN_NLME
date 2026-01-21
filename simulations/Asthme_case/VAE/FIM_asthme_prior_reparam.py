import time
import jax
from jax import numpy as jnp
import numpy as np

import diffrax
from diffrax import Tsit5, diffeqsolve, ODETerm, SaveAt, PIDController
import jax.numpy as jnp
import pandas as pd
from scipy.stats import norm
from asthme_model import asthme_vector_field

import warnings
warnings.filterwarnings('ignore')
jax.config.update("jax_enable_x64", True)

# matplotlib.use('TKAgg')
def log_likelihood(params, epsilons, y, timepoints=20, T=400):
    term = ODETerm(asthme_vector_field())

    b_i_Kp = jnp.exp(params[1]) * epsilons
    individual_Kp = jnp.exp(params[0]) * jnp.exp(b_i_Kp)

    ODE_params = {
            "k_p": individual_Kp,
            "p_max": 1.0,
            "k_ap": 1.0,
            "eta_ap": 1.0,
            "k_cp": 0.01,
            "gamma_c": 1.0,
            "gamma_p": 1.0,
            "k_pc": 1.0,
            "phi_c": 0.1,
            "k_s": 0.2,
            "nu": 30.0,
            "t_i": list(range(50, 250, 40)),
            "k_ac": 0.01,
            "eta_ac": 1.0,
            "gamma_a": 0.01,
            "k_b": 1.0,
            "k_pm": 0.1,
            "k_apm": 0.1,
            "eta_apm": 10.0,
            "phi_m": 0.01
        }

    solver = diffrax.Kvaerno5()

    t0 = 0
    t1 = T

    times = jnp.linspace(t0, t1, timepoints)
    # times = times / self.time_scale

    y0 = jnp.array([0.1, 0.8, 0.01, 0.9])

    dt0 = 0.01
    saveat = diffrax.SaveAt(ts=times)

    # Solve the ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6),
        args=ODE_params,
        saveat=saveat,
        # max_steps = 100000,
        adjoint = diffrax.ForwardMode()
    )

    noise_var = 0.1 ** 2
    const = 2. * jnp.pi
    output = sol.ys[:, 2]
    epsilon = 1e-8
    Y_hat = jnp.log10(jax.nn.relu(output) + epsilon)
    log_pxz = -0.5*timepoints*jnp.log(const) - 0.5 * timepoints *jnp.log(noise_var) - 0.5 *(1/noise_var) * jnp.sum((Y_hat - y)**2)

    return log_pxz

def variance_estimation(all_datas, model_params, N_subjects, hessian=True, correct_param=True, b_sample_size=2000, T=400, timepoints=20):
    observed_FIM_list = []
    num_params = len(model_params)
    latent_shape = 1

    for i in range(N_subjects):
    
        standard_mean = jnp.zeros(latent_shape)
        standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
        # epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=b_sample_size)
        epsilons = np.random.normal(size=b_sample_size)
        
        log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,0,None))(jnp.array(model_params), epsilons,  all_datas[i,:])
        
        grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None,0,None)))(jnp.array(model_params), epsilons,  all_datas[i,:])
        likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        
        grad_marginal_Y = jnp.mean(jnp.einsum('ij,i-> ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
        marginal_Y = jnp.mean(likelihood_Y) #monte carlo to approximate marginal likelihood of Y

        if hessian:
            hessian_log_likelihood_Y_b =  jax.jacfwd(jax.jacfwd(
                jax.vmap(log_likelihood, in_axes=(None,0,None))))(jnp.array(model_params), epsilons,  all_datas[i,:])
            # hessian_log_likelihood_Y_b =  jax.hessian(
            #     jax.vmap(log_likelihood, in_axes=(None,0,None)))(jnp.array(model_params), epsilons,  all_datas[i,:])
            
            pxz_hessian_part = hessian_log_likelihood_Y_b + np.expand_dims(grad_log_likelihood_Y_b, axis=2) * np.expand_dims(grad_log_likelihood_Y_b, axis=1)
            
            pxz_hessian = np.einsum('ijk,i-> ijk' ,pxz_hessian_part, likelihood_Y) # hessian of P(Y|b)

            hessian_marginal_Y = np.mean(pxz_hessian, axis=0)
        
        # if correct_param:
        #     correct_params = model_params - jnp.dot(inverse_H, score)
        
        if not np.isclose(np.squeeze(marginal_Y) ** 2, 0):
            if hessian:
                observed_information = - (hessian_marginal_Y * marginal_Y 
                                        - jnp.expand_dims(grad_marginal_Y, axis=1) * jnp.expand_dims(grad_marginal_Y, axis=0)) / (marginal_Y ** 2)
            else:
                observed_information = - (-jnp.expand_dims(grad_marginal_Y, axis=1) * jnp.expand_dims(grad_marginal_Y, axis=0)) / (marginal_Y ** 2)

            observed_FIM_list.append(observed_information)

    
    FIM = np.sum(np.array(observed_FIM_list), axis=0)

    return FIM

if __name__ == "__main__":

    from pathlib import Path
    timepoints = 20
    b_sample_size = 5000
    params = pd.read_csv('results/asthme_1latent_Kp_50s_kp.txt', sep=" ", header=None).values
    N_subjects = 50
    
    for i in range(1,100):
        try:
            data_file = 'asthme_datasets/20_400_50_1latent_Kp/dataset_'+str(i)+'.npy'
            all_datas = np.load(data_file)
            O_FIM = variance_estimation(all_datas, jnp.log(params[i-1, :]), N_subjects, hessian=False, b_sample_size=b_sample_size, timepoints=timepoints)
            variance_matrix = np.linalg.inv(O_FIM)
            result_folder = 'EXPs/asthme_nohessian_kp_50s_20d_'+str(b_sample_size)
            
            Path(result_folder).mkdir(parents=True, exist_ok=True)
            np.save(result_folder+'/variance_matrix_'+str(i), variance_matrix)
        except:
            print(O_FIM, i)
