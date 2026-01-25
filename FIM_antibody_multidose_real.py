# import os
# os.environ["JAX_PLATFORMS"] = "cpu"
import jax
from jax import numpy as jnp
import numpy as np

import diffrax
import jax.numpy as jnp
import pandas as pd
from real_ab_model import Antibody_ode

import warnings
warnings.filterwarnings('ignore')
jax.config.update("jax_enable_x64", True)

def log_likelihood(params, epsilons, padded_y, mask, padded_ts, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(params[3]) + jnp.exp(params[4])#delta_Ab
    # Ab_degrad_rate = 0.08
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = jnp.exp(params[3])

    # padded_y = y["padded_y"]
    # mask = y["mask"]
    # padded_ts = y["padded_ts"]
    # injection_times = y["injection_times"]

    term = diffrax.ODETerm(Antibody_ode())
    solver = diffrax.Tsit5()

    # === 1. UNPAD a single subject's data using the mask ===
    # Calculate the true number of time points for this subject
    true_length = jnp.sum(mask).astype(jnp.int32)
    
    # Get the real start and end times
    t0 = 0.0
    t1 = padded_ts[true_length - 1]


    y0 = jnp.array([0.01, 0.1])

    dt0 = 0.01
    saveat = diffrax.SaveAt(ts=padded_ts)

    b_i_theta = jnp.exp(params[5]) * epsilons[0]
    Ab_production_init_acc = jnp.exp(params[0]) * jnp.exp(b_i_theta)
    b_i_F2 = jnp.exp(params[6]) * epsilons[1]
    individual_F2 = jnp.exp(params[1]) * jnp.exp(b_i_F2)
    F3 = jnp.exp(params[2])

    # The full set of arguments for the vector field
    ode_args = (death_rate_S_cell, vaccine_autigen_decline_rate, Ab_production_init_acc, 
            individual_F2, F3, Ab_degrad_rate, injection_times)

    # Solve the ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8),
        args=ode_args,
        saveat=saveat,
        # max_steps = 100000
        adjoint = diffrax.ForwardMode()
    )

    processed_result = jnp.log10(sol.ys[:, 1:2]) # Or any other processing
    # Use the mask to zero-out the results for the padded time steps
    final_output = processed_result * mask[:, None] # Broadcast mask
    const = 2. * jnp.pi

    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * params[-1] - 0.5 * jnp.exp(-params[-1]) * (final_output - padded_y)**2
    masked_log_likelihood = jnp.sum(log_p_x_given_z * mask) 

    return masked_log_likelihood

def matrix_health(A):
    A = _symmetrize(A)
    eig = np.linalg.eigvalsh(A)
    eigmin, eigmax = eig.min(), eig.max()
    cond = eigmax / max(eigmin, 1e-300)
    return eigmin, eigmax, cond

def _symmetrize(A):
    return 0.5 * (A + A.T)

def _ridge_pd(A, eps=1e-8):
    """Make matrix safely invertible (very small ridge)."""
    A = _symmetrize(A)
    ridge = eps * np.trace(A) / A.shape[0]
    return A + ridge * np.eye(A.shape[0])

def variance_estimation(
    all_datas, model_params, N_subjects,
    hessian=False, sandwich=True,
    b_sample_size=2000, T=400, ni=10,
    eps_ridge=1e-8
):

    latent_shape = 2
    phi = jnp.array(model_params)

    J_list = []   # stores Ji = observed information (Hessian-based)
    K_list = []   # stores Ki = score outer product

    for i in range(N_subjects):

        epsilons = np.random.multivariate_normal(
            mean=np.zeros(latent_shape),
            cov=np.eye(latent_shape),
            size=b_sample_size
        )

        log_likelihoods = jax.vmap(log_likelihood, 
                                   in_axes=(None,0,None, None, None, None))(model_params, epsilons,  all_datas["padded_y"][i], all_datas["mask"][i], 
         all_datas["padded_ts"][i], all_datas["injection_times"][i])

        grad_log_likelihood_Y_b = jax.jacfwd(
            jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None)),
            argnums=0
        )(model_params, epsilons,  all_datas["padded_y"][i], all_datas["mask"][i], 
         all_datas["padded_ts"][i], all_datas["injection_times"][i])   # (S,P)

        # Stable weights
        w = jax.nn.softmax(log_likelihoods, axis=0)  # (S,)

        # Marginal score s_i = sum_s w_s * g_s
        score = jnp.einsum("s,sp->p", w, grad_log_likelihood_Y_b)  # (P,)

        # --- Meat contribution Ki = s_i s_i^T ---
        Ki = jnp.outer(score, score)
        K_list.append(np.array(Ki))

        if hessian:
            hess_log_likelihood_Y_b = jax.jacfwd(
                jax.jacfwd(
                    jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None)),
                    argnums=0
                ),
                argnums=0
            )(jnp.array(model_params), epsilons,  all_datas["padded_y"][i], all_datas["mask"][i], 
         all_datas["padded_ts"][i], all_datas["injection_times"][i])  # (S,P,P)

            g = grad_log_likelihood_Y_b
            H = hess_log_likelihood_Y_b

            EwH   = jnp.einsum("s,spq->pq", w, H)
            EwggT = jnp.einsum("s,sp,sq->pq", w, g, g)

            H_logp = EwH + EwggT - jnp.outer(score, score)  # Hessian of log p(y_i)
            Ji = -H_logp  # observed information for subject i

            J_list.append(np.array(Ji))

    # Aggregate
    K = np.sum(np.stack(K_list, axis=0), axis=0)
    K = _ridge_pd(K, eps=eps_ridge)

    if not hessian:
        # score-only covariance (OPG / BHHH)
        return np.linalg.inv(K)

    J = np.sum(np.stack(J_list, axis=0), axis=0)
    J = _ridge_pd(J, eps=eps_ridge)

    if sandwich:
        # --- Decide if we trust sandwich ---
        # thresholds 
        COND_MAX = 1e3
        MIN_EIG_MIN = 1e-10
        DIAG_RATIO_MAX = 1e3
        eigminJ, eigmaxJ, condJ = matrix_health(np.array(J))
        ABS_MAX_DIAG = 1 

        Jinv = np.linalg.inv(J)
        variance_matrix = Jinv @ K @ Jinv
        
        # # If J is bad -> fallback to score
        print(eigminJ, eigmaxJ, condJ)
        if (eigminJ <= MIN_EIG_MIN) or (condJ >= COND_MAX) or (not np.isfinite(condJ)):
            return np.linalg.inv(K)
        
        # Sandwich covariance
        Jinv = np.linalg.inv(J)
        variance_matrix = Jinv @ K @ Jinv
    else:
        # Hessian-only covariance
        Jinv = np.linalg.inv(J)
        variance_matrix = Jinv
        if np.any(np.diag(variance_matrix) < 0):
            return np.linalg.inv(K)

    return variance_matrix
    
# def variance_estimation(data, model_params, N_subjects, hessian=False, correct_param=True, b_sample_size=2000):    

#     observed_FIM_list = []
#     latent_shape = 2

#     for i in range(N_subjects):

#         standard_mean = jnp.zeros(latent_shape)
#         standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
#         epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=b_sample_size)
        
#         log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
        
#         grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None)))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
#         likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        
#         grad_marginal_Y = jnp.mean(jnp.einsum('ij,i ->ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
#         marginal_Y = jnp.mean(likelihood_Y) #monte carlo to approximate marginal likelihood of Y

#         if hessian:
#             hessian_log_likelihood_Y_b =  jax.jacfwd(jax.jacfwd(
#                 jax.vmap(log_likelihood, in_axes=(None, 0, None, None, None, None)))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i]))
#             # hessian_log_likelihood_Y_b =  jax.hessian(
#             #     jax.vmap(log_likelihood, in_axes=(None,0,None)))(jnp.array(model_params), epsilons,  all_datas[i,:])
            
#             pxz_hessian_part = hessian_log_likelihood_Y_b + np.expand_dims(grad_log_likelihood_Y_b, axis=2) * np.expand_dims(grad_log_likelihood_Y_b, axis=1)
            
#             pxz_hessian = np.einsum('ijk, i-> ijk' ,pxz_hessian_part, likelihood_Y) # hessian of P(Y|b)

#             hessian_marginal_Y = np.mean(pxz_hessian, axis=0)
        
#         # if correct_param:
#         #     correct_params = model_params - jnp.dot(inverse_H, score)
        
#         if not np.isclose(np.squeeze(marginal_Y) ** 2, 0):
#             if hessian:
#                 observed_information = - (hessian_marginal_Y * marginal_Y 
#                                           - jnp.expand_dims(grad_marginal_Y, axis=1) * jnp.expand_dims(grad_marginal_Y, axis=0)) / (marginal_Y ** 2)
#             else:
#                 observed_information = - (-jnp.expand_dims(grad_marginal_Y, axis=1) * jnp.expand_dims(grad_marginal_Y, axis=0)) / (marginal_Y ** 2)

#             observed_FIM_list.append(observed_information)

#     FIM = np.sum(np.array(observed_FIM_list), axis=0)
#     variance_matrix = np.linalg.inv(FIM)
    
#     return variance_matrix

    

    
    

    

