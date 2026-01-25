import time
import jax
from jax import numpy as jnp
import numpy as np

import diffrax
from diffrax import Tsit5, diffeqsolve, ODETerm, SaveAt, PIDController
import jax.numpy as jnp
import pandas as pd
from antibody_multidoses_model import Antibody_ode

import warnings
warnings.filterwarnings('ignore')
jax.config.update("jax_enable_x64", True)

def log_likelihood(params, epsilons, y, times, ni, T):

        term = ODETerm(Antibody_ode())
        Ab_degrad_rate = jnp.exp(params[6])+jnp.exp(params[7]) # delta_Ab
        vaccine_autigen_decline_rate = 2.7  # delta_v
        death_rate_S_cell = jnp.exp(params[6])  # delta_s

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
    all_datas, model_params, N_subjects, timepoints=None,
    hessian=False, sandwich=False,
    b_sample_size=2000, T=400, ni=10,
    eps_ridge=1e-8
):
    all_datas = all_datas[:, :, 1]  #only antibody data
    latent_shape = 2
    phi = jnp.array(model_params)

    J_list = []   # stores Ji = observed information (Hessian-based)
    K_list = []   # stores Ki = score outer product

    for i in range(N_subjects):
        if timepoints is None:
            times = jnp.linspace(0, T, ni)
        else:
            times = timepoints[i,:]

        epsilons = np.random.multivariate_normal(
            mean=np.zeros(latent_shape),
            cov=np.eye(latent_shape),
            size=b_sample_size
        )

        log_likelihoods = jax.vmap(
            log_likelihood, in_axes=(None, 0, None, None, None, None)
        )(phi, epsilons, all_datas[i, :], times, ni, T)

        grad_log_likelihood_Y_b = jax.jacfwd(
            jax.vmap(log_likelihood, in_axes=(None, 0, None, None, None, None)),
            argnums=0
        )(phi, epsilons, all_datas[i, :], times, ni, T)   # (S,P)

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
                    jax.vmap(log_likelihood, in_axes=(None, 0, None, None, None, None)),
                    argnums=0
                ),
                argnums=0
            )(phi, epsilons, all_datas[i, :], times, ni, T)  # (S,P,P)

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


if __name__ == "__main__":

    from pathlib import Path
    timepoints = 10
    b_sample_size = 8000
    params = pd.read_csv('results/antibody_400d_10p_50s_irregular_deltaAb_noise_2latents.txt', sep=" ", header=None).values
    N_subjects = 50
    
    for i in range(1, 100):
        data_file = 'antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_'+str(i)+'.npy'
        timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_timepoints_'+str(i)+'.npy')
        all_datas = np.load(data_file)

        variance_matrix = variance_estimation(all_datas, params[i-1, :], N_subjects, timepoints=timepoints, b_sample_size=b_sample_size)

        result_folder = 'EXPs/antibody_3dose_irregular_deltaAB_nohessian_variance_400d_10p_50s_'+str(b_sample_size)
        
        Path(result_folder).mkdir(parents=True, exist_ok=True)
        np.save(result_folder+'/variance_matrix_'+str(i), variance_matrix)


    

    
    

    

