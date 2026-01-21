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

def preprocess_data(df):
    """
    Pads all sequence data to a uniform length for use with jax.lax.scan or jax.vmap.
    """
    grouped = df.groupby("id")
    T_max = grouped.size().max()

    padded_y_list, padded_ts_list, mask_list, injection_times_list = [], [], [], []

    for subject_id, group in grouped:
        group = group.sort_values("Time")
        ts = group["Time"].values

        y = group["ED50_BAU"].values
        second_inj = group.iloc[0]["SecondInjTime"]
        third_inj = group.iloc[0]["ThirdInjTime"]

        Ti = len(ts)
        pad_len = T_max - Ti

        # Pad all sequence data to T_max. Use float for the mask.
        y_padded = np.pad(y.reshape(-1, 1), ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        last_valid_time = ts[-1]
        ts_padded = np.pad(ts, (0, pad_len), 'constant', constant_values=last_valid_time)
        mask = np.array([1.0] * Ti + [0.0] * pad_len)

        padded_y_list.append(y_padded)
        padded_ts_list.append(ts_padded)
        mask_list.append(mask)
        injection_times_list.append(np.array([second_inj, third_inj]))

    # Stack into JAX arrays 
    return {
        "padded_y": jnp.stack(padded_y_list),
        "padded_ts": jnp.stack(padded_ts_list),
        "mask": jnp.stack(mask_list),
        "injection_times": jnp.stack(injection_times_list),
    }


def log_likelihood(params, epsilons, padded_y, mask, padded_ts, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(params[7]) #delta_Ab
    # Ab_degrad_rate = 0.08
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = jnp.exp(params[6])

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

    b_i_theta = jnp.exp(params[3]) * epsilons[0]
    Ab_production_init_acc = jnp.exp(params[0]) * jnp.exp(b_i_theta)
    b_i_F2 = jnp.exp(params[4]) * epsilons[1]
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

    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * params[5] - 0.5 * jnp.exp(-params[5]) * (final_output - padded_y)**2
    masked_log_likelihood = jnp.sum(log_p_x_given_z * mask) 

    return masked_log_likelihood
    
def variance_estimation(data, model_params, N_subjects, hessian=True, correct_param=True, b_sample_size=2000, T=400, timepoints=17):    
    observed_FIM_list = []
    num_params = len(model_params)
    latent_shape = 2

    for i in range(N_subjects):

        standard_mean = jnp.zeros(latent_shape)
        standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
        epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=b_sample_size)
        
        log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
        
        grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None)))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
        likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        
        grad_marginal_Y = jnp.mean(jnp.einsum('ij,i ->ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
        marginal_Y = jnp.mean(likelihood_Y) #monte carlo to approximate marginal likelihood of Y

        if hessian:
            hessian_log_likelihood_Y_b =  jax.jacfwd(jax.jacfwd(
                jax.vmap(log_likelihood, in_axes=(None,0,None, None, None, None)))(jnp.array(model_params), epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i]))
            # hessian_log_likelihood_Y_b =  jax.hessian(
            #     jax.vmap(log_likelihood, in_axes=(None,0,None)))(jnp.array(model_params), epsilons,  all_datas[i,:])
            
            pxz_hessian_part = hessian_log_likelihood_Y_b + np.expand_dims(grad_log_likelihood_Y_b, axis=2) * np.expand_dims(grad_log_likelihood_Y_b, axis=1)
            
            pxz_hessian = np.einsum('ijk, i-> ijk' ,pxz_hessian_part, likelihood_Y) # hessian of P(Y|b)

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
    variance_matrix = np.linalg.inv(FIM)
    
    return variance_matrix

if __name__ == "__main__":
    
    from pathlib import Path
    timepoints = 17
    b_sample_size = 10000
    N_subjects = 25

    df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
    # ---- Clean + ensure correct types ----
    df["id"] = df["id"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["ED50_BAU"] = df["ED50_BAU"].astype(float)
    df = df.dropna()
    data = preprocess_data(df)

    # model_params = jnp.array([jnp.log(24.5), jnp.log(7.1), jnp.log(18.5), jnp.log(0.5), jnp.log(0.9)])
    variance_matrix = variance_estimation(data, jnp.array([3.65, 1.36, 2.51, jnp.log(0.47), jnp.log(0.05), jnp.log(0.22**2), -4.72, -1.93]), N_subjects, hessian=False, b_sample_size=b_sample_size)

    result_folder = 'EXPs/real_ab_nohessian_'+str(b_sample_size)
    
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    np.save(result_folder+'/variance_matrix_scenario', variance_matrix)
    print(np.diag(variance_matrix))

    

    
    

    

