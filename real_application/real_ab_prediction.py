# import os
# os.environ['JAX_PLATFORMS']='cpu'
from real_ab_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.rcParams.update({
    "font.size": 14,           # base font size
    "axes.titlesize": 16,      # title font size
    "axes.labelsize": 14,      # x/y label size
    "xtick.labelsize": 13,     # x tick labels
    "ytick.labelsize": 13,     # y tick labels
    "legend.fontsize": 12,     # legend text
    "figure.titlesize": 18     # figure title
})

jax.config.update("jax_enable_x64", True)
latent_shape = 2
b_sample_size = 5000

def preprocess_data(df):
    """
    Pads all sequence data to a unif,rm length for use with jax.lax.scan or jax.vmap.
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

def log_likelihood(params, log_noise_var, epsilons, padded_y, mask, padded_ts, injection_times):
    # Parameters
    # Ab_degrad_rate = jnp.exp(params[7]) #delta_Ab
    Ab_degrad_rate = 0.08
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = 0.01 #jnp.exp(params[6])

    # padded_y = y["padded_y"]
    # mask = y["mask"]
    # padded_ts = y["padded_ts"]
    # injection_times = y["injection_times"]

    term = diffrax.ODETerm(Antibody_ode())
    solver = diffrax.Tsit5()

    # === 1. UNPAD a single subject's data using the mask ===
    # Calculate the true number of time points for this subject
    true_length = jnp.sum(mask).astype(jnp.int32)
    print(true_length)
    
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
    print(final_output.shape)
    const = 2. * jnp.pi

    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * log_noise_var - 0.5 * jnp.exp(-log_noise_var) * (final_output - padded_y)**2
    masked_log_likelihood = jnp.sum(log_p_x_given_z * mask) 

    return masked_log_likelihood

def variance_estimation(data, model_params, N_subjects, noise_variance, hessian=True, correct_param=True, b_sample_size=2000, T=400, timepoints=17):    
    observed_FIM_list = []
    num_params = len(model_params)
    latent_shape = 2

    for i in range(N_subjects):

        standard_mean = jnp.zeros(latent_shape)
        standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
        epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=b_sample_size)
        
        log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,None, 0,None, None, None, None))(jnp.array(model_params), noise_variance, epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
        
        grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None, None, 0,None, None, None, None)))(jnp.array(model_params), noise_variance, epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i])
        likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        
        grad_marginal_Y = jnp.mean(jnp.einsum('ij,i ->ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
        marginal_Y = jnp.mean(likelihood_Y) #monte carlo to approximate marginal likelihood of Y

        if hessian:
            hessian_log_likelihood_Y_b =  jax.jacfwd(jax.jacfwd(
                jax.vmap(log_likelihood, in_axes=(None, None, 0,None, None, None, None)))(jnp.array(model_params), noise_variance, epsilons,  data["padded_y"][i], data["mask"][i],data["padded_ts"][i], data["injection_times"][i]))
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

def population_mean(fixed_effect_params, T, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(fixed_effect_params[4])
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = jnp.exp(fixed_effect_params[3])

    term = diffrax.ODETerm(Antibody_ode())

    solver = diffrax.Tsit5()

    # === 1. UNPAD a single subject's data using the mask ===
    # Calculate the true number of time points for this subject
    true_length = jnp.sum(mask).astype(jnp.int32)
    
    # Get the real start and end times
    t0 = 0.0
    t1 = T
    ts = jnp.linspace(t0, t1, 500)


    y0 = jnp.array([0.01, 0.1])

    dt0 = 0.01
    saveat = diffrax.SaveAt(ts=ts)

    theta = jnp.exp(fixed_effect_params[0])
    F2 = jnp.exp(fixed_effect_params[1])
    F3 = jnp.exp(fixed_effect_params[2])

    # The full set of arguments for the vector field
    ode_args = (death_rate_S_cell, vaccine_autigen_decline_rate, theta, F2, F3, Ab_degrad_rate, injection_times)

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

    result = jnp.log10(sol.ys[:, 0:1]) # Or any other processing

    return result


def predict_y(params, b_i, mask, padded_ts, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(params[4]) #delta_Ab
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = jnp.exp(params[3])

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

    b_i_theta = b_i[0]
    Ab_production_init_acc = jnp.exp(params[0]) * jnp.exp(b_i_theta)
    b_i_F2 = b_i[1]
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

    return final_output

# ---- Load your CSV file ----
df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
# ---- Clean + ensure correct types ----
df["id"] = df["id"].astype(int)
df["Time"] = df["Time"].astype(float)
df["ED50_BAU"] = df["ED50_BAU"].astype(float)
df = df.dropna()

model_folder = './EXPs/exp_real_ab/'

###########Scenario 1 parameter values###########
theta = jnp.exp(3.65)
F2 = jnp.exp(1.36)
F3 = jnp.exp(2.50)
noise_std = jnp.log(0.22 ** 2)
scenario1_params = jnp.array([theta, F2, F3, F3, F3, F3,F3])
########################################

data = preprocess_data(df)
padded_y = data["padded_y"]
mask = data["mask"]
padded_ts = data["padded_ts"]
injection_times = data["injection_times"]

seed = 42
key = jr.PRNGKey(seed)
key, _ =  jr.split(key, 2)
N_subjects, T_max, _ = padded_y.shape
keys = jr.split(key, N_subjects)

model = Antibody(latent_shape, scenario1_params, key, n_z=50, timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=16, out_channel=16)

model = eqx.tree_deserialise_leaves("EXPs/exp_real_ab/Antibody_real_test.eqx", model)
params = jnp.array([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB])

_, X, post_mean, post_logstd = jax.vmap(model,  axis_name="batch")(padded_y, mask, keys, padded_ts, injection_times)
# Sigma_phi = variance_estimation(data, scenario1_params, N_subjects, noise_variance=model.logvar_noise, hessian=False, b_sample_size=b_sample_size)

# from pathlib import Path
result_folder = 'EXPs/real_ab_nohessian_10000'

Sigma_phi = np.load(result_folder+'/variance_matrix_scenario.npy')
idx = jnp.array([0, 1, 2, 6, 7])
# 2. Extract the 5×5 submatrix from the 8×8 covariance
Sigma_phi_sub = Sigma_phi[jnp.ix_(idx, idx)]  # JAX-safe slicing

y_hat = jax.vmap(predict_y, in_axes=(None, 0, 0, 0, 0))(params, post_mean, data["mask"],data["padded_ts"], data["injection_times"])
J_phi = jax.jacfwd(jax.vmap(predict_y, in_axes=(None, 0, 0, 0, 0)))(params, post_mean, data["mask"],data["padded_ts"], data["injection_times"])
J_b   = jax.vmap(jax.jacfwd(predict_y, argnums=1), in_axes=(None, 0, 0, 0, 0))(params, post_mean, data["mask"],data["padded_ts"], data["injection_times"])
Sigma_b_i = jnp.exp(post_logstd) ** 2

J_phi = J_phi.squeeze(2)  # [B, T_max, n_params]
J_b = J_b.squeeze(2)      # [B, T_max, n_b]
y_hat = y_hat.squeeze(-1)
n_cols = 5
n_rows = math.ceil(N_subjects / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False)
axes = axes.flatten()

pred_variance = []
all_points = 0
for i in range(N_subjects):
    ax = axes[i]
    true_len = int(mask[i].sum())
    all_points += true_len
    t = padded_ts[i, :true_len]
    y_hat_i = y_hat[i, :true_len]
    J_phi_i = J_phi[i, :true_len, :]
    J_b_i   = J_b[i, :true_len, :]
    y_real_i = 10 ** (padded_y[i, :true_len])

    sigma_b_i = jnp.diag(jnp.exp(post_logstd[i]) ** 2)
    var_i = (J_phi_i @ Sigma_phi_sub @ J_phi_i.T
           + J_b_i   @ sigma_b_i     @ J_b_i.T
           + jnp.exp(model.logvar_noise) * jnp.eye(true_len))

    sd_pred_i = jnp.sqrt(jnp.diag(var_i))
    ci_lower = y_hat_i - 1.96 * sd_pred_i
    ci_upper = y_hat_i + 1.96 * sd_pred_i
    nat_ci_lower = 10 ** ci_lower
    nat_ci_upper = 10 ** ci_upper
    
    ax.fill_between(t, nat_ci_lower, nat_ci_upper, color="skyblue", alpha=0.4, label="95% PI")
    ax.plot(t, 10 ** y_hat_i, color="navy", lw=2, label="Predicted mean")
    ax.scatter(t, y_real_i, color="black", s=20, zorder=3, label="Observed")

    ax.set_title(f"Subject {i+1}")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Antibody (log10 scale)")
    ax.grid(alpha=0.3)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

axes[-1].set_xlabel("Time")
fig.supylabel("Anti-S IgG (BAU/ml)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/individual_real_ab_PI.pdf')
plt.close()

standard_mean = jnp.zeros(5)
standard_cov = jnp.eye(5)  # Identity covariance ensures independence
epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=500)
fixed_effect_params = jnp.array([3.65, 1.36, 2.50, -4.72, -1.93])
fixed_effect_params_SAEM = jnp.array([3.62, 1.35, 2.62, -4.72, -1.97])
sd_SAEM = jnp.array([0.19, 0.14, 0.12, 0.067, 0.23])
# 1. Select the 5 parameters of interest
idx = jnp.array([0, 1, 2, 6, 7])

### VAE prediction interval ###
# 2. Extract the 5×5 submatrix from the 8×8 covariance
Sigma_phi_sub = Sigma_phi[jnp.ix_(idx, idx)]  # JAX-safe slicing
sub_sigma_diag = jnp.diag(Sigma_phi)[idx]
# 3. Compute Cholesky factor for correlated sampling
L = jnp.linalg.cholesky(Sigma_phi_sub)
print(jnp.diag(sub_sigma_diag))
pop_mean = population_mean(fixed_effect_params, 500, jnp.array([27, 269]))
pop_mean =  jnp.squeeze(pop_mean, axis=1)
J_phi = jax.jacfwd(population_mean)(fixed_effect_params, 500, [27, 269])
J_phi = jnp.squeeze(J_phi, axis=1)
print('J_phi', J_phi.shape)
population_mean_var = J_phi @ Sigma_phi_sub @ J_phi.T
sd_population_mean = jnp.sqrt(jnp.diag(population_mean_var))
pred_lower = pop_mean - 1.96 * sd_population_mean
pred_upper = pop_mean + 1.96 * sd_population_mean
print(pred_upper.shape, sd_population_mean.shape)

#Prediction interval SAEM
SAEM_pop_mean = population_mean(fixed_effect_params_SAEM, 500, jnp.array([27, 269]))
SAEM_pop_mean =  jnp.squeeze(SAEM_pop_mean, axis=1)
param_samples = fixed_effect_params_SAEM + (jnp.diag(sd_SAEM) @ epsilons.T).T
predictions = []
for params in param_samples:
    prediction = population_mean(params, 500, jnp.array([27, 269]))
    predictions.append(prediction)
predictions = np.array(predictions).reshape(500, 500)
SAEM_predictions_std = np.sqrt(np.var(predictions, axis=0))
SAEM_pred_lower = np.percentile(predictions, 2.5, axis=0)
SAEM_pred_upper = np.percentile(predictions, 97.5, axis=0)

# ---- Plot ----
times = np.linspace(0, 500,500)
plt.plot(times, 10**pop_mean, label='VAE', color='black')
plt.plot(times, 10**SAEM_pop_mean, label='SAEM', color='blue')
plt.fill_between(times, 10**pred_lower, 10**pred_upper, color='black', alpha=0.3, label='VAE 95% interval')
plt.fill_between(times, 10**SAEM_pred_lower, 10**SAEM_pred_upper, color='blue', alpha=0.3, label='SAEM 95% interval')
plt.xlabel('Time after first injections (Days)')
plt.ylabel('S-cell')
plt.legend()

plt.savefig('figures/mean_real_ab_PI_S.pdf')