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
    "axes.labelsize": 16,      # x/y label size
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 16,     # legend text
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

def population_mean(fixed_effect_params, T, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(fixed_effect_params[4]) + jnp.exp(fixed_effect_params[3])
    vaccine_autigen_decline_rate = 2.7  #delta_v
    death_rate_S_cell = jnp.exp(fixed_effect_params[3])

    term = diffrax.ODETerm(Antibody_ode())

    solver = diffrax.Tsit5()
    
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

    return sol.ys


def predict_y(params, b_i, mask, padded_ts, injection_times):
    # Parameters
    Ab_degrad_rate = jnp.exp(params[4]) + jnp.exp(params[3]) #delta_Ab
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

def prediction_interval(model, df, VAE_params, VAE_variances, SAEM_params, SAEM_stds):

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

    _, _, b_i, post_logstd = jax.vmap(model,  axis_name="batch")(padded_y, mask, keys, padded_ts, injection_times)

    # Extract the 5×5 submatrix from the 8×8 covariance for fixed effect parameters
    fixed_effect_idx = jnp.array([0, 1, 2, 6, 7])
    Sigma_phi_sub = VAE_variances[jnp.ix_(fixed_effect_idx, fixed_effect_idx)]  # JAX-safe slicing
    VAE_fixed_effect_params = jnp.array(VAE_params[fixed_effect_idx])

    #delta method for individual prediction interval
    y_hat = jax.vmap(predict_y, in_axes=(None, 0, 0, 0, 0))(VAE_fixed_effect_params, b_i, data["mask"],data["padded_ts"], data["injection_times"])
    J_phi = jax.jacfwd(jax.vmap(predict_y, in_axes=(None, 0, 0, 0, 0)))(VAE_fixed_effect_params, b_i, data["mask"],data["padded_ts"], data["injection_times"])
    J_b   = jax.vmap(jax.jacfwd(predict_y, argnums=1), in_axes=(None, 0, 0, 0, 0))(VAE_fixed_effect_params, b_i, data["mask"],data["padded_ts"], data["injection_times"])

    J_phi = J_phi.squeeze(2)  # [B, T_max, n_params]
    J_b = J_b.squeeze(2)      # [B, T_max, n_b]
    y_hat = y_hat.squeeze(-1)
    n_cols = 5
    n_rows = math.ceil(N_subjects / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False)
    axes = axes.flatten()

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

        sigma_b_i = jnp.diag(jnp.exp(post_logstd[i]) ** 2) #create a diagonal variance-covariance matrix 
        var_i = J_phi_i @ Sigma_phi_sub @ J_phi_i.T + J_b_i @ sigma_b_i @ J_b_i.T + jnp.exp(model.logvar_noise) * jnp.eye(true_len)

        sd_pred_i = jnp.sqrt(jnp.diag(var_i))
        ci_lower = y_hat_i - 1.96 * sd_pred_i
        ci_upper = y_hat_i + 1.96 * sd_pred_i
        nat_ci_lower = 10 ** ci_lower
        nat_ci_upper = 10 ** ci_upper
        
        ax.fill_between(t, nat_ci_lower, nat_ci_upper, color="skyblue", alpha=0.4, label="95% PI")
        ax.plot(t, 10 ** y_hat_i, color="navy", lw=2, label="Predicted mean")
        ax.scatter(t, y_real_i, color="black", s=20, zorder=3, label="Observed")

        ax.set_title(f"Subject {i+1}")
        ax.grid(alpha=0.3)

    #individual prediction with PI plot: Figure 6 in the manuscript
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    axes[-1].set_xlabel("Time")
    fig.supylabel("Anti-S IgG (BAU/ml)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/individual_real_ab_PI.pdf')
    plt.close()

    #population mean with CI plot: Figure 7 in the manuscript
    second_dose = 27
    third_dose = 269
    standard_mean = jnp.zeros(5) # 5 parameters for fixed effect 
    standard_cov = jnp.eye(5)  # Identity covariance ensures independence
    epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=500)

    VAE_fixed_effect_params = jnp.array(VAE_params[fixed_effect_idx])
    SAEM_fixed_effect_params = jnp.array(SAEM_params)
    SAEM_sd = jnp.array(SAEM_stds)

    pop_mean = population_mean(VAE_fixed_effect_params, 500, jnp.array([second_dose, third_dose]))
    J_phi = jax.jacfwd(population_mean)(VAE_fixed_effect_params, 500, [second_dose, third_dose])  #(500, 2, p)
    population_mean_var = jnp.einsum("tkp,pq,tkq->tk", J_phi, Sigma_phi_sub, J_phi)
    sd_population_mean = jnp.sqrt(jnp.diag(population_mean_var))
    pred_lower = pop_mean - 1.96 * sd_population_mean
    pred_upper = pop_mean + 1.96 * sd_population_mean

    pop_mean = population_mean(VAE_fixed_effect_params, 500, jnp.array([second_dose, third_dose]))
    L = jnp.linalg.cholesky(Sigma_phi_sub + 1e-9 * jnp.eye(5))
    param_samples = VAE_fixed_effect_params + (L @ epsilons.T).T
    predictions = []
    for params in param_samples:
        prediction = population_mean(params, 500, jnp.array([second_dose, third_dose]))
        predictions.append(prediction)
    predictions = np.array(predictions).reshape(500, 500,2)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)

    #SAEM: population mean with CI
    SAEM_pop_mean = population_mean(SAEM_fixed_effect_params, 500, jnp.array([second_dose, third_dose]))
    SAEM_param_samples = SAEM_fixed_effect_params + (jnp.diag(SAEM_sd) @ epsilons.T).T
    SAEM_predictions = []
    for params in SAEM_param_samples:
        SAEM_prediction = population_mean(params, 500, jnp.array([second_dose, third_dose]))
        SAEM_predictions.append(SAEM_prediction)
    SAEM_predictions = np.array(SAEM_predictions).reshape(500, 500, 2)
    SAEM_pred_lower = np.percentile(SAEM_predictions, 2.5, axis=0)
    SAEM_pred_upper = np.percentile(SAEM_predictions, 97.5, axis=0)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
    times = np.linspace(0, 500, 500)

    ax1.plot(times, pop_mean[:,0], label='VAE', color='black')
    ax1.plot(times, SAEM_pop_mean[:,0], label='SAEM', color='blue')
    ax1.fill_between(times, pred_lower[:,0], pred_upper[:,0], color='black', alpha=0.3, label='VAE 95% interval')
    ax1.fill_between(times, SAEM_pred_lower[:,0], SAEM_pred_upper[:,0], color='blue', alpha=0.3, label='SAEM 95% interval')
    ax1.set_xlabel('Time after first injections (Days)')
    ax1.set_ylabel('S-cell')

    ax2.plot(times, pop_mean[:,1], label='VAE', color='black')
    ax2.plot(times, SAEM_pop_mean[:,1], label='SAEM', color='blue')
    ax2.fill_between(times, pred_lower[:,1], pred_upper[:,1], color='black', alpha=0.3, label='VAE 95% interval')
    ax2.fill_between(times, SAEM_pred_lower[:,1], SAEM_pred_upper[:,1], color='blue', alpha=0.3, label='SAEM 95% interval')
    ax2.set_xlabel('Time after first injections (Days)')
    ax2.set_ylabel('Anti-S IgG (BAU/ml)')
    
    plt.legend()
    plt.savefig('figures/mean_real_ab_PI_S_AB.pdf')