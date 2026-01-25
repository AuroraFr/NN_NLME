import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reordering and log-transform of estimated_values_2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 14,           # base font size
    "axes.titlesize": 14,      # title font size
    "axes.labelsize": 14,      # x/y label size
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 12,     # legend text
    "figure.titlesize": 16     # figure title
})

from real_ab_model import *

jax.config.update("jax_enable_x64", True)
latent_shape = 2
b_sample_size = 10000

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

    return jnp.log10(sol.ys)

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
    log_noise_var = np.log(VAE_params[-1]**2)
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
    fixed_effect_idx = jnp.array([0, 1, 2, 3, 4])
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
    import math
    n_rows = math.ceil(N_subjects / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False)
    axes = axes.flatten()
    ax0 = axes.flat[0]

    all_points = 0
    for i in range(N_subjects):
        ax = axes[i]
        if ax is not ax0:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both",
                       which="both",
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)

        true_len = int(mask[i].sum())
        all_points += true_len
        t = padded_ts[i, :true_len]
        y_hat_i = y_hat[i, :true_len]
        J_phi_i = J_phi[i, :true_len, :]
        J_b_i   = J_b[i, :true_len, :]
        y_real_i = 10 ** (padded_y[i, :true_len])

        sigma_b_i = jnp.diag(jnp.exp(post_logstd[i]) ** 2) #create a diagonal variance-covariance matrix 
        var_i = J_phi_i @ Sigma_phi_sub @ J_phi_i.T + J_b_i @ sigma_b_i @ J_b_i.T + jnp.exp(log_noise_var) * jnp.eye(true_len)

        sd_pred_i = jnp.sqrt(jnp.diag(var_i))
        ci_lower = y_hat_i - 1.96 * sd_pred_i
        ci_upper = y_hat_i + 1.96 * sd_pred_i
        nat_ci_lower = 10 ** ci_lower
        nat_ci_upper = 10 ** ci_upper
        
        ax.fill_between(t, nat_ci_lower, nat_ci_upper, color="black", alpha=0.3, label="95% PI")
        ax.plot(t, 10 ** y_hat_i, color="black", lw=2, label="Predicted mean")
        ax.scatter(t, y_real_i, color="navy", s=20, zorder=3, label="Observed")

        ax.grid(alpha=0.3)

    #individual prediction with PI plot: Figure 6 in the manuscript
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Time after first injections (Days)")
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
    epsilons = np.random.multivariate_normal(standard_mean, standard_cov, size=1000)

    VAE_fixed_effect_params = jnp.array(VAE_params[fixed_effect_idx])
    SAEM_fixed_effect_params = jnp.array(SAEM_params)
    SAEM_sd = jnp.array(SAEM_stds)

    # pop_mean = population_mean(VAE_fixed_effect_params, 500, jnp.array([second_dose, third_dose]))
    # J_phi = jax.jacfwd(population_mean)(VAE_fixed_effect_params, 500, [second_dose, third_dose])  #(500, 2, p)
    # population_mean_var = jnp.einsum("tkp,pq,tkq->tk", J_phi, Sigma_phi_sub, J_phi)
    # sd_population_mean = jnp.sqrt(jnp.diag(population_mean_var))
    # pred_lower = pop_mean - 1.96 * sd_population_mean
    # pred_upper = pop_mean + 1.96 * sd_population_mean

    pop_mean = population_mean(VAE_fixed_effect_params, 1000, jnp.array([second_dose, third_dose]))
    L = jnp.linalg.cholesky(Sigma_phi_sub + 1e-9 * jnp.eye(5))
    param_samples = VAE_fixed_effect_params + (L @ epsilons.T).T
    predictions = []
    for params in param_samples:
        prediction = population_mean(params, 500, jnp.array([second_dose, third_dose]))
        predictions.append(prediction)
    predictions = np.array(predictions).reshape(1000, 500,2)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)

    #SAEM: population mean with CI
    SAEM_pop_mean = population_mean(SAEM_fixed_effect_params, 500, jnp.array([second_dose, third_dose]))
    SAEM_param_samples = SAEM_fixed_effect_params + (jnp.diag(SAEM_sd) @ epsilons.T).T
    SAEM_predictions = []
    for params in SAEM_param_samples:
        SAEM_prediction = population_mean(params, 500, jnp.array([second_dose, third_dose]))
        SAEM_predictions.append(SAEM_prediction)
    SAEM_predictions = np.array(SAEM_predictions).reshape(1000, 500, 2)
    SAEM_pred_lower = np.percentile(SAEM_predictions, 2.5, axis=0)
    SAEM_pred_upper = np.percentile(SAEM_predictions, 97.5, axis=0)

    return 10**(SAEM_pop_mean), 10**(SAEM_pred_upper), 10**(SAEM_pred_lower), 10**(pop_mean), 10**(pred_lower), 10**(pred_upper)


latent_shape = 2
n_z = 1

seed = 42
key = jr.PRNGKey(seed)

estimated_values_ELBO_deltaAb= np.loadtxt('results/real_ab_ELBO_deltaAB_identifiability_results.txt')
estimated_values_SAEM_deltaAb = np.loadtxt('results/real_ab_SAEM_deltaAb_identifiability_results.txt')
init_values = np.loadtxt('antibody_datasets/real_ab_convergence_initparams.txt')
print(estimated_values_SAEM_deltaAb)

init_Ab_production_init_acc = jnp.log(30)  # theta
init_Ab_production_init_acc_logstd = jnp.log(0.8) # random effect std on theta
init_F2 = jnp.log(7)
init_F2_logstd = jnp.log(0.5)
init_F3 = jnp.log(18.0)
init_logvar_noise = jnp.log(0.4**2)
init_deltaS = -2.9645
init_deltaAB = -2.885
init_params = jnp.array([init_Ab_production_init_acc, init_F2, init_F3, init_Ab_production_init_acc_logstd, init_deltaS,init_deltaAB, init_F2_logstd,init_logvar_noise])

model = Antibody(latent_shape, init_params, key, n_z=n_z,timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=32, out_channel=32)
model_path = 'EXPs/exp_real_ab_identifiability/Antibody_real_'+str(3)+'.eqx'
model = eqx.tree_deserialise_leaves(model_path, model)
print(model)
print("loaded_model", np.sqrt(np.exp(model.logvar_noise)), model.theta)

# === Transform and Reorder ===
# Reorder columns: [0, 1, 2, 5, 6, 3, 4]
new_order = [0, 1, 2, 5, 6, 3, 4, 7]
init_values = init_values[:, new_order].copy()

init_values[:, 5] = np.exp(init_values[:, 5])  # omega_theta
init_values[:, 6] = np.exp(init_values[:, 6])
init_values[:, 7] = np.sqrt(np.exp(init_values[:, 7]))

latex_labels3 = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$", r"$log(\delta_S)$", r"$log(\delta_{Ab})$", 
    r"$\omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$",
     r"$\sigma_\epsilon$"
]

import pandas as pd
import real_ab_train 
from FIM_antibody_multidose_real import variance_estimation
df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
df["id"] = df["id"].astype(int)
df["Time"] = df["Time"].astype(float)
df["ED50_BAU"] = df["ED50_BAU"].astype(float)
df = df.dropna()
data = real_ab_train.preprocess_data(df)
print(estimated_values_ELBO_deltaAb)
params = np.array(estimated_values_ELBO_deltaAb[3, :])
print(params)
params[5:7] = np.log(params[5:7])
params[-1] = np.log(params[-1]**2)
print(params)
data = preprocess_data(df)
VAE_variance_matrix = variance_estimation(data, jnp.array(params), 25, b_sample_size=b_sample_size)
variances = np.diag(VAE_variance_matrix)
variances = np.array(variances, copy=True)
variances[5:7] = np.exp(2 * params[5:7]) * variances[5:7]
print(variances)
VAE_sd = np.sqrt(variances)
ci = 1.96 * VAE_sd
params[5:7] = np.exp(params[5:7])
print(params)
print(params - ci, params + ci)
SAEM_params = jnp.array([3.62, 1.35, 2.62, -4.72, -1.91]) 
SAEM_stds = jnp.array([0.19, 0.14, 0.12, 0.067, 0.23])

SAEM_preds, SAEM_upper, SAEM_lower, VAE_preds, VAE_upper, VAE_lower = prediction_interval(model, 
                                                                                          df, estimated_values_ELBO_deltaAb[3, :], 
                                                                                          VAE_variance_matrix, SAEM_params, SAEM_stds)
from matplotlib.gridspec import GridSpec
# -------------------------
# Build a 3-panel figure
# -------------------------
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])   # Panel A
ax2 = fig.add_subplot(gs[0, 1])   # Panel B
ax3 = fig.add_subplot(gs[1, :])   # Panel C (spans both columns)

times = np.linspace(0, 500, 500)

# -------------------------
# Panel A: S-cell
# -------------------------
ax1.plot(times, VAE_preds[:, 0], label="VAE", color="black")
ax1.plot(times, SAEM_preds[:, 0], label="SAEM", color="blue")
ax1.fill_between(times, VAE_lower[:, 0], VAE_upper[:, 0],
                 color="black", alpha=0.3, label="VAE 95% interval")
ax1.fill_between(times, SAEM_lower[:, 0], SAEM_upper[:, 0],
                 color="blue", alpha=0.3, label="SAEM 95% interval")
ax1.set_xlabel("Time after first injections (Days)")
ax1.set_ylabel("S-cell")

handles_ab, labels_ab = ax1.get_legend_handles_labels()

fig.legend(
    handles_ab, labels_ab,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

# -------------------------
# Panel B: Anti-S IgG
# -------------------------
ax2.plot(times, VAE_preds[:, 1], label="VAE", color="black")
ax2.plot(times, SAEM_preds[:, 1], label="SAEM", color="blue")
ax2.fill_between(times, VAE_lower[:, 1], VAE_upper[:, 1],
                 color="black", alpha=0.3, label="VAE 95% interval")
ax2.fill_between(times, SAEM_lower[:, 1], SAEM_upper[:, 1],
                 color="blue", alpha=0.3, label="SAEM 95% interval")
ax2.set_xlabel("Time after first injections (Days)")
ax2.set_ylabel("Anti-S IgG (BAU/ml)")

# -------------------------
# Panel C: parameter estimates scatter
# -------------------------
latex_labels3 = [
    r"$\log(\vartheta)$",
    r"$\log(\bar{f}_{M_2})$",
    r"$\log(\bar{f}_{M_3})$",
    r"$\log(\delta_S)$",
    r"$\log(\delta_{Ab})$",
    r"$\omega_{\vartheta}$",
    r"$\omega_{\bar{f}_{M_2}}$",
    r"$\sigma_\epsilon$"
]

offset = 0.05
for col_idx in range(len(latex_labels3)):
    for i in range(len(init_values)):
        # Init values
        ax3.plot(col_idx, init_values[i, col_idx],
                 marker="D", color="green", markersize=3, linestyle="None")

        # VAE estimates
        ax3.plot(col_idx + offset, estimated_values_ELBO_deltaAb[i, col_idx],
                 marker="o", color="black", markersize=5, linestyle="None")

        # SAEM estimates
        ax3.plot(col_idx - offset, estimated_values_SAEM_deltaAb[i, col_idx],
                 marker="X", color="red", markeredgewidth=0.4,
                 markersize=5, linestyle="None")

ax3.set_xticks(range(len(latex_labels3)))
ax3.set_xticklabels(latex_labels3, rotation=45, ha="right")
ax3.set_ylabel("Parameter Value")
ax3.grid(True)

# -------------------------
# Panel labels (A, B, C)
# -------------------------
ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")
ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")
ax3.text(-0.03, 1.02, "C", transform=ax3.transAxes, fontsize=14, fontweight="bold")

# -------------------------
# One shared legend for the whole figure
# -------------------------
legend_elements = [
    Line2D([0], [0], color='green', marker='D', linestyle='None', markersize=6, label='Init value'),
    Line2D([0], [0], color='red', marker='X', linestyle='None', markeredgewidth=0.4, label='SAEM estimations'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='VAE estimations')
]

fig.legend(handles=legend_elements, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.2, 1, 1])  # keep space for bottom legend
plt.savefig("figures/panel_3plots.pdf", bbox_inches="tight")

