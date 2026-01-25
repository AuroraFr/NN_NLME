import os
# os.environ["JAX_PLATFORMS"] = "cpu"
from real_ab_model import Antibody
import optax
import numpy as np
import time
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from pathlib import Path

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
        if y.shape[0] < 5:
            continue
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

def get_kl_weight(epoch, warmup_epochs=100):
    return jnp.minimum(1.0, epoch / warmup_epochs)

@eqx.filter_jit
def loss(model, y, key, n_z, beta=1):
    """
    Computes the total loss for a batch of subjects using jax.vmap for parallelization.
    """
    # Define inputs from the data dictionary `y`
    padded_y = y["padded_y"]
    mask = y["mask"]
    padded_ts = y["padded_ts"]
    injection_times = y["injection_times"]
    
    B, T_max, _ = padded_y.shape
    keys = jr.split(key, B)

    const = 2.0 * jnp.pi
    klloss, X, post_mean, post_logstd = jax.vmap(model,  axis_name="batch")(padded_y, mask, keys, padded_ts, injection_times)

    # Sum the KL losses for each dimension over the batch
    kl_loss_vector = jnp.sum(klloss, axis=0) # Resulting shape: (latent_shape,)

    kl_loss = jnp.sum(kl_loss_vector)

    X_reshaped = X.reshape(-1, 1)
    targets_reshaped = jnp.repeat(padded_y, repeats=n_z, axis=0).reshape(-1,1)
    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * model.logvar_noise - 0.5 * jnp.exp(-model.logvar_noise) * (X_reshaped - targets_reshaped)**2
    log_p_x_given_z_structured = log_p_x_given_z.reshape(B, n_z, T_max)

    log_p_x_given_z_mean = jnp.mean(log_p_x_given_z_structured, axis=1)

    masked_log_likelihood = log_p_x_given_z_mean * mask 
    reconstruction_loss = jnp.sum(masked_log_likelihood)

    # Final negative ELBO for minimization
    return -(reconstruction_loss - beta * kl_loss),  (kl_loss, post_mean.mean(axis=0), (jnp.exp(post_logstd).mean(axis=0)))

@eqx.filter_jit
def make_step(model, opt_state, y, key, n_z, optim, beta=1):
    
    params_old = eqx.filter(model, eqx.is_inexact_array)
    (loss_value, (kl_loss, post_mean, post_std)), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, y, key, n_z, beta=beta)               
    
    updates, opt_state = optim.update(grads, opt_state, params_old)
    
    # Apply the noisy update
    params_new = optax.apply_updates(params_old, updates)
    model = eqx.apply_updates(model, updates)
    
    key_i = jr.split(key, 2)[0]

    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    delta_params = jax.tree_util.tree_map(lambda a, b: a - b, params_new, params_old)
    delta_norm = jnp.sqrt(sum(jnp.sum(d**2) for d in jax.tree_util.tree_leaves(delta_params)))
    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params_old)))
    ratio = delta_norm / (param_norm + 1e-8)

    return model, loss_value, grad_norm, ratio, opt_state, key_i, kl_loss, post_mean, post_std

def main(data, model_folder="EXPs/exp_real_ab/", model_name="Antibody_real.eqx", steps=25000):

    #Training
    learning_rate = 0.005
    latent_shape = 2 #latent variables dim.
    n_z = 50
    full_stop_criteria = True

    seed = 0
    key = jr.PRNGKey(seed)
    train_key, _ =  jr.split(key, 2)
    eps_grad_norm = 1e-4
    eps_delta_norm = 1e-6


    model_folder = 'EXPs/exp_real_ab/'
    result_folder = 'result_summaries/real_ab/'

    init_theta = 4  # theta
    init_theta_logstd = jnp.log(0.8) # random effect std on theta
    init_F2 = 2
    init_F2_logstd = jnp.log(0.95)
    init_F3 = 3.5
    init_logvar_noise = jnp.log(0.4**2)
    init_deltaS = -3.88 #
    init_deltaAB = -2.34
    init_params = jnp.array([init_theta, init_F2, init_F3, init_theta_logstd, init_F2_logstd, init_logvar_noise, init_deltaS, init_deltaAB])

    model = Antibody(latent_shape, init_params, train_key, n_z=n_z, timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=32, out_channel=32)
    optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optim.init(params)

    best_loss = jnp.inf
    patience = 300 #We might reduce the patience epoch to accelerate the training
    
    parameter_values = []
    losses = []

    for step in range(steps):

        if patience == 50:
            opt_state.hyperparams['learning_rate'] = 0.001
        
        start = time.time()
        beta = get_kl_weight(step)
        model, loss_value, grad_norm, ratio, opt_state, train_key, kl_loss, post_mean, post_std = make_step(model, opt_state, 
                                                                                                            data, train_key, n_z, optim, beta=beta)
        
        if loss_value < best_loss:
            patience = 300
            best_loss = loss_value
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(model_folder+model_name, model)
        else:
            patience = patience - 1

        end = time.time()
        print(f"Step: {step}, Loss: {loss_value}, KL: {kl_loss}, Computation time: {end - start}, grad_norm : {grad_norm}, ratio : {ratio}")
        
        if (not full_stop_criteria) and patience == 0:
            print(f"early stop, Step {step}, Best loss {best_loss}")
            break

        elif full_stop_criteria:
            if patience == 0 or (grad_norm <= eps_grad_norm and ratio <= eps_delta_norm):
                print(f"Convergence may reach, Step {step}, Best loss {best_loss}, Parameter Ratio {ratio}, Gradient norm {grad_norm}")
                break

    params = np.array([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB, model.logstd_theta, model.logstd_F2, model.logvar_noise])
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    np.savetxt(result_folder+'real_ab_params.txt', params)
    
    return model, params
            
if __name__ == "__main__":
    main()
