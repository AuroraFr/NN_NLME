from real_ab_model import Antibody
import optax
import numpy as np
import time
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pandas as pd

jax.config.update("jax_disable_jit", False)
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
def compute_loss(model, y, key, n_z, beta=1):
    """
    Computes the total loss for a batch of subjects using jax.vmap for parallelization.
    """
    padded_y = y["padded_y"]
    mask = y["mask"]
    padded_ts = y["padded_ts"]
    injection_times = y["injection_times"]
    
    B, T_max, _ = padded_y.shape
    keys = jr.split(key, B)

    const = 2.0 * jnp.pi
    klloss, X, post_mean, post_logstd = jax.vmap(model,  axis_name="batch")(padded_y, mask, keys, padded_ts, injection_times)

    kl_loss_vector = jnp.sum(klloss, axis=0) # Resulting shape: (latent_shape,)

    kl_weights = jnp.array([1.0, 1.0]) 

    weighted_kl_loss = jnp.sum(kl_loss_vector * kl_weights)

    X_reshaped = X.reshape(-1, 1)
    targets_reshaped = jnp.repeat(padded_y, repeats=n_z, axis=0).reshape(-1,1)
    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * model.logvar_noise - 0.5 * jnp.exp(-model.logvar_noise) * (X_reshaped - targets_reshaped)**2
    log_p_x_given_z_structured = log_p_x_given_z.reshape(B, n_z, T_max)

    log_p_x_given_z_mean = jnp.mean(log_p_x_given_z_structured, axis=1)

    masked_log_likelihood = log_p_x_given_z_mean * mask 
    reconstruction_loss = jnp.sum(masked_log_likelihood)

    # Final negative ELBO for minimization
    return -(reconstruction_loss - beta * weighted_kl_loss),  (weighted_kl_loss, post_mean.mean(axis=0), (jnp.exp(post_logstd).mean(axis=0)))

@eqx.filter_jit
def make_step(model, optim, opt_state, y, key, n_z, beta=1):
    start = time.time()
    (loss_value, (kl_loss, post_mean, post_std)), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, y, key, n_z, beta=beta)              
    
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optim.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    
    key_i = jr.split(key, 2)[0]

    return model, loss_value, opt_state, key_i, kl_loss, post_mean, post_std

def main(steps=25000):

    #Training
    learning_rate = 0.005
    latent_shape = 2 #latent variables dim * 2.
    n_z = 100

    # ---- Load your CSV file ----
    df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
    # ---- Clean + ensure correct types ----
    df["id"] = df["id"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["ED50_BAU"] = df["ED50_BAU"].astype(float)
    df = df.dropna()

    model_folder = './EXPs/exp_real_ab_identifiability/'

    data = preprocess_data(df)

    init_params = np.loadtxt('antibody_datasets/real_ab_convergence_initparams.txt')

    for j, init_param in enumerate(init_params):

        seed = 0
        key = jr.PRNGKey(seed)
        train_key, _ =  jr.split(key, 2)

        new_order = [0, 1, 2, 3, 4, 7, 5, 6]
        init_param = init_param[new_order]
        print(init_param)

        model = Antibody(latent_shape, init_param, train_key, n_z=n_z, timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=32, out_channel=32)
        optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optim.init(params)

        best_loss = jnp.inf
        patience = 500 #We might reduce the patience epoch to accelerate the training
        
        parameter_values = []
        losses = []

        for step in range(steps):

            if patience == 0:
                print("early-stop..", step, best_loss)
                break

            if patience == 50:
                opt_state.hyperparams['learning_rate'] = 0.001
            
            start = time.time()
            beta = get_kl_weight(step)
            model, loss_value, opt_state, train_key,kl_loss, post_mean, post_std = make_step(model, optim, opt_state, data, train_key, n_z, beta=beta)
            
            if loss_value < best_loss:
                patience = 500
                best_loss = loss_value
                from pathlib import Path
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(model_folder+"Antibody_real_"+str(j)+".eqx", model)
            else:
                patience = patience - 1

            end = time.time()
            
            parameter_values.append([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB, jnp.exp(model.logstd_theta), jnp.exp(model.logstd_F2), jnp.sqrt(jnp.exp(model.logvar_noise))])
            print(f"Step: {step}, Loss: {loss_value}, KL: {kl_loss}, Computation time: {end - start}, post_std : {post_std}, post_mean : {post_mean}")
    
    np.savetxt('results/real_ab_identifiability.txt', parameter_values, fmt='%.4f')
    return parameter_values

if __name__ == "__main__":
    main()
