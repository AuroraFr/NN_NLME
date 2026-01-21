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

def log_prior_logstd(logstd_prior, mu=0.0, tau=1.0):
    """Log-probability under Normal(mu, tau^2)"""
    return -0.5 * jnp.sum(((logstd_prior - mu) / tau)**2 + jnp.log(2 * jnp.pi * tau**2))

def log_half_cauchy(sigma, scale=1.0):
    return -jnp.sum(jnp.log(jnp.pi * scale * (1 + (sigma / scale)**2)))

@eqx.filter_jit
def compute_loss(model, y, key, n_z, beta=1):
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

    # --- Define different weights for each latent variable ---
    # We assume latent_dim 0 is theta, latent_dim 1 is F2.
    # Penalize theta's KL loss 3x more than F2's.
    kl_weights = jnp.array([1.0, 1.0]) 

    # Apply the weights to get the final, single KL loss value
    weighted_kl_loss = jnp.sum(kl_loss_vector * kl_weights)

    X_reshaped = X.reshape(-1, 1)
    targets_reshaped = jnp.repeat(padded_y, repeats=n_z, axis=0).reshape(-1,1)
    log_p_x_given_z = -0.5 * jnp.log(const) - 0.5 * model.logvar_noise - 0.5 * jnp.exp(-model.logvar_noise) * (X_reshaped - targets_reshaped)**2
    log_p_x_given_z_structured = log_p_x_given_z.reshape(B, n_z, T_max)

    log_p_x_given_z_mean = jnp.mean(log_p_x_given_z_structured, axis=1)
    # In your compute_loss function
    # second_inj = injection_times[:, 0, None]
    # third_inj = injection_times[:, 1, None]

    # is_in_f2_window = (padded_ts > second_inj) & (padded_ts < third_inj)
    # is_in_f3_window = (padded_ts >= third_inj)

    # # Use nested where for tiered weights (e.g., 5.0 for F2, 2.0 for F3)
    # time_weights = jnp.where(is_in_f2_window, 5.0,
    #                     jnp.where(is_in_f3_window, 2.0, 1.0))

    masked_log_likelihood = log_p_x_given_z_mean * mask 
    reconstruction_loss = jnp.sum(masked_log_likelihood)

    # 4. KL LOSS (This part was already correct)
    kl_loss = jnp.sum(klloss)
    lambda_F2 = 1.5
    lambda_theta = 0.8
    # entropy = 0.5 * jnp.sum(2 * post_logstd + jnp.log(2 * jnp.pi * jnp.e), axis=-1)
    prior_penalty = -log_prior_logstd(model.get_logstd_prior(), mu=0.0, tau=1.0)
    # + lambda_F2 * model.logstd_F2 ** 2 + lambda_theta * model.logstd_theta ** 2 + 0.2 * model.theta**2

    # Final negative ELBO for minimization
    return -(reconstruction_loss - beta * weighted_kl_loss),  (weighted_kl_loss, post_mean.mean(axis=0), (jnp.exp(post_logstd).mean(axis=0)))

def main(steps=25000):

    @eqx.filter_jit
    def make_step(model, opt_state, y, key, n_z, beta=1):
        start = time.time()
        # with jax.disable_jit():
        (loss_value, (kl_loss, post_mean, post_std)), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, y, key, n_z, beta=beta)
        # print("grad logvar_noise =", eqx.filter(grads, eqx.is_inexact_array).logvar_noise)
        # jtu.tree_map(lambda x: print(x.shape, x.dtype) if x is not None else print("None"), grads)
        # print("loss =", loss_value)
        # print("grad logvar_noise =", eqx.filter(grads, eqx.is_inexact_array).logvar_noise)
        # grads_filtered = eqx.filter(grads, eqx.is_inexact_array)
        # jax.tree_util.tree_map(lambda x: print("grad shape:", x.shape, "norm:", jnp.linalg.norm(x)), grads_filtered)                
        
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optim.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        
        key_i = jr.split(key, 2)[0]

        return model, loss_value, opt_state, key_i, kl_loss, post_mean, post_std

    #Training
    learning_rate = 0.005
    latent_shape = 2 #latent variables dim * 2.
    n_z = 100

    seed = 42
    key = jr.PRNGKey(seed)
    train_key, _ =  jr.split(key, 2)

    # ---- Load your CSV file ----
    df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
    # ---- Clean + ensure correct types ----
    df["id"] = df["id"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["ED50_BAU"] = df["ED50_BAU"].astype(float)
    df = df.dropna()

    model_folder = './EXPs/exp_real_ab/'

    # init_Ab_production_init_acc = jnp.log(30)  # theta
    # init_Ab_production_init_acc_logstd = jnp.log(0.8) # random effect std on theta
    # init_F2 = jnp.log(8.0)
    # init_F2_logstd = jnp.log(0.5)
    # init_F3 = jnp.log(20)
    # init_logvar_noise = jnp.log(0.4**2)
    # init_deltaS = -3.96 #
    # init_deltaAB = -2.88
    # init_params = jnp.array([init_Ab_production_init_acc, init_F2, init_F3, init_Ab_production_init_acc_logstd, init_F2_logstd,init_logvar_noise, init_deltaS, init_deltaAB])

    data = preprocess_data(df)

    init_params = np.loadtxt('real_ab3doses_deltaAB_identifiability_initparams.txt')
    for j, init_param in enumerate(init_params):

        model = Antibody(latent_shape, init_param, n_z=n_z, timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=32, out_channel=32)
        optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

        # optim = optax.chain(
        # optax.clip_by_global_norm(1.0),  # Clip gradients if their norm > 1.0
        # optax.adam(learning_rate=learning_rate))

        # filter_spec = jtu.tree_map(lambda _: False, model)
        # filter_spec = eqx.tree_at(lambda m: m.encoder, filter_spec, replace=jtu.tree_map(lambda _: True, model.encoder))

        params = eqx.filter(model, eqx.is_inexact_array)
        # jax.tree_util.tree_map(lambda x: print(type(x), getattr(x, 'shape', None)), params)
        opt_state = optim.init(params)

        
        best_loss = jnp.inf
        patience = 500 #We might reduce the patience epoch to accelerate the training

        model_path = 'EXPs/exp_real_ab/Antibody_real_deltaAb.eqx'
        # model = eqx.tree_deserialise_leaves(model_path, model)
        
        parameter_values = []
        losses = []

        for step in range(steps):

            if patience == 0:

                # np.savetxt(model_folder+'loss_values_'+dataset_name, losses, fmt='%.4f')
                # np.savetxt(model_folder+'parameter_values_'+dataset_name, losses, fmt='%.4f')
                
                print("early-stop..", step, best_loss)
                break

            if patience == 50:
                opt_state.hyperparams['learning_rate'] = 0.001
            
            start = time.time()
            beta = get_kl_weight(step)
            model, loss_value, opt_state, train_key,kl_loss, post_mean, post_std = make_step(model, opt_state, data, train_key, n_z, beta=beta)
            
            if loss_value < best_loss:
                patience = 500
                best_loss = loss_value
                from pathlib import Path
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(model_folder+"Antibody_real_deltaS_identi_"+str(j)+".eqx", model)
            else:
                patience = patience - 1

            end = time.time()
            # losses.append(test_loss)
            # parameter_values.append([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB, jnp.exp(model.logstd_theta), jnp.exp(model.logstd_F2)])
            print(f"Step: {step}, Loss: {loss_value}, KL: {kl_loss}, Computation time: {end - start}, post_std : {post_std}, post_mean : {post_mean}")

if __name__ == "__main__":
    main()
