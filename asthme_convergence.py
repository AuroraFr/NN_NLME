from asthme_model import Asthme
import optax
import numpy as np
from time import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import time
from jaxtyping import PyTree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
irregular = False

def loss(model, data, state, key_i, n_z, beta = 1):

    y = data['y']
    times = data['times']
    # train_tnorm = data['t_norm']
    # train_deltat = data['delta_t']
    batch_size = y.shape[0]
    
    targets = y
    # train_x = jnp.stack([y, train_tnorm, train_deltat], axis=1)  # (N, 3, T)
    key_sub = jr.split(key_i, batch_size)
    klloss, X, state, _, _ = jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), 
                                      out_axes=(0, 0, None, 0, 0))(y, times, key_sub, state)
    
    X = X.reshape(-1, 1)
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    
    const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.1 **2)
    log_p_x_given_z = - 0.5*jnp.log(const) - 0.5*model.logvar_noise - 0.5 * jnp.exp(-model.logvar_noise) * (X - targets)**2
    log_p_x_given_z = log_p_x_given_z.reshape(-1, n_z, y.shape[1])
    average_per_subject = jnp.mean(log_p_x_given_z, 1)
    reconstruction_loss = jnp.sum(average_per_subject)
    kl = jnp.sum(klloss)
    
    return beta * kl - reconstruction_loss, state

@eqx.filter_jit
def eval_loss(model, state, data, key_i, n_z, beta = 1):

    y = data['y']
    times = data['times']
    # test_tnorm = data['t_norm']
    # test_deltat = data['delta_t']

    # test_x = jnp.stack([y, test_tnorm, test_deltat], axis=1)
    
    targets = y
    batch_size = y.shape[0]
    key_sub = jr.split(key_i, batch_size)
    
    klloss, X, _, post_mean, post_logstd = jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), 
                                                    out_axes=(0, 0, None, 0, 0))(y, times, key_sub, state)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    log_noise_var = jnp.log(0.1**2)

    all_reconstruct_error = - 0.5 * jnp.log(const) - 0.5 * model.logvar_noise - 0.5 * jnp.exp(-model.logvar_noise) * (X - targets)**2
    all_reconstruct_error = all_reconstruct_error.reshape(-1,n_z, y.shape[1])
    average_results = jnp.mean(all_reconstruct_error, 1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)

    return beta * kl - reconstruction_loss, kl, reconstruction_loss, post_mean, post_logstd

@eqx.filter_jit
def make_step(model, state, opt_state, data, key_i, n_z, optim):
    
    grads, state = eqx.filter_grad(loss, has_aux=True)(model, data, state, key_i, n_z)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    
    key_i = jr.split(key_i, 1)[0]

    return state, model, opt_state, key_i

def _get_label(path: PyTree, leaf: jnp.ndarray) -> str:
    """Helper that returns a label string for a single parameter."""
    param_name = path[-1].name
    if param_name == "phiC":
        return "phiC_lr"
    else:
        return "default_lr"

def label_fn(params: PyTree) -> PyTree:
    return jax.tree_util.tree_map_with_path(_get_label, params)

def main(lower_bound, upper_bound, steps=50000):

    #Training
    latent_shape = 1
    n_z = 50

    seed = 42
    key = jr.PRNGKey(seed)
    train_key, test_key =  jr.split(key, 2)
    measurements = 20
    T = 400

    train_y_list = []
    train_ts_list = []
    train_tnorm_list = []
    train_deltat_list = []

    test_y_list = []
    test_ts_list = []
    test_tnorm_list = []
    test_deltat_list = []

    if not irregular:
        train_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_1.npy')
        batch_size = train_data.shape[0]
        test_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_0.npy')
        data_timepoints = np.linspace(0, T, measurements)

        for j in range(batch_size):
            train_y_list.append(train_data[j, :])
            train_ts_list.append(data_timepoints)

            test_y_list.append(test_data[j, :])
            test_ts_list.append(data_timepoints)

        train_data_dict = {"y": jnp.stack(train_y_list), "times": jnp.stack(train_ts_list)}
        test_data_dict = {"y": jnp.stack(test_y_list), "times": jnp.stack(test_ts_list)}

    else:
        train_data = np.load('asthme_datasets/irregular_20_400_50_1latent_Kp/dataset_1.npy')
        train_data_timepoints = np.load('asthme_datasets/irregular_20_400_50_1latent_Kp/dataset_timepoints_1.npy')
        test_data = np.load('asthme_datasets/irregular_20_400_50_1latent_Kp/dataset_0.npy')
        test_data_timepoints = np.load('asthme_datasets/irregular_20_400_50_1latent_Kp/dataset_timepoints_0.npy')
        batch_size = train_data.shape[0]
        time_length = train_data.shape[1]
        print('irregular sampling', time_length)

        for j in range(batch_size):

            y = train_data[j, :]
            t = train_data_timepoints[j, :]

            # ---- normalized time ----
            t_norm = t / T

            # ---- delta_t ----
            delta_t = np.diff(t, prepend=t[0])     # delta_t[0] = 0
            delta_t_norm = delta_t / T

            train_y_list.append(y)
            train_ts_list.append(t)
            train_tnorm_list.append(t_norm)
            train_deltat_list.append(delta_t_norm)


            # ---------- test ----------
            y_test = test_data[j, :]
            t_test = test_data_timepoints[j, :]

            t_test_norm = t_test / T
            delta_t_test = np.diff(t_test, prepend=t_test[0])
            delta_t_test_norm = delta_t_test / T

            test_y_list.append(y_test)
            test_ts_list.append(t_test)
            test_tnorm_list.append(t_test_norm)
            test_deltat_list.append(delta_t_test_norm)


        train_data_dict = {
            "y": jnp.stack(train_y_list),          # (N, T)
            "times": jnp.stack(train_ts_list),     # (N, T)
            "t_norm": jnp.stack(train_tnorm_list), # (N, T)
            "delta_t": jnp.stack(train_deltat_list)   # (N, T)
        }

        test_data_dict = {
            "y": jnp.stack(test_y_list),
            "times": jnp.stack(test_ts_list),
            "t_norm": jnp.stack(test_tnorm_list),
            "delta_t": jnp.stack(test_deltat_list)
        }

    model_folder = './EXPs/exp_asthme_1latent_noise_20_convergence_kp_kb_kac_mylaptop/'

    batch_size = train_data.shape[0]
    init_params = np.loadtxt('asthme_datasets/Asthme_1latent_init_params_2.txt')

    for index, init_param in enumerate(init_params[lower_bound:upper_bound, :]):

        init_param = jnp.append(init_param, jnp.log(0.2**2))
        print(init_param)

        model, state = eqx.nn.make_with_state(Asthme)(latent_shape, init_param, train_key, 
                                                      n_z=n_z, T=400, batchnorm=False,
                                                    timepoints=measurements, time_scale=1, conv=True, irregular=False,
                                                    kernel_size=7, rnn=False, input_channel=1, hidden_dim=4)
               
        # Set different learning rates here
        special_rate = 5e-5  # A very slow learning rate for phiC
        default_rate = 5e-4  # A faster rate for kp, Kac, etc.

        schedule = optax.piecewise_constant_schedule(
            init_value=default_rate,
            boundaries_and_scales={5000: 0.5} 
        )

        inner_optimizer_dict = {
            "default_lr": optax.adam(learning_rate=schedule), # <-- Schedule here
            "phiC_lr": optax.adam(learning_rate=special_rate)        # <-- Fixed rate here
        }

        partitioned_optimizer = optax.multi_transform(inner_optimizer_dict, label_fn)

        optim = optax.chain(
            optax.clip_by_global_norm(1.0),  # 1. Clipping is applied globally
            # partitioned_optimizer            # 2. Partitioned optimizers are applied
            optax.adam(learning_rate=schedule)
        )

        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optim.init(params)
    
        best_loss = jnp.inf
        patience = 500 

        for step in range(steps):

            if patience == 0:
                print("early-stop..", step, best_loss)
                break
            start = time.time()

            state, model, opt_state, train_key = make_step(model, state, opt_state, train_data_dict, train_key, n_z, optim)
            
            #eval
            inference_model = eqx.nn.inference_mode(model)
            # inference_model = eqx.Partial(inference_model, state=state, return_latent=True)
            test_loss, kl, recon_loss, post_mean, post_logstd = eval_loss(inference_model, state, test_data_dict, test_key, n_z)
            
            if test_loss < best_loss:
                patience = 500
                best_loss = test_loss
                from pathlib import Path
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                file_index = index + lower_bound
                eqx.tree_serialise_leaves(model_folder+"Asthme_1latent_Kp_"+str(file_index)+".eqx", model)
            else:
                patience = patience - 1
            end = time.time()
            print(f"Step: {step}, Loss: {test_loss}, KL: {kl}, Computation time: {end - start}")


if __name__ == "__main__":
    import sys
    lower_bound = int(sys.argv[1])
    upper_bound = int(sys.argv[2])
    main(lower_bound, upper_bound)
