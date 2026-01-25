###SIMULATION 
# antibody kinetic for regular sampling dataset
###

import os
# os.environ['JAX_PLATFORMS']='cpu'
from antibody_multidoses_model import Antibody
import optax
import numpy as np
import time
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from exp_utils import gaussian_log_prob

jax.config.update("jax_enable_x64", True)

def loss(model, data, state, key_i, n_z, beta = 1):
        
    y = data['y']
    times = data['times']
    batch_size = y.shape[0]
    
    targets = y.copy()
    key_sub = jr.split(key_i, batch_size)
    klloss, X, state, _, _, _= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(y, times, key_sub, state)
    X = X.reshape(-1, 1)
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    
    # const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.1**2)
    # log_p_x_given_z = - 0.5 * jnp.log(const) - 0.5 * model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
    log_p_y_given_b = gaussian_log_prob(model, X, targets)
    log_p_y_given_b = log_p_y_given_b.reshape(-1, n_z, y.shape[1])
    average_per_subject = jnp.mean(log_p_y_given_b, 1)
    reconstruction_loss = jnp.sum(average_per_subject)
    kl = jnp.sum(klloss)
    
    return beta * kl - reconstruction_loss, state
    
@eqx.filter_jit
def eval_loss(model, state, data, key_i, n_z, beta= 1):
    
    y = data['y']
    times = data['times']
    batch_size = y.shape[0]
    
    targets = y.copy()
    key_i = jr.split(key_i, batch_size)
    
    klloss, X, _, _, _, _= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(y, times, key_i, state)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    # const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.1**2)
    all_reconstruct_error = gaussian_log_prob(model, X, targets)
    all_reconstruct_error = all_reconstruct_error.reshape(-1,n_z, y.shape[1])
    average_results = jnp.mean(all_reconstruct_error, axis=1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)
    return beta * kl - reconstruction_loss

@eqx.filter_jit
def make_step(model, state, opt_state, y, key_i, n_z, optim):
    
    params_old = eqx.filter(model, eqx.is_inexact_array)
    grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, state, key_i, n_z)           
    
    updates, opt_state = optim.update(grads, opt_state, params_old)
    
    # Apply the noisy update
    params_new = optax.apply_updates(params_old, updates)
    model = eqx.apply_updates(model, updates)
    
    key_i = jr.split(key_i, 1)[0]

    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    delta_params = jax.tree_util.tree_map(lambda a, b: a - b, params_new, params_old)
    delta_norm = jnp.sqrt(sum(jnp.sum(d**2) for d in jax.tree_util.tree_leaves(delta_params)))
    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params_old)))
    ratio = delta_norm / (param_norm + 1e-8)

    return state, model, opt_state, key_i, grad_norm, ratio

def main(i, seed=42, steps=50000, model_folder='./EXPs/exp_antibody_3dose_2latent/10p_400d_50s_2latent_theta_F2_deltaAb_noise_check/', data_folder='antibody_datasets/scipy_antibody_3dose_15_400_50/'):

    #Training
    learning_rate = 0.01
    latent_shape = 2 
    n_z = 50
    time_length = 0
    T = 400
    full_stop_criteria = True
    irregular = False
    eps_grad_norm = 1e-4
    eps_delta_norm = 1e-6

    key = jr.PRNGKey(seed)
    train_key, test_key =  jr.split(key, 2)
    
    train_y_list = []
    train_ts_list = []

    test_y_list = []
    test_ts_list = []
 
    # train_data = np.load('antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_'+str(i)+'.npy')[:,:,1]
    train_data = np.load(data_folder+'dataset'+str(i)+'.npy')
    time_length = train_data.shape[1]
    print("regular sampling", time_length, train_data.shape)
    train_data_timepoints = jnp.linspace(0, T, time_length)
    
    # test_data = np.load('antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_0.npy')[:,:,1]
    test_data = np.load(data_folder+'dataset0.npy')
    test_data_timepoints = jnp.linspace(0, T, time_length)
    batch_size = train_data.shape[0]

    for j in range(batch_size):
        train_y_list.append(train_data[j, :])
        train_ts_list.append(train_data_timepoints)

        test_y_list.append(test_data[j, :])
        test_ts_list.append(test_data_timepoints)


    train_data_dict = {"y": jnp.stack(train_y_list), "times": jnp.stack(train_ts_list)}
    test_data_dict = {"y": jnp.stack(test_y_list), "times": jnp.stack(test_ts_list)}

    dataset_name = str(i)

    ############################ True Parameters ####################
    fold_change_2nd_dose = 7.1 #F2
    fold_chnage_3rd_dose = 18.5 #F3
    fold_change_2nd_dose_std = 0.9 #F2 std
    Ab_degrad_rate = 0.08  # delta_Ab
    vaccine_autigen_decline_rate = 2.7  # delta_v
    death_rate_S_cell = 0.01  # delta_s
    Ab_noise_std = 0.1
    theta = 24.5  # theta
    theta_std = 0.5  # random effect std on theta
    #################################################################


    init_theta = jnp.log(30)
    init_theta_logstd = jnp.log(0.8)
    init_F2 = jnp.log(8)
    init_F2_logstd = jnp.log(0.5)
    init_F3 = jnp.log(22)
    init_deltaS = -3.5
    init_deltaAB = -2.885 #lambda, real_deltaAB = exp(lambda) + deltaS
    init_noise_logvar = jnp.log(0.2**2)

    init_params = jnp.array([init_theta, init_F2, init_F3, init_theta_logstd, init_F2_logstd, init_deltaS, init_deltaAB, init_noise_logvar])

    model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, train_key, n_z=n_z, T=400, kernel_size=3,
                                                    timepoints=time_length, time_scale=1, conv=True, rnn=False, irregular=irregular, input_channel=1, out_channel=1, hidden_dim=4)

    optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optim.init(params)

    best_loss = jnp.inf
    patience = 500 #We might reduce the patience epoch to accelerate the training

    for step in range(steps):

        if patience == 300:
            opt_state.hyperparams['learning_rate'] = 0.005
        
        start = time.time()
        state, model, opt_state, train_key, grad_norm, ratio = make_step(model, state, opt_state, train_data_dict, train_key, n_z, optim) 
        
        #eval
        inference_model = eqx.nn.inference_mode(model)

        test_loss = eval_loss(inference_model, state, test_data_dict, test_key, n_z)
        
        if test_loss < best_loss:
            patience = 500
            best_loss = test_loss
            from pathlib import Path
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(model_folder+"Antibody_"+dataset_name+".eqx", model)
        else:
            patience = patience - 1

        end = time.time()
        print(f"Step: {step}, Loss: {test_loss}, Computation time: {end - start}, Parameter Ratio {ratio}, Gradient norm {grad_norm}")

        if (not full_stop_criteria) and patience == 0:
            print(f"early stop, Step {step}, Best loss {best_loss}")
            break

        elif full_stop_criteria:
            if patience == 0 or (grad_norm <= eps_grad_norm and ratio <= eps_delta_norm):
                print(f"Convergence may reach, Step {step}, Best loss {best_loss}, Parameter Ratio {ratio}, Gradient norm {grad_norm}")
                break
    
    return np.array([model.theta,model.F2, model.F3, model.logstd_theta, model.logstd_F2, model.deltaS, model.deltaAB, model.log_noise_var])

if __name__ == "__main__":
    import sys
    GPU = int(sys.argv[1])
    if GPU == 0:
        for i in range(35, 50):
            main(i)
    elif GPU == 1:
        for i in range(78, 100):
            main(i)
