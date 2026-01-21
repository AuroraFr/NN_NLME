import os
# os.environ['JAX_PLATFORMS']='cpu'
from jax_antibody_multidoses_joint_model import Antibody
import optax
import numpy as np
import time
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)
irregular = False

def loss(model, data, state, key_i, n_z, beta = 1):
        
        y = data['y']
        times = data['times']
        # train_tnorm = data['t_norm']
        # train_deltat = data['delta_t']
        # after30 = data["after30"]
        # after250 = data["after250"]

        # Example: 5-channel input (Ab, t_norm, delta_t, after30, after250)
        # train_x = jnp.stack([y, train_tnorm, train_deltat], axis=1)  # (N, 5, T)

        targets = y
        print("data", y.shape, times.shape)
       
        batch_size = y.shape[0]
        key_sub = jr.split(key_i, batch_size)
        klloss, X, state, _, _, _= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(y, times, key_sub, state)
        X = X.reshape(-1, 1)

        targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
        const = 2. * jnp.pi
        
        # log_noise_var = jnp.log(0.1**2)
        log_p_x_given_z = - 0.5 * jnp.log(const) - 0.5 * model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
        log_p_x_given_z = log_p_x_given_z.reshape(-1,n_z, y.shape[1])
        average_per_subject = jnp.mean(log_p_x_given_z, 1)
        reconstruction_loss = jnp.sum(average_per_subject)
        kl = jnp.sum(klloss)
        
        return beta * kl - reconstruction_loss, state
    
@eqx.filter_jit
def eval_loss(model, state, data, key_i, n_z, beta= 1):
    
    y = data['y']
    times = data['times']
    # test_tnorm = data['t_norm']
    # test_deltat = data['delta_t']
    # after30 = data["after30"]
    # after250 = data["after250"]

    # test_x = jnp.stack([y, test_tnorm, test_deltat], axis=1)
    
    targets = y
    batch_size = y.shape[0]
    key_i = jr.split(key_i, batch_size)
    
    klloss, X, _, _, _, _= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(y, times, key_i, state)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.1**2)

    all_reconstruct_error = - 0.5 * jnp.log(const) - 0.5*model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
    all_reconstruct_error = all_reconstruct_error.reshape(-1,n_z, y.shape[1])
    average_results = jnp.mean(all_reconstruct_error, 1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)
    return beta * kl - reconstruction_loss

def main(i, steps=50000):

    #Training
    learning_rate = 0.01
    latent_shape = 2 
    n_z = 50
    time_length = 0
    T = 400

    seed = 42
    key = jr.PRNGKey(seed)
    train_key, test_key =  jr.split(key, 2)
    
    train_y_list = []
    train_ts_list = []
    train_tnorm_list = []
    train_deltat_list = []
    train_after30_list = []
    train_after250_list = []

    test_y_list = []
    test_ts_list = []
    test_tnorm_list = []
    test_deltat_list = []
    test_after30_list = []
    test_after250_list = []

    dose1 = 30.0
    dose2 = 250.0

    if irregular:

        train_data = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_'+str(i)+'.npy')[:,:,1]
        train_data_timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_timepoints_'+str(i)+'.npy')
        test_data = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_0.npy')[:,:,1]
        test_data_timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/dataset_timepoints_0.npy')
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

            # ---- indicators ----
            after30 = (t >= dose1).astype(np.float32)
            after250 = (t >= dose2).astype(np.float32)

            train_y_list.append(y)
            train_ts_list.append(t)
            train_tnorm_list.append(t_norm)
            train_deltat_list.append(delta_t_norm)
            train_after30_list.append(after30)
            train_after250_list.append(after250)

            # ---------- test ----------
            y_test = test_data[j, :]
            t_test = test_data_timepoints[j, :]

            t_test_norm = t_test / T
            delta_t_test = np.diff(t_test, prepend=t_test[0])
            delta_t_test_norm = delta_t_test / T

            after30_test = (t_test >= dose1).astype(np.float32)
            after250_test = (t_test >= dose2).astype(np.float32)

            test_y_list.append(y_test)
            test_ts_list.append(t_test)
            test_tnorm_list.append(t_test_norm)
            test_deltat_list.append(delta_t_test_norm)
            test_after30_list.append(after30_test)
            test_after250_list.append(after250_test)


        train_data_dict = {
            "y": jnp.stack(train_y_list),          # (N, T)
            "times": jnp.stack(train_ts_list),     # (N, T)
            "t_norm": jnp.stack(train_tnorm_list), # (N, T)
            "delta_t": jnp.stack(train_deltat_list),   # (N, T)
            "after30": jnp.stack(train_after30_list),  # (N, T)
            "after250": jnp.stack(train_after250_list) # (N, T)
        }

        test_data_dict = {
            "y": jnp.stack(test_y_list),
            "times": jnp.stack(test_ts_list),
            "t_norm": jnp.stack(test_tnorm_list),
            "delta_t": jnp.stack(test_deltat_list),
            "after30": jnp.stack(test_after30_list),
            "after250": jnp.stack(test_after250_list),
        }

    else:
        # train_data = np.load('antibody_datasets/scipy_antibody_3dose_15_400_50/dataset_'+str(i)+'.npy')[:,:,1]
        train_data = np.load('antibody_datasets/scipy_antibody_3dose_15_400_50/dataset'+str(i)+'.npy')
        time_length = train_data.shape[1]
        print("regular sampling", time_length, train_data.shape)
        train_data_timepoints = jnp.linspace(0, T, time_length)
        
        # train_data_timepoints = np.array([0, 20, 45, 85, 130, 200, 260, 310, 350, 400])
        # test_data = np.load('antibody_datasets/scipy_antibody_3dose_15_400_50/dataset_0.npy')[:,:,1]
        test_data = np.load('antibody_datasets/scipy_antibody_3dose_15_400_50/dataset0.npy')
        test_data_timepoints = jnp.linspace(0, T, time_length)
        # test_data_timepoints = np.array([0, 20, 45, 85, 130, 200, 260, 310, 350, 400])
        batch_size = train_data.shape[0]

        for j in range(batch_size):
            train_y_list.append(train_data[j, :])
            train_ts_list.append(train_data_timepoints)

            test_y_list.append(test_data[j, :])
            test_ts_list.append(test_data_timepoints)


        train_data_dict = {"y": jnp.stack(train_y_list), "times": jnp.stack(train_ts_list)}
        test_data_dict = {"y": jnp.stack(test_y_list), "times": jnp.stack(test_ts_list)}


    if irregular:
        model_folder = './EXPs/exp_antibody_3dose_2latent/irregular_'+str(time_length)+'p_400d_50s_2latent_theta_F2/'
    else:
        model_folder = './EXPs/exp_antibody_3dose_2latent/'+str(time_length)+'p_400d_50s_2latent_theta_F2_deltaAB_noise/'

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
    
    parameter_values = []
    losses = []

    @eqx.filter_jit
    def make_step(model, state, opt_state, y, key_i, n_z):
        
        start = time.time()
        grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, state, key_i, n_z)

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        # grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, state, key_i, n_z)
        key_i = jr.split(key_i, 1)[0]

        return state, model, opt_state, key_i

    for step in range(steps):

        if patience == 0:

            # np.savetxt(model_folder+'loss_values_'+dataset_name, losses, fmt='%.4f')
            # np.savetxt(model_folder+'parameter_values_'+dataset_name, losses, fmt='%.4f')
            
            print("early-stop..", step, best_loss)
            break

        if patience == 300:
            opt_state.hyperparams['learning_rate'] = 0.005
        
        start = time.time()
        state, model, opt_state, train_key = make_step(model, state, opt_state, train_data_dict, train_key, n_z)
        
        #eval
        inference_model = eqx.nn.inference_mode(model)
        # inference_model = eqx.Partial(inference_model, state=state)

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
        # losses.append(test_loss)
        # parameter_values.append([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB, jnp.exp(model.logstd_theta), jnp.exp(model.logstd_F2)])
        print(f"Step: {step}, Loss: {test_loss}, Computation time: {end - start}")

if __name__ == "__main__":

    import sys
    GPU = int(sys.argv[1])
    if GPU == 0:
        for i in range(1, 2):
            main(i)
    elif GPU == 1:
        for i in range(50, 100):
            main(i)
