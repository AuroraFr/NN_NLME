from antibody_multidoses_model import *
import optax
from jax.scipy.sparse.linalg import cg
import numpy as np
import jax.tree_util as jtu
from time import time

jax.config.update("jax_enable_x64", True)

def loss(model, data, state, key_i, n_z, beta = 1):

    y = data['y']
    times = data['times']
    train_tnorm = data['t_norm']
    train_deltat = data['delta_t']
    batch_size = y.shape[0]
    
    targets = y
    train_x = jnp.stack([y, train_tnorm, train_deltat], axis=1)  # (N, 3, T)
    key_sub = jr.split(key_i, batch_size)
    klloss, X, state,_,_,_= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(train_x, times, key_sub, state)
    
    X = X.reshape(-1, 1)
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    
    const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.1 **2)
    log_p_x_given_z = - 0.5*jnp.log(const) - 0.5*model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
    log_p_x_given_z = log_p_x_given_z.reshape(-1, n_z, y.shape[1])
    average_per_subject = jnp.mean(log_p_x_given_z, 1)
    reconstruction_loss = jnp.sum(average_per_subject)
    kl = jnp.sum(klloss)
    
    return beta * kl - reconstruction_loss, state

@eqx.filter_jit
def eval_loss(model, state, data, key_i, n_z, beta = 1):

    y = data['y']
    times = data['times']
    test_tnorm = data['t_norm']
    test_deltat = data['delta_t']

    test_x = jnp.stack([y, test_tnorm, test_deltat], axis=1)
    
    targets = y
    batch_size = y.shape[0]
    key_sub = jr.split(key_i, batch_size)
    
    klloss, X, _, _, _, _= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None, 0, 0, 0))(test_x, times, key_sub, state)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    log_noise_var = jnp.log(0.1**2)

    all_reconstruct_error = - 0.5 * jnp.log(const) - 0.5 * model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
    all_reconstruct_error = all_reconstruct_error.reshape(-1,n_z, y.shape[1])
    average_results = jnp.mean(all_reconstruct_error, 1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)

    return beta * kl - reconstruction_loss


def main(i, steps=55000):

    @eqx.filter_jit
    def make_step(model, state, opt_state, y, key_i, n_z):
        
        grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, state, key_i, n_z)

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        key_i = jr.split(key_i, 1)[0]

        return state, model, opt_state, key_i

    #Training
    learning_rate = 0.01
    latent_shape = 2 #latent variables dim * 2.
    n_z = 50
    irregular = True
    measurements = 10
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
        train_data = np.load('antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_'+str(i)+'.npy')[:,:,1]
        test_data = np.load('antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_0.npy')[:,:,1]
        data_timepoints = jnp.linspace(0, T, measurements)

        for j in range(batch_size):
            train_y_list.append(train_data[j, :])
            train_ts_list.append(data_timepoints)

            test_y_list.append(test_data[j, :])
            test_ts_list.append(data_timepoints)

        train_data_dict = {"y": jnp.stack(train_y_list), "times": jnp.stack(train_ts_list)}
        test_data_dict = {"y": jnp.stack(test_y_list), "times": jnp.stack(test_ts_list)}

    else:
        suffix = '_400_50_2latent_theta_F2'
        train_data = np.load('antibody_datasets/scipy_antibody_irregular_3dose/'+str(measurements)+suffix+'/dataset_'+str(i)+'.npy')[:,:,1]
        train_data_timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/'+str(measurements)+suffix+'/dataset_timepoints_'+str(i)+'.npy')
        test_data = np.load('antibody_datasets/scipy_antibody_irregular_3dose/'+str(measurements)+suffix+'/dataset_0.npy')[:,:,1]
        test_data_timepoints = np.load('antibody_datasets/scipy_antibody_irregular_3dose/'+str(measurements)+suffix+'/dataset_timepoints_0.npy')
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
            "delta_t": jnp.stack(train_deltat_list)  # (N, T)
        }

        test_data_dict = {
            "y": jnp.stack(test_y_list),
            "times": jnp.stack(test_ts_list),
            "t_norm": jnp.stack(test_tnorm_list),
            "delta_t": jnp.stack(test_deltat_list)
        }


    batch_size = train_data.shape[0]


    init_params = np.loadtxt('antibody_datasets/simu_ab_convergence_initparams.txt')

    if irregular:
        model_folder = './EXPs/exp_antibody_3dose_2latent/JASA_irregular_identifiability_'+str(time_length)+'p_400d_50s_deltaS_noise_2latent_theta_F2/'
    else:
        model_folder = './EXPs/exp_antibody_3dose_2latent/identifiability_'+str(time_length)+'p_400d_50s_2latent_theta_F2/'


    for j, init_param in enumerate(init_params):
        
        print(init_param)
        
        seed = 42
        key = jr.PRNGKey(seed)
        train_key, test_key =  jr.split(key, 2)

        model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_param, train_key, 
                                                        n_z=n_z, T=400, timepoints=measurements, irregular=irregular, 
                                                        time_scale=1, conv=False, rnn=True, input_channel=3, hidden_dim=32)

        optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optim.init(params)

        best_loss = jnp.inf
        patience = 800 #We might reduce the patience epoch to accelerate the training

        for step in range(steps):

            if patience == 0:
                print("early-stop..", step, best_loss)
                break

            if patience == 500:
                opt_state.hyperparams['learning_rate'] = 0.005
            
            start = time()
            state, model, opt_state, train_key = make_step(model, state, opt_state, train_data_dict, train_key, n_z)
            
            #eval
            inference_model = eqx.nn.inference_mode(model)
            # inference_model = eqx.Partial(inference_model, state=state)

            test_loss = eval_loss(inference_model, state, test_data_dict, test_key, n_z)
            
            if test_loss < best_loss:
                patience = 800
                best_loss = test_loss
    
                from pathlib import Path
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(model_folder+"Antibody_"+str(j)+".eqx", model)
            else:
                patience = patience - 1
            end = time()
            print(f"Step: {step}, Loss: {test_loss}, Computation time: {end - start}")

if __name__ == "__main__":   
    main(1)
