from jax_pkpd_model import *
import optax

irregular = False

def loss(model, y, times, state, key_i, n_z, normalized=False):
        
        batch_size = y[:,0,:].shape[0]
        targets = y[:,0,:]

        if normalized:
            y = y/jnp.max(y)

        key_i = jr.split(key_i, batch_size)
        klloss, X, state= jax.vmap(model, axis_name="batch", in_axes=(0, 0, 0, None), out_axes=(0, 0, None))(y, times, key_i, state)
        X = X.reshape(-1, 1)
        
        targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
        const = 2. * jnp.pi
        
        log_noise_var = jnp.log(0.2**2)
        all_reconstruct_error = - 0.5*jnp.log(const) - 0.5*model.log_noise_var - 0.5 * jnp.exp(-model.log_noise_var) * (X - targets)**2
        all_reconstruct_error = all_reconstruct_error.reshape(-1, n_z, y[:,0,:].shape[1])
        
        average_results = jnp.mean(all_reconstruct_error, 1)
        reconstruction_loss = jnp.sum(average_results)
        kl = jnp.sum(klloss)
        return kl - reconstruction_loss, state
    
@eqx.filter_jit
def eval_loss(model, y, times, key_i, n_z, normalized=False):

    batch_size = y[:,0,:].shape[0]
    targets = y[:,0,:]

    if normalized:
        y = y/jnp.max(y)
    
    key_i = jr.split(key_i, batch_size)
    klloss, X, _= jax.vmap(model)(y, times, key_i)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    # log_noise_var = jnp.log(0.2**2)
    log_noise_var = model.func.log_noise_var
    all_reconstruct_error = -0.5*jnp.log(const) - 0.5 * log_noise_var - 0.5 * jnp.exp(-log_noise_var) * (X - targets)**2
    all_reconstruct_error = all_reconstruct_error.reshape(-1, n_z, y[:,0,:].shape[1])
    average_results = jnp.mean(all_reconstruct_error, 1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)
    return kl - reconstruction_loss

@eqx.filter_jit
def make_step(model, state, optim, opt_state, y, times, key_i, n_z):
    start = time.time()

    grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, times, state, key_i, n_z)
    
    end = time.time()
    print(f"grad Computation time: {end - start}")

    key_i = jr.split(key_i, 1)[0]

    start = time.time()
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    end = time.time()
    print(f"all step: {end - start}")
    return state, model, opt_state, key_i

def main(i, seed=42, steps=5000):

    learning_rate = 0.01
    latent_shape = 1
    n_z = 50
    hidden_dim = 16
    measurements = 5
    T = 10
    model_folder = "EXPs/PKPD_exp/irregular_"+str(measurements)+"p_100_10d_noise"

    key = jr.PRNGKey(seed)
    train_key, test_key =  jr.split(key, 2)

    if not irregular:
        train_x = np.loadtxt('PKPD_datasets/PKPD_quality_dataset/linear_data_'+str(i)+'.txt').reshape(100, measurements, -1)[:,:,0]
        test_x = np.loadtxt('PKPD_datasets/PKPD_quality_dataset/linear_data_0.txt').reshape(100, measurements, -1)[:,:,0]
        input_dim = train_x.shape[1]
    else:
        train_data = np.load('PKPD_datasets/'+str(measurements)+'_irregular_1latent_theta1/dataset_'+str(i)+'.npy', allow_pickle=True)
        train_df = pd.DataFrame(train_data.tolist())
        
        y = np.stack(train_df['Y'].to_numpy())
        times = np.stack(train_df['t'].to_numpy())
        # ---- normalized time ----
        train_tnorm = times / T
        # ---- delta_t ----
        delta_t = np.diff(times, axis=1, prepend=times[:, :1])      # delta_t[0] = 0
        delta_t_norm = delta_t / T
        train_x = jnp.stack([y, train_tnorm, delta_t_norm], axis=1)  # (N, 3, T)

        test_data = np.load('PKPD_datasets/'+str(measurements)+'_irregular_1latent_theta1/dataset_0.npy', allow_pickle=True)
        test_df = pd.DataFrame(test_data.tolist())
        
        test_y = np.stack(test_df['Y'].to_numpy())
        test_times = np.stack(test_df['t'].to_numpy())

        test_tnorm = test_times / T
        # ---- delta_t ----
        test_delta_t = np.diff(test_times, axis=1, prepend=test_times[:,:1])     # delta_t[0] = 0
        test_delta_t_norm = test_delta_t / T
        test_x = jnp.stack([test_y, test_tnorm, test_delta_t_norm], axis=1)  # (N, 5, T)
        print(test_x.shape)
    
        input_dim = y.shape[1]

    init_theta1 = jnp.log(0.3)
    init_theta2 = jnp.log(1)
    init_prior_logstd = jnp.log(0.8)
    init_noise_logvar = jnp.log(0.3 ** 2)
    init_params = jnp.array([init_theta1, init_theta2, init_prior_logstd, init_noise_logvar])

    model, state = eqx.nn.make_with_state(PKPD)(latent_shape, init_params, input_dim,  hidden_dim, train_key, 
                                                n_z=n_z, conv=False, rnn=True, input_channel=3, irregular=irregular)

    # optim = optax.adam(learning_rate)
    optim = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    best_loss = jnp.inf
    patience = 100

    for step in range(steps):

        if patience == 0:
            #one-step correction
            print("early-stop..", step, best_loss)
            break

        if patience == 50:
            opt_state.hyperparams['learning_rate'] = 0.005
            # optim.update(params, opt_state)
        
        start = time.time()
        state, model, opt_state, train_key = make_step(model, state, optim, opt_state, train_x, times, train_key, n_z)
        
        #eval
        inference_model = eqx.nn.inference_mode(model)
        inference_model = eqx.Partial(inference_model, state=state)

        test_loss = eval_loss(inference_model, test_x, test_times, test_key, n_z)

        if test_loss < best_loss:
            patience = 500
            best_loss = test_loss
            from pathlib import Path
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(model_folder+"/PKPD_"+str(i)+".eqx", model)
        else:
            patience = patience - 1
        end = time.time()
        print(f"Step: {step}, Loss: {test_loss}, Computation time: {end - start}")

if __name__ == "__main__":
    for i in range(100):
        main(i, seed=i)
