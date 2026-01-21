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

def loss(model, y, state, key_i, n_z, beta = 1, normalized=True):
    targets = y
    batch_size = y.shape[0]

    key_sub = jr.split(key_i, batch_size)
    klloss, X, state = jax.vmap(model, axis_name="batch", in_axes=(0, 0, None), out_axes=(0, 0, None))(y, key_sub, state)
    
    
    X = X.reshape(-1, 1)
    print('all trajectories', X.shape)

    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    
    log_noise_var = jnp.log(0.1 **2)
    log_p_x_given_z = -0.5*jnp.log(const) - 0.5*log_noise_var - 0.5 * jnp.exp(-log_noise_var) * (X - targets)**2
    log_p_x_given_z = log_p_x_given_z.reshape(-1,n_z, y.shape[1])
    average_per_subject = jnp.mean(log_p_x_given_z, 1)
    reconstruction_loss = jnp.sum(average_per_subject)
    kl = jnp.sum(klloss)
    
    return beta* kl - reconstruction_loss, state

@eqx.filter_jit
def eval_loss(model, y, key_i, n_z, beta= 1):
    
    targets = y
    batch_size = y.shape[0]

    key_i = jr.split(key_i, batch_size)
    
    klloss, X, _, post_mean, post_logstd = jax.vmap(model)(y, key_i)
    X = X.reshape(-1, 1)
    
    targets = jnp.repeat(targets, repeats=n_z, axis=0).reshape(-1, 1)
    const = 2. * jnp.pi
    log_noise_var = jnp.log(0.1**2)

    all_reconstruct_error = -0.5*jnp.log(const) - 0.5*log_noise_var - 0.5 * jnp.exp(-log_noise_var) * (X - targets)**2
    all_reconstruct_error = all_reconstruct_error.reshape(-1, n_z, y.shape[1])
    average_results = jnp.mean(all_reconstruct_error, 1)
    reconstruction_loss = jnp.sum(average_results)
    kl = jnp.sum(klloss)
    return beta * kl - reconstruction_loss, kl, reconstruction_loss, post_mean, post_logstd

@eqx.filter_jit
def make_step(model, state, opt_state, y, key_i, n_z, optim):
    
    grads, state = eqx.filter_grad(loss, has_aux=True)(model, y, state, key_i, n_z)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    
    key_i = jr.split(key_i, 1)[0]

    return state, model, opt_state, key_i


# --- 1. Define your helper function ---
# This is the function we *thought* was the label_fn
def _get_label(path: PyTree, leaf: jnp.ndarray) -> str:
    """Helper that returns a label string for a single parameter."""
    param_name = path[-1].name
    if param_name == "phiC":
        return "phiC_lr"
    else:
        return "default_lr"

# --- 2. Define the *actual* label_fn ---
# This is the function you must pass to optax.multi_transform.
# It takes the full params PyTree and returns a PyTree of labels.
def label_fn(params: PyTree) -> PyTree:
    return jax.tree_util.tree_map_with_path(_get_label, params)

def main(i, steps=50000, model_path=None):

    #Training
    latent_shape = 1
    n_z = 50

    seed = 42
    key = jr.PRNGKey(seed)
    train_key, test_key =  jr.split(key, 2)

    train_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_'+str(i)+'.npy')[:, :]
    test_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_0.npy')[:, :]

    model_folder = './EXPs/exp_asthme_1latent_quality_kp/'

    batch_size = train_data.shape[0]
    dataset_name = str(i)


    init_Kp = jnp.log(1.5)
    init_phiC = jnp.log(0.2)
    init_Kac = jnp.log(0.05)
    init_Kb = jnp.log(0.8)
    init_Ks = jnp.log(0.1)
    init_Kp_logstd = jnp.log(0.3)
    init_Ks_logstd = jnp.log(0.1)
    init_Kb_logstd = jnp.log(0.7)
    init_phiC_logstd = jnp.log(0.5)

    init_params = jnp.array([init_Kp, init_Kp_logstd])

    model, state = eqx.nn.make_with_state(Asthme)(latent_shape, init_params, train_key, n_z=n_z, T=400, batchnorm=False,
                                                  timepoints=20, time_scale=1, conv=True, kernel_size=7, rnn=False)
    
    if model_path:
        model = eqx.tree_deserialise_leaves(model_path, model)
    
    
    # Set your different learning rates here
    special_rate = 5e-5  # A very slow learning rate for phiC
    default_rate = 5e-4  # A faster rate for kp, Kac, etc.

    schedule = optax.piecewise_constant_schedule(
        init_value=default_rate,
        boundaries_and_scales={5000: 0.5} 
    )

    # This dictionary maps labels to their core optimizers (adam)
    # Do NOT put chain or clip here.
    inner_optimizer_dict = {
        "default_lr": optax.adam(learning_rate=schedule), # <-- Schedule here
        "phiC_lr": optax.adam(learning_rate=special_rate)        # <-- Fixed rate here
    }

    # --- 4. Create the Partitioned Optimizer ---
    # This applies the correct adam optimizer to the correct parameter
    partitioned_optimizer = optax.multi_transform(inner_optimizer_dict, label_fn)

    # --- 5. Create the FINAL, Top-Level Chain ---
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),  # 1. Clipping is applied globally
        # partitioned_optimizer            # 2. Partitioned optimizers are applied
        optax.adam(learning_rate=schedule)
    )

    # --- 6. Initialize Your Optimizer ---
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optim.init(params)
   
    best_loss = jnp.inf
    patience = 500 #We might reduce the patience epoch to accelerate the training
    
    # parameter_values = []
    # losses = []

    for step in range(steps):

        if patience == 0:

            # np.savetxt(model_folder+'loss_values_'+dataset_name, losses, fmt='%.4f')
            # np.savetxt(model_folder+'parameter_values_'+dataset_name, losses, fmt='%.4f')
            
            print("early-stop..", step, best_loss)
            break

        # if patience == 300:
        #     opt_state.hyperparams['learning_rate'] = 0.001
        
        start = time.time()
        # diff_model, static_model = eqx.partition(model, filter_spec)

        state, model, opt_state, train_key = make_step(model, state, opt_state, train_data, train_key, n_z, optim)
        
        #eval
        inference_model = eqx.nn.inference_mode(model)
        inference_model = eqx.Partial(inference_model, state=state, return_latent=True)

        test_loss, kl, recon_loss, post_mean, post_logstd = eval_loss(inference_model, test_data, test_key, n_z)
        
        if test_loss < best_loss:
            patience = 500
            best_loss = test_loss
            from pathlib import Path
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(model_folder+"Asthme_1latent_Kp_"+dataset_name+".eqx", model)
        else:
            patience = patience - 1
        end = time.time()
        # losses.append(test_loss)
        print(f"Step: {step}, Loss: {test_loss}, KL: {kl}, recon_loss: {recon_loss},\n"
        f"Computation time: {end - start}, post_mean mean-std:, {post_mean.mean(axis=0), post_mean.std(axis=0)}, post_std (exp(logstd)) mean:, {jnp.exp(post_logstd).mean(axis=0)}")


if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True)
    # with jax.disable_jit():
    import sys
    GPU = int(sys.argv[1])
    if GPU == 0:
        for i in range(47,50):
            main(i)
    if GPU == 1:
        for i in range(79,80):
            main(i)
