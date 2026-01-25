import os
os.environ['JAX_PLATFORMS']='cpu'
from antibody_multidoses_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np

jax.config.update("jax_enable_x64", True)

latent_shape = 2
n_z = 10
T = 400

init_Ab_production_init_acc = jnp.log(30.0)  # theta
init_Ab_production_init_acc_logvar = jnp.log(0.8**2) # random effect std on theta
init_delta_S = 0.05
init_noise_var = jnp.log(0.1 ** 2)
init_fold_change_2nd_dose = jnp.log(8.0)
init_fold_change_2nd_dose_logvar = jnp.log(1)

init_params = jnp.array([init_Ab_production_init_acc, init_Ab_production_init_acc_logvar, 0.1, 0.2])

model, state = eqx.nn.make_with_state(Antibody)(latent_shape, init_params, n_z=n_z, timepoints=15, hidden_dim=4, T=400, rnn=True, conv=False)

file_path = 'EXPs/exp_antibody_3dose_2latent/rnn_enc_15p_400d_2latent_theta_F2/'
model = eqx.tree_deserialise_leaves(file_path+'Antibody_1.eqx', model)

data_file = 'antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/dataset_1.npy'
all_datas = np.load(data_file)[:,:,1]

seed = 42
key = jr.PRNGKey(seed)
latent_shape = 2
vector_field = Antibody_ode()

estimated_parameters = jnp.array([model.theta, model.F2, model.F3, model.logstd_theta, model.logstd_F2])

def sampling(y, key, model, state):
    key1, key2 = jr.split(key, 2)
    
    post_mean, logstd, state = model.encoder(y, key2, state)
    std = jnp.exp(logstd)
    standard_mean = jnp.zeros(latent_shape)
    standard_cov = jnp.eye(latent_shape)  # Identity covariance ensures independence
    samples = jr.multivariate_normal(key1, standard_mean, standard_cov, shape=(n_z,))
    latent_samples = post_mean + samples * std
    
    return latent_samples, post_mean, logstd

# Decoder of the VAE
def decoder(samples, parameters):
    samples = jnp.exp(samples).reshape(-1,latent_shape)

    theta = jnp.exp(parameters[0]) * samples[:, 0].squeeze()
    F2 = jnp.exp(parameters[1]) * samples[:, 1].squeeze()
    
    def solve(individual_theta, individual_F2):
        # Parameters
        Ab_degrad_rate = 0.08  # delta_Ab
        vaccine_autigen_decline_rate = 2.7  # delta_v
        death_rate_S_cell = 0.01  # delta_s

        term = diffrax.ODETerm(vector_field)

        solver = diffrax.Tsit5()
        # solver = diffrax.Kvaerno4()

        t0 = 0
        t1 = T

        times = jnp.linspace(t0, t1, 15)
        # times = times / self.time_scale

        y0 = jnp.array([0.01, 0.1])

        dt0 = 0.01
        saveat = diffrax.SaveAt(ts=times)

        # Solve the ODE
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8),
            args=(death_rate_S_cell, vaccine_autigen_decline_rate, individual_theta, individual_F2, jnp.exp(parameters[2]), Ab_degrad_rate),
            saveat=saveat,
            adjoint = diffrax.DirectAdjoint()
        )

        return sol.ys

    start_time = time.time()
    print(theta.shape, F2.shape)
    X = jax.vmap(solve)(theta.squeeze(), F2.squeeze())
    # print("solver time", time.time() - start_time, X.shape)
    Ab_trajectories = jnp.log10(X[:, :, 1])

    
    return Ab_trajectories

# @jax.jit
def joint_loglikelihood(parameters, all_datas, key):
    timepoints = all_datas.shape[1]
    key_sub = jr.split(key, all_datas.shape[0])
    
    samples = jax.vmap(sampling, in_axes=(0, 0, None, None))(all_datas, key_sub, model, state)

    y_hat = jax.vmap(decoder, in_axes=(0, None))(samples, parameters)
    targets = jnp.repeat(all_datas, repeats=n_z, axis=0).reshape(-1, 1)
    noise_var = 0.1 ** 2
    const = 2. * jnp.pi

    log_pxz = -0.5*timepoints*jnp.log(const) - 0.5 * timepoints *jnp.log(noise_var) - 0.5 *(1/noise_var) * (y_hat.reshape(-1)- targets.reshape(-1))**2
    log_pxz = log_pxz.reshape(-1,n_z)
    average_results = jnp.mean(log_pxz, 1)
    full_log_pxz = jnp.sum(average_results)
    
    log_pz = jnp.sum(jax.vmap(jax.scipy.stats.multivariate_normal.logpdf, in_axes=(0, None, None))(samples, jnp.zeros(latent_shape), jnp.diag(jnp.array([parameters[3]**2, parameters[4]**2]))))
    joint_ll = full_log_pxz + log_pz
    return joint_ll


def joint_loglikelihood_importance(parameters, all_datas, key):
    timepoints = all_datas.shape[1]
    batch_size = all_datas.shape[0]
    latent_dim = 2  # Or however many dims you have
    key_sub = jr.split(key, batch_size)

    # Sample z ~ q(z | y)
    samples = jax.vmap(sampling, in_axes=(0, 0, None, None))(all_datas, key_sub, model, state)  # [batch, n_z, latent_dim]
    
    # Flatten for decoder
    samples_flat = samples.reshape(-1, latent_dim)
    y_repeated = jnp.repeat(all_datas, repeats=n_z, axis=0)  # [batch * n_z, timepoints]

    # p(y | z): Gaussian decoder
    y_hat = jax.vmap(decoder, in_axes=(0, None))(samples_flat, parameters)
    noise_var = 0.1 ** 2
    log_px_given_z = -0.5 * timepoints * (jnp.log(2 * jnp.pi) + jnp.log(noise_var)) \
        - 0.5 * (1 / noise_var) * jnp.sum((y_hat - y_repeated) ** 2, axis=1)  # [batch * n_z]

    log_px_given_z = log_px_given_z.reshape(batch_size, n_z)

    # log p(z): Prior (assumed diagonal Gaussian)
    prior_std = jnp.array([parameters[3], parameters[4]])
    log_pz = -0.5 * jnp.sum(
        latent_dim * jnp.log(2 * jnp.pi) + jnp.log(prior_std ** 2)
        + (samples / prior_std[None, None, :]) ** 2,
        axis=2
    )  # [batch, n_z]

    # log q(z | y): approximate posterior — assume diagonal Gaussian from encoder
    def log_q_z_given_y(z, y):
        mu, log_std = model.encoder(y, parameters)  # returns (mean, log std)
        var = jnp.exp(2 * log_std)
        return -0.5 * jnp.sum(
            jnp.log(2 * jnp.pi) + 2 * log_std + ((z - mu) ** 2) / var
        )

    log_qz = jax.vmap(
        lambda y, z_samples: jax.vmap(lambda z: log_q_z_given_y(z, y))(z_samples),
        in_axes=(0, 0)
    )(all_datas, samples)  # [batch, n_z]

    # Compute log weights: log p(y,z) - log q(z|y)
    log_weights = log_px_given_z + log_pz - log_qz
    weights = jax.nn.softmax(log_weights, axis=1)  # [batch, n_z]

    # Compute IS-weighted joint log-likelihood
    joint_logpz = log_px_given_z + log_pz
    weighted_loglik = jnp.sum(jnp.sum(weights * joint_logpz, axis=1))  # scalar

    return weighted_loglik



def score(parameters, key):
    grads = jax.grad(joint_loglikelihood)(parameters, all_datas, key)
    hessian = jax.hessian(joint_loglikelihood)(parameters, all_datas, key)
    
    return hessian, grads
    








