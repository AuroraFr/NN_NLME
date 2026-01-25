import os
# os.environ['JAX_PLATFORMS']='cpu'
from real_ab_model import *
import numpy as np
import jax.numpy as jnp
import equinox as eqx
jax.config.update("jax_enable_x64", True)

def inference(model_path='EXPs/exp_real_ab_identifiability/Antibody_real_deltaAb_7.eqx'):
    parameters = []
    latent_shape = 2
    n_z = 1

    seed = 42
    key = jr.PRNGKey(seed)

    init_Ab_production_init_acc = jnp.log(30)  # theta
    init_Ab_production_init_acc_logstd = jnp.log(0.8) # random effect std on theta
    init_F2 = jnp.log(7)
    init_F2_logstd = jnp.log(0.5)
    init_F3 = jnp.log(18.0)
    init_logvar_noise = jnp.log(0.4**2)
    init_deltaS = -2.9645
    init_deltaAB = -2.885
    init_params = jnp.array([init_Ab_production_init_acc, init_F2, init_F3, init_Ab_production_init_acc_logstd, init_deltaS,init_deltaAB, init_F2_logstd,init_logvar_noise])

    model = Antibody(latent_shape, init_params, key, n_z=n_z,timepoints=17, time_scale=1, conv=True, use_mask=True, hidden_dim=32, out_channel=32)

    for i in range(10):
        model_path = 'EXPs/exp_real_ab_identifiability/Antibody_real_'+str(i)+'.eqx'
        model = eqx.tree_deserialise_leaves(model_path, model)
        print(f'log_theta : {model.theta}, log_F2:{model.F2}, log_F3: {model.F3}, deltaS: {model.deltaS}, deltaAb: {model.deltaAB}, std_F2: {jnp.exp(model.logstd_F2)}, std_theta: {jnp.exp(model.logstd_theta)}, noise_variance:{jnp.exp(model.logvar_noise)}')
        parameters.append([model.theta, model.F2, model.F3, model.deltaS, model.deltaAB, jnp.exp(model.logstd_theta), jnp.exp(model.logstd_F2),jnp.sqrt(jnp.exp(model.logvar_noise))])

    np.savetxt('real_ab_ELBO_deltaAB_identifiability_results.txt', parameters, fmt="%.4f")
if __name__ == "__main__":
    inference()