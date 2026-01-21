import os
os.environ['JAX_PLATFORMS']='cpu'
from real_ab_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

jax.config.update("jax_enable_x64", True)
latent_shape = 2
n_z = 1

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

# ---- Load your CSV file ----
df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
# ---- Clean + ensure correct types ----
df["id"] = df["id"].astype(int)
df["Time"] = df["Time"].astype(float)
df["ED50_BAU"] = df["ED50_BAU"].astype(float)
df = df.dropna()

model_folder = './EXPs/exp_real_ab/'
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
init_params = jnp.array([init_Ab_production_init_acc, init_F2, init_F3, init_Ab_production_init_acc_logstd, init_F2_logstd,init_logvar_noise])

data = preprocess_data(df)

model = Antibody(latent_shape, init_params, key, n_z=n_z,timepoints=17, time_scale=1, conv=True, use_mask=True,hidden_dim=16, out_channel=16)

for i in range(1):
    model_path = 'EXPs/exp_real_ab/Antibody_real_test.eqx'
    model = eqx.tree_deserialise_leaves(model_path, model)

    padded_y = data["padded_y"]
    mask = data["mask"]
    padded_ts = data["padded_ts"]
    injection_times = data["injection_times"]

    print(f'log_theta : {model.theta}, log_F2:{model.F2}, log_F3: {model.F3}, deltaS: {model.deltaS}, deltaAb: {model.deltaAB}, std_F2: {jnp.exp(model.logstd_F2)}, std_theta: {jnp.exp(model.logstd_theta)}, noise_variance:{jnp.exp(model.logvar_noise)}')

