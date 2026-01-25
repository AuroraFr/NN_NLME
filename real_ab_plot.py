import os
os.environ['JAX_PLATFORMS']='cpu'
from real_ab_model import *
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.rcParams.update({
    "font.size": 14,           # base font size
    "axes.titlesize": 16,      # title font size
    "axes.labelsize": 14,      # x/y label size
    "xtick.labelsize": 13,     # x tick labels
    "ytick.labelsize": 13,     # y tick labels
    "legend.fontsize": 12,     # legend text
    "figure.titlesize": 18     # figure title
})

jax.config.update("jax_enable_x64", True)
latent_shape = 2
b_sample_size = 5000

def preprocess_data(df):
    """
    Pads all sequence data to a unif,rm length for use with jax.lax.scan or jax.vmap.
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

def plot_preprocessed_data(data_dict):
    """
    Plots padded JAX data structures.
    Ignores padded values by utilizing the mask.
    """
    # 1. Extract and convert JAX arrays to standard Numpy arrays
    # Flatten y to shape (N, T) if it is (N, T, 1)
    padded_y = np.array(data_dict["padded_y"]).squeeze() 
    padded_ts = np.array(data_dict["padded_ts"])
    mask = np.array(data_dict["mask"])
    
    # Optional: Injection times
    injection_times = np.array(data_dict["injection_times"])

    # 2. Setup Plot
    plt.figure(figsize=(10, 6))
    
    num_subjects = padded_y.shape[0]

    # 3. Iterate through each subject
    for i in range(num_subjects):
        # Create a boolean mask for this specific subject
        # We explicitly cast to bool to select indices
        valid_idx = mask[i] == 1.0
        
        # Slice the time and y arrays using the mask
        t_plot = padded_ts[i][valid_idx]
        y_plot = padded_y[i][valid_idx]
        
        # Plot the trajectory
        # alpha=0.5 makes lines semi-transparent so you can see overlap
        plt.plot(t_plot, 10**y_plot, marker='o', markersize=3, alpha=0.6, label=f'ID {i}' if num_subjects < 10 else None)

        # Optional: Plot injection times for this subject
        # We only plot them if they fall within the valid time range
        inj_2, inj_3 = injection_times[i]
        plt.axvline(inj_2, color='grey', alpha=0.1, linestyle='--', linewidth=1)
        plt.axvline(inj_3, color='black', alpha=0.5, linestyle='--', linewidth=1)

    # 4. Formatting
    plt.xlim(0, 550)
    plt.xlabel("Time (Days)")
    plt.ylabel("Antibody (BAU/ml)")
    plt.grid(True, alpha=0.3)
    
    # Only show legend if there aren't too many subjects
    if num_subjects <= 10:
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('figures/raw_real_data.pdf')

# ---- Load your CSV file ----
df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
# ---- Clean + ensure correct types ----
df["id"] = df["id"].astype(int)
df["Time"] = df["Time"].astype(float)
df["ED50_BAU"] = df["ED50_BAU"].astype(float)
df = df.dropna()

data = preprocess_data(df)
plot_preprocessed_data(data)
