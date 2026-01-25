from scipy.integrate import odeint
import numpy as np
from pathlib import Path

np.random.seed(42)

def vector_field(y0, time, death_rate_S_cell, vaccine_autigen_decline_rate, Ab_production_init_acc, Ab_degrad_rate, fold_change_2, fold_change_3):

    S, Ab = y0
    fold_change = 1  # Assuming fold_change is constant here
    second_dose = 30
    third_dose = 250
    diff_time = time

    if time - second_dose >= 0 and time - third_dose < 0:
        fold_change = fold_change_2
        diff_time = time - second_dose
    elif time - third_dose >= 0:
        fold_change = fold_change_3
        diff_time = time - third_dose
    
    dSdt = fold_change * np.exp(-vaccine_autigen_decline_rate * diff_time) - death_rate_S_cell * S
    dAbdt = Ab_production_init_acc * S - Ab_degrad_rate * Ab

    return dSdt, dAbdt

def generate_data(n_sample, T, timepoints):

    Ab_production_init_acc = 24.5 # theta
    Ab_production_init_acc_std = 0.5 # random effect std on theta
    fold_change_2 = 7.1
    fold_change_2_std = 0.9
    fold_change_3 = 18.5

    Ab_degrad_rate = 0.08  # delta_Ab
    vaccine_autigen_decline_rate = 2.7  # delta_v
    death_rate_S_cell = 0.01  # delta_s

    Ab_noise_std = 0.1
    S_noise_std = 0.1

    # theta_random_values = np.random.normal(scale=Ab_production_init_acc_std, size=n_sample)
    F2_random_values = np.random.normal(scale=fold_change_2_std, size=n_sample)
        
    times = np.linspace(0, T, timepoints)

    y0 = (0.01, 0.1)

    theta_random_values = np.random.normal(scale=Ab_production_init_acc_std, size=n_sample)
    Ab_production_init_acc_values = np.array(Ab_production_init_acc * np.exp(theta_random_values))
    F2_values = np.array(fold_change_2 * np.exp(F2_random_values))
    datas = []
    data_nonoise = []
    
    #every individual
    for Ab_production_init_acc_value, individual_F2 in zip(Ab_production_init_acc_values, F2_values):
        sol = odeint(vector_field, y0, times, args=(death_rate_S_cell, vaccine_autigen_decline_rate, 
                                                    Ab_production_init_acc_value, Ab_degrad_rate, individual_F2, fold_change_3))

        Ab_noises = np.random.normal(scale=Ab_noise_std, size=len(times))
        S_noises = np.random.normal(scale=S_noise_std, size=len(times))

        S_trajectories = np.log10(sol[:, 0]) + S_noises
        Ab_trajectories = np.log10(sol[:, 1]) + Ab_noises
       
        datas.append(np.stack((S_trajectories, Ab_trajectories), axis=1))
        data_nonoise.append(np.log10(sol))

    datas = np.array(datas)
    data_nonoise = np.array(data_nonoise)
    
    return datas, data_nonoise


def generate_jittered_times(T=400.0, n_points=10, jitter_frac=0.1):
    """
    Generate 'total_points' timepoints per subject by:
      - Starting from an equal grid: np.linspace(0, T, total_points)
      - Adding small Gaussian jitter to each point
      - Clipping to [0, T]
      - Keeping 0 and T fixed (optional but usually nice)
    
    Args:
        T: final time (e.g. 400)
        total_points: number of timepoints (e.g. 15)
        jitter_frac: jitter std as fraction of grid step
                     e.g. 0.05 → 5% of step (~1.4 days for 0..400, 15 pts)
    
    Returns:
        times: (total_points,) sorted array of timepoints
    """
    base = np.array([0, 20, 45, 85, 130, 160, 200, 260, 320, 400])
    # base = np.linspace(0, T, n_points)
    step = T / (n_points - 1)                 # ≈ 28.6 for 0..400, 15 pts
    jitter_std = step * jitter_frac               # e.g. 0.05 * 28.6 ≈ 1.43 days

    # Jitter around the base grid
    times = base + np.random.normal(
        loc=0.0, scale=jitter_std, size=n_points
    )

    # Clip to [0, T]
    times = np.clip(times, 0.0, T)

    # (Optional) keep first and last exactly at 0 and T
    times[0] = 0.0
    times[-1] = T

    # Sort to ensure increasing order
    times = np.sort(times)

    return times


def generate_data_irregular(n_sample, T, n_points):

    # ---- Model parameters ----
    Ab_production_init_acc = 24.5
    Ab_production_init_acc_std = 0.5
    fold_change_2 = 7.1
    fold_change_2_std = 0.9
    fold_change_3 = 18.5

    Ab_degrad_rate = 0.08
    vaccine_autigen_decline_rate = 2.7
    death_rate_S_cell = 0.01

    Ab_noise_std = 0.1
    S_noise_std = 0.1

    # ---- Random effects ----
    theta_random = np.random.normal(scale=Ab_production_init_acc_std, size=n_sample)
    F2_random = np.random.normal(scale=fold_change_2_std, size=n_sample)

    Ab_theta_values = Ab_production_init_acc * np.exp(theta_random)
    F2_values = fold_change_2 * np.exp(F2_random)

    y0 = (0.01, 0.1)

    datas = []
    datas_nonoise = []
    times_list = []

    # ---- Per subject simulation ----
    for Ab_theta, F2 in zip(Ab_theta_values, F2_values):

        # Generate subject-specific irregular times
        times = generate_jittered_times(T, n_points=n_points)
        times_list.append(times)

        sol = odeint(
            vector_field, y0, times,
            args=(death_rate_S_cell, vaccine_autigen_decline_rate,
                  Ab_theta, Ab_degrad_rate, F2, fold_change_3)
        )

        # Add noise (log10 space)
        S_noisy = np.log10(sol[:, 0]) + np.random.normal(scale=S_noise_std, size=len(times))
        Ab_noisy = np.log10(sol[:, 1]) + np.random.normal(scale=Ab_noise_std, size=len(times))

        datas.append(np.stack((S_noisy, Ab_noisy), axis=1))
        datas_nonoise.append(np.log10(sol))

    return np.array(datas), np.array(datas_nonoise), np.array(times_list)

        
irregular = False

for i in range(100):

    subjects = 50
    duration = 400
    measurements = 15
    if irregular:
        dataset, dataset_nonoise, timelist = generate_data_irregular(subjects, duration, measurements)
        dataset_folder = 'antibody_datasets/scipy_antibody_irregular_3dose/10_'+str(duration)+'_'+str(subjects)+'_2latent_theta_F2_2/'
        Path(dataset_folder).mkdir(parents=True, exist_ok=True)
        np.save(dataset_folder+'dataset_'+str(i), dataset)
        np.save(dataset_folder+'dataset_nonoise_'+str(i), dataset_nonoise)
        np.save(dataset_folder+'dataset_timepoints_'+str(i), timelist)
    else:
        dataset, dataset_nonoise = generate_data(subjects, duration, measurements)
        dataset_folder = 'antibody_datasets/scipy_antibody_3dose_2/'+str(measurements)+'_'+str(duration)+'_'+str(subjects)+'_2latent_theta_F2/'
        Path(dataset_folder).mkdir(parents=True, exist_ok=True)
        np.save(dataset_folder+'dataset_'+str(i), dataset)
        np.save(dataset_folder+'dataset_nonoise_'+str(i), dataset_nonoise)

