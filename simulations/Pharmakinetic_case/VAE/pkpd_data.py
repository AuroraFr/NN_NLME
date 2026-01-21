import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
# ----------------------------------------------------
# 1. TRUE PARAMETERS
# ----------------------------------------------------
theta1 = np.log(0.5)    # population mean log(phi_1)
theta2 = np.log(2.0)    # log(phi_2)
Omega = 0.5             # SD of random effect b_i

phi2_true = np.exp(theta2)
x10, x20 = 2.0, 3.0     # initial conditions
sigma_eps = 0.2

# simulation setup
T_end = 10
num_points = 5
# t_obs = np.linspace(0, T_end, num_points)

N_subjects = 100   # you can change

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

    base = np.linspace(0, T, n_points)
    base = [0.0, 0.2, 0.7, 2.5, 5.0, 10]
    step = T / (n_points - 1)                 # ≈ 28.6 for 0..10, 6 pts
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

# ----------------------------------------------------
# 2. ODE SYSTEM
# ----------------------------------------------------
def pk_ode(t, X, phi1_i, phi2):
    X1, X2 = X
    dX1 = phi2 * X2 - phi1_i * X1
    dX2 = -phi2 * X2
    return [dX1, dX2]

# ----------------------------------------------------
# 3. SIMULATE ALL SUBJECTS
# ----------------------------------------------------
for j in range(100):
    data = []
    for i in range(N_subjects):
        # random effect
        b_i = np.random.normal(0, Omega)
        phi1_i = np.exp(theta1 + b_i)
        t_obs = generate_jittered_times(T=T_end, n_points=6, jitter_frac=0.01)

        # solve ODE
        sol = solve_ivp(
            fun=lambda t, X: pk_ode(t, X, phi1_i, phi2_true),
            t_span=(0, T_end),
            y0=[x10, x20],
            t_eval=t_obs,
            method="RK45"
        )

        X1 = sol.y[0]  # observed state

        # add noise
        Y = X1 + np.random.normal(0, sigma_eps, size=X1.shape)

        data.append({
            "t": t_obs,
            "X1": X1,
            "Y": Y
        })

    dataset_folder = 'PKPD_datasets/'+str(num_points)+'_irregular_1latent_theta1/'
    Path(dataset_folder).mkdir(parents=True, exist_ok=True)
    np.save(dataset_folder+'dataset_'+str(j), data)
