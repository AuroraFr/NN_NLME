import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

kp_pop = 1.15
kb_pop = 1
ks_pop = 0.2
phiC_pop = 0.1

omega_kp = 0.05
omega_kb = 0.3
omega_ks = 0.1
omega_phiC = 0.1

N_subjects = 50
np.random.seed(42)

# Define the ODE system (Eq. 3.3)
def asthme_vector_field(t, y, params, k_b, k_s, k_p, phi_c):
    p, c, a, m = y
    # k_p = params["k_p"]
    p_max = params["p_max"]
    k_ap = params["k_ap"]
    eta_ap = params["eta_ap"]
    k_cp = params["k_cp"]
    gamma_c = params["gamma_c"]
    gamma_p = params["gamma_p"]
    k_pc = params["k_pc"]
    # phi_c = params["phi_c"]
    # k_s = params["k_s"]
    nu = params["nu"]
    t_is = params["t_i"]
    k_ac = params["k_ac"]
    eta_ac = params["eta_ac"]
    gamma_a = params["gamma_a"]
    # k_b = params["k_b"]
    k_pm = params["k_pm"]
    k_apm = params["k_apm"]
    eta_apm = params["eta_apm"]
    phi_m = params["phi_m"]

    stimulus = sum((k_s / nu) * np.exp(-((t - t_i) ** 2) / nu ** 2) for t_i in t_is)

    dpdt = k_p * p * (1 - p / p_max) * (1 + (k_ap * a * p) / (eta_ap + a * p)) + k_cp * gamma_c / gamma_p * c - k_pc * p
    dcdt = k_pc * gamma_p / gamma_c * p - (k_cp + phi_c) * c
    dadt = (stimulus + (k_ac * a * c) / (eta_ac + a * c)) * c * m / gamma_a - k_b * (gamma_p * p / gamma_c + c) * a - a
    dmdt = (k_pm + (k_apm * a * p) / (eta_apm + a * p)) * gamma_p * p - phi_m * m

    return [dpdt, dcdt, dadt, dmdt]

# Simulate subject data
def simulate_a_t(params, y0, t_eval, kb=1, ks=0.2, kp=1.15, phiC=0.1):
    def wrapped(t, y): return asthme_vector_field(t, y, params, k_b=kb, k_s=ks, k_p=kp, phi_c=phiC)
    sol = solve_ivp(wrapped, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval)
    return sol # a(t)

# Define parameters
params = {
    # "k_p": 1.15,
    "p_max": 1.0,
    "k_ap": 1.0,
    "eta_ap": 1.0,
    "k_cp": 0.01,
    "gamma_c": 1.0,
    "gamma_p": 1.0,
    "k_pc": 1.0,
    # "k_s": 0.2,
    "nu": 30.0,
    # "phi_c": 0.1,
    "t_i": list(range(50, 250, 40)),
    "k_ac": 0.01,
    "eta_ac": 1.0,
    "gamma_a": 0.01,
    # "k_b": 1.0,
    "k_pm": 0.1,
    "k_apm": 0.1,
    "eta_apm": 10.0,
    "phi_m": 0.01
}

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
    # base = np.array([0, 20, 55, 95, 135, 175, 215, 260, 350, 400])      # shape (10,)
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

# Initial condition and time grid
y0 = [0.1, 0.8, 0.01, 0.9]
measurements = 20
t_eval = np.linspace(0, 400, 10)
T = 400

dataset_folder = 'asthme_datasets/'+'irregular_'+str(measurements)+'_'+str(T)+'_'+str(50)+'_1latent_Kp/'
Path(dataset_folder).mkdir(parents=True, exist_ok=True)
eps = 1e-6

# sol = simulate_a_t(params, y0, t_eval, kb=kb_pop, ks=ks_pop, kp=1.15, phiC=0.1)
# # # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(sol.t, sol.y[0], label="p(t) — Proliferating ASM", linestyle='--')
# plt.plot(sol.t, sol.y[1], label="c(t) — Contractile ASM", linestyle='--')
# plt.plot(sol.t, sol.y[2], label="a(t) — Active TGF-β")
# plt.plot(sol.t, sol.y[3], label="m(t) — ECM", linestyle='--')
# plt.xlabel("Time (Days)")
# plt.legend()
# plt.tight_layout()
# plt.savefig('figures/asthme_10.pdf')

for i in range(100):
    times_list = []
    print(i)
    datas = []
    nonoise_datas = []

    kp_random_values = np.random.normal(scale=omega_kp, size=N_subjects)
    true_kps = kp_pop*np.exp(kp_random_values)
    ks_random_values = np.random.normal(scale=omega_ks, size=N_subjects)
    true_KSs = ks_pop*np.exp(ks_random_values)
    kb_random_values = np.random.normal(scale=omega_kb, size=N_subjects)
    true_Kbs = kb_pop*np.exp(kb_random_values)
    phiC_random_values = np.random.normal(scale=omega_phiC, size=N_subjects)
    true_phiCs = phiC_pop*np.exp(phiC_random_values)

    # for (true_kb, true_KS) in zip(true_Kbs, true_KSs):
    # for (true_kp, true_phiC) in zip(true_kps, true_phiCs):
    # for (true_kp, true_KS) in zip(true_kps, true_KSs):
    for true_kp in true_kps:

        times = generate_jittered_times(T, n_points=measurements)
        times_list.append(times)
        sol = simulate_a_t(params, y0, times, kb=kb_pop, ks=ks_pop)
        
        # Add Gaussian noise to a(t)
        noise_std = 0.1
        a_obs = np.log10(np.maximum(sol.y[2], eps)) + np.random.normal(0, noise_std, size=len(sol.y[2]))
        datas.append(a_obs)
        nonoise_datas.append([sol.y[3], sol.y[2]])

    # Plot results
    datas = np.array(datas)
    nonoise_datas = np.array(nonoise_datas)

    np.save(dataset_folder+'dataset_'+str(i), datas)
    np.save(dataset_folder+'dataset_timepoints_'+str(i), times_list)
    np.save(dataset_folder+'dataset_nonoise_'+str(i), nonoise_datas)