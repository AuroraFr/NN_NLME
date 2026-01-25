import numpy as np

true_Kp = np.log(1.15)
true_Kac = np.log(0.01)
true_Kb = np.log(1)
true_phiC = np.log(0.1)
true_std_Kp = np.log(0.05)

alphas = np.linspace(1, 2, 10)
epsilon_pop_1 = np.random.normal(0, 0.2, 10)
epsilon_pop_2 = np.random.normal(0, 1, 10)
epsilon_pop_3 = np.random.normal(0, 0.5, 10)
init_Kp_values = true_Kp + alphas * epsilon_pop_1
init_Kac_values = true_Kac + alphas * epsilon_pop_2
init_Kb_values = true_Kb + alphas * epsilon_pop_3
init_phiC_values = true_phiC + alphas * epsilon_pop_1

alphas_std = np.linspace(1, 2, 10)
epsilon_std_1 = np.random.normal(0, 0.8, 10)
init_Kp_stds = true_std_Kp + np.abs(alphas_std * epsilon_std_1)

init_params = list(zip(init_Kp_values, init_Kb_values, init_Kac_values, init_phiC_values, init_Kp_stds))
np.savetxt('asthme_datasets/Asthme_1latent_init_params_3.txt',init_params, fmt='%.4f')
init_params = np.loadtxt('asthme_datasets/Asthme_1latent_init_params_3.txt')
print(np.exp(init_params))