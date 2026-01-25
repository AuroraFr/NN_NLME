import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reordering and log-transform of estimated_values_2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# === Set global font sizes ===
plt.rcParams.update({
    "font.size": 14,           # base font size
    "axes.titlesize": 16,      # title font size
    "axes.labelsize": 14,      # x/y label size
    "xtick.labelsize": 15,     # x tick labels
    "ytick.labelsize": 15,     # y tick labels
    "legend.fontsize": 12,     # legend text
    "figure.titlesize": 18     # figure title
})

# === Data Arrays ===
alphas_std = np.linspace(1, 2, 10)
epsilon_std_2 = np.random.normal(0, 1, 10)
init_noise_variances = np.log(0.1 ** 2) + np.abs(alphas_std * epsilon_std_2)
init_noise_variances = np.sqrt(np.exp(init_noise_variances))

init_values = np.loadtxt("antibody_datasets/antibody3doses_deltaAB_identifiability_initparams.txt")
init_values = np.column_stack((init_values, init_noise_variances)) 
deltaAb_new_order = [0, 1, 2, 5, 6, 3, 4, 7]
init_values = init_values[:, deltaAb_new_order]
init_values[:,5] = np.exp(init_values[:, 5])
init_values[:,6] = np.exp(init_values[:, 6])
basic_init_values = init_values[:, [0,1,2,5,6,7]]
deltaS_init_values = init_values[:, [0,1,2,3,5,6,7]]

deltaAb_estimated_values_ELBO = np.loadtxt("results/antibody_3doses_400d_10p_50s_deltaAb_irregular_identifiability_2latents.txt")
deltaAb_estimated_values_SAEM = np.loadtxt("results/monolix_deltaAb_irregular_noise_identifiability.csv", delimiter=",", skiprows=1)

basic_estimated_values_ELBO = deltaAb_estimated_values_ELBO[:, [0,1,2,5,6,-1]]
basic_estimated_values_SAEM = np.loadtxt('results/monolix_basic_irregular_identifiability.csv', delimiter=",", skiprows=1)

deltaS_estimated_values_ELBO = np.loadtxt("results/antibody_3doses_400d_10p_50s_deltaS_irregular_identifiability_2latents.txt")
deltaS_estimated_values_SAEM = np.loadtxt('results/monolix_deltaS_irregular_identifiability.csv', delimiter=",", skiprows=1)
true_values = np.log([24.5, 7.1, 18.5])
true_values_basic = [true_values[0], true_values[1], true_values[2], 0.5, 0.9, 0.1]
true_values_deltaS = [true_values[0], true_values[1], true_values[2], np.log(0.01), 0.5, 0.9, 0.1]
true_values_full = [true_values[0], true_values[1], true_values[2],  np.log(0.01), np.log(0.08) , 0.5, 0.9, 0.1]

print(basic_estimated_values_ELBO, basic_init_values)

# LaTeX labels
deltaAb_latex_labels = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$",
    r"$log(\delta_S)$", r"$log(\delta_{Ab})$" ,r"$\omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$",r"$\epsilon_{Ab}$"
]

basic_latex_labels = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$",
    r"$\omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$", r"$\epsilon_{Ab}$"
]

deltaS_latex_labels = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$",
    r"$log(\delta_S)$", r"$\omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$", r"$\epsilon_{Ab}$"
]

#=== Plot ===
plt.figure(figsize=(12, 4))
#fig, axes = plt.subplots(3, 1, sharey=True, figsize=(12, 10))  # 1 row, 3 columns
offset = 0.1
for col_idx in range(len(basic_latex_labels)):
   plt.plot([col_idx - 0.1, col_idx + 0.1],
                [true_values_basic[col_idx]] * 2,
                color='blue', linewidth=2, label='True value' if col_idx == 0 else None)
   for i in range(len(basic_init_values)):
       # Init → Estimate 1
       y1 = [basic_init_values[i, col_idx], basic_estimated_values_ELBO[i, col_idx]]
       plt.plot(col_idx, basic_init_values[i, col_idx], marker='D', color='green', markersize=3)
       plt.plot(col_idx+offset, basic_estimated_values_ELBO[i, col_idx], 'o', color='black', markersize=6)

       y_vals = basic_init_values[:, col_idx]
       x_vals = [col_idx] * len(y_vals)
       plt.plot(col_idx-offset, basic_estimated_values_SAEM[i, col_idx], 'X', color='red', markersize=6, markeredgewidth=0.4)

# Formatting
plt.xticks(range(len(basic_latex_labels)),basic_latex_labels, rotation=45)
plt.yticks([0, 2, 4, 6])
plt.ylabel("Parameter Value")
plt.grid(True)

# Legend
legend_elements = [
    Line2D([0], [0], color='green', marker='D', linestyle='None', markersize=5, label='Init value'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='VAE estimations'),
    Line2D([0], [0], color='red', marker='X', linestyle='None',  markeredgewidth=0.4, label='SAEM estimations'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='True value')
]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("figures/antibody_3doses_basic_50s_irregular_identifiability.pdf", format="pdf")
plt.close()

plt.figure(figsize=(12, 6))
offset = 0.1
for col_idx in range(len(deltaAb_latex_labels)):
    plt.plot([col_idx - 0.1, col_idx + 0.1],
                  [true_values_full[col_idx]] * 2,
                  color='blue', linewidth=2, label='True value' if col_idx == 0 else None)
    for i in range(len(init_values)):
        # Init → Estimate 1
        y1 = [init_values[i, col_idx], deltaAb_estimated_values_ELBO[i, col_idx]]
        plt.plot(col_idx, init_values[i, col_idx], marker='D', color='green', markersize=3)
        plt.plot(col_idx+offset, deltaAb_estimated_values_ELBO[i, col_idx], 'o', color='black', markersize=6)

        y_vals = init_values[:, col_idx]
        x_vals = [col_idx] * len(y_vals)
        # plt.plot(x_vals, y_vals, color='green', linestyle='-', linewidth=1.5, alpha=0.6)

        # Estimate 2 (log-transformed)
        plt.plot(col_idx-offset, deltaAb_estimated_values_SAEM[i, col_idx], 'X', color='red', markersize=6, markeredgewidth=0.4)

# Formatting
plt.xticks(range(len(deltaAb_latex_labels)),deltaAb_latex_labels, rotation=45)
plt.ylabel("Parameter Value")
plt.grid(True)

# Legend
legend_elements = [
    Line2D([0], [0], color='green', marker='D', linestyle='None', markersize=5, label='Init value'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='VAE estimations'),
    Line2D([0], [0], color='red', marker='X', linestyle='None',  markeredgewidth=0.4, label='SAEM estimations'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='True value')
]
plt.legend(handles=legend_elements)
plt.tight_layout()

plt.savefig("figures/antibody_3doses_deltaAb_50s_irregular_identifiability.pdf", format="pdf")
plt.close()


plt.figure(figsize=(12, 6))
offset = 0.1
for col_idx in range(len(deltaS_latex_labels)):
    plt.plot([col_idx - 0.1, col_idx + 0.1],
                 [true_values_deltaS[col_idx]] * 2,
                 color='blue', linewidth=2, label='True value' if col_idx == 0 else None)
    for i in range(len(deltaS_init_values)):
        # Init → Estimate 1
        y1 = [deltaS_init_values[i, col_idx], deltaS_estimated_values_ELBO[i, col_idx]]
        plt.plot(col_idx, deltaS_init_values[i, col_idx], marker='D', color='green', markersize=3)
        plt.plot(col_idx+offset, deltaS_estimated_values_ELBO[i, col_idx], 'o', color='black', markersize=6)

        y_vals = deltaS_init_values[:, col_idx]
        x_vals = [col_idx] * len(y_vals)
        # plt.plot(x_vals, y_vals, color='green', linestyle='-', linewidth=1.5, alpha=0.6)

        # Estimate 2 (log-transformed)
        plt.plot(col_idx-offset, deltaS_estimated_values_SAEM[i, col_idx], 'X', color='red', markersize=6, markeredgewidth=0.4)

# Formatting
plt.xticks(range(len(deltaS_latex_labels)),deltaS_latex_labels, rotation=45)
plt.ylabel("Parameter Value")
plt.grid(True)

# Legend
legend_elements = [
    Line2D([0], [0], color='green', marker='D', linestyle='None', markersize=5, label='Init value'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='VAE estimations'),
    Line2D([0], [0], color='red', marker='X', linestyle='None',  markeredgewidth=0.4, label='SAEM estimations'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='True value')
]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("figures/antibody_3doses_50s_deltaS_irregular_identifiability.pdf", format="pdf",bbox_inches="tight")
