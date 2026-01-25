import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reordering and log-transform of estimated_values_2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

init_values = np.loadtxt('antibody_datasets/real_ab_convergence_initparams.txt')

estimated_values_ELBO_deltaAb= np.loadtxt('results/real_ab_ELBO_deltaAB_identifiability_results.txt')
estimated_values_SAEM_deltaAb = np.loadtxt('results/real_ab_SAEM_deltaAb_identifiability_results.txt')

# === Transform and Reorder ===

# Reorder columns: [0, 1, 2, 5, 6, 3, 4]
new_order = [0, 1, 2, 5, 6, 3, 4, 7]
init_values = init_values[:, new_order].copy()

init_values[:, 5] = np.exp(init_values[:, 5])  # omega_theta
init_values[:, 6] = np.exp(init_values[:, 6])
init_values[:, 7] = np.sqrt(np.exp(init_values[:, 7]))

# LaTeX labels
latex_labels = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$",
    r"$\Omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$", r"$log(\epsilon)$"
]
latex_labels2 = [
    r"$log(\vartheta$)", r"$\log(F_{\bar{M_2}})$", r"$\log(F_{\bar{M_3}})$",
    r"$\Omega_{\vartheta}$", r"$\omega_{F_{\bar{M_2}}}$",r"$(\epsilon)$",
    r"$log(\delta_S)$"
]
latex_labels3 = [
    r"$log(\vartheta$)", r"$\log(\bar{f}_{M_2})$", r"$\log(\bar{f}_{M_3})$", r"$log(\delta_S)$", r"$log(\delta_{Ab})$", 
    r"$\omega_{\vartheta}$", r"$\omega_{\bar{f}_{M_2}}$",
     r"$\sigma_\epsilon$"
]

# === Plot ===
print(len(init_values))
plt.figure(figsize=(12, 6))
offset = 0.05
for col_idx in range(len(latex_labels3)):
    for i in range(len(init_values)):
        # Init → Estimate 1
        plt.plot(col_idx, init_values[i, col_idx], marker='D', color='green', markersize=3)
        
        y_vals = init_values[:, col_idx]
        x_vals = [col_idx] * len(y_vals)
        # plt.plot(x_vals, y_vals, color='green', linestyle='-', linewidth=1.5, alpha=0.6)

        # Estimate 2 (log-transformed)
        plt.plot(col_idx+offset, estimated_values_ELBO_deltaAb[i, col_idx], 'o', color='black', markersize=5)
        plt.plot(col_idx-offset, estimated_values_SAEM_deltaAb[i, col_idx], 'X', color='red', markeredgewidth=0.4, markersize=5)

# Formatting
plt.xticks(range(len(latex_labels3)), latex_labels3, rotation=45)
plt.ylabel("Parameter Value")
plt.grid(True)

# Legend
legend_elements = [
    Line2D([0], [0], color='green', marker='D', linestyle='None', markersize=6, label='Init value'),
    Line2D([0], [0], color='red', marker='X', linestyle='None', markeredgewidth=0.4, label='SAEM estimations'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='VAE estimations'),
]
plt.legend(handles=legend_elements)
plt.tight_layout()
# plt.savefig("figures/antibody_real_identifiability.pdf", format="pdf")
