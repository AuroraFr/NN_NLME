import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})
# ----------------------------
# Your data (as you pasted)
# ----------------------------
init_values = np.loadtxt("antibody_datasets/simu_ab_convergence_initparams.txt")
init_values[:, -1] = np.sqrt(np.exp(init_values[:, -1]))

regular_deltaAb_estimated_values_ELBO = np.array([
    [3.1874, 2.0572, 2.8766, -0.6991, -0.1952, -4.6001, -2.5363],
    [3.1857, 2.0594, 2.8773, -0.6983, -0.2013, -4.6001, -2.5359],
    [3.1861, 2.0513, 2.8749, -0.6991, -0.1987, -4.6000, -2.5369],
    [3.1844, 2.0498, 2.8766, -0.6989, -0.1964, -4.5999, -2.5364],
    [3.1893, 2.0579, 2.8744, -0.6993, -0.1936, -4.6002, -2.5366],
    [3.1885, 2.0564, 2.8745, -0.7022, -0.1968, -4.5995, -2.5371],
    [3.1885, 2.0564, 2.8745, -0.7022, -0.1968, -4.5995, -2.5371],
    [3.1892, 2.0556, 2.8744, -0.6983, -0.1952, -4.6006, -2.5366],
    [3.1821, 2.0568, 2.8795, -0.7037, -0.2057, -4.5980, -2.5371],
    [3.1881, 2.0477, 2.8743, -0.6954, -0.1923, -4.6006, -2.5365]
])

regular_deltaAb_estimated_values_SAEM = np.array([
    [3.1300, 2.0707, 2.8926, -4.5927, -2.5821, 0.5024, 0.8368],
    [3.1623, 2.0761, 2.8921, -4.5983, -2.5441, 0.5013, 0.8368],
    [3.1500, 2.0592, 2.8825, -4.5970, -2.5680, 0.5041, 0.8429],
    [3.1642, 2.0670, 2.8862, -4.5988, -2.5479, 0.5021, 0.8406],
    [5.3273, 1.9885, 2.7070, -0.3028, -4.8416, 0.4978, 0.8812],
    [5.7764, 1.9857, 2.7066, -4.8486, 0.1484, 0.5024, 0.8851],
    [3.1853, 2.0555, 2.8758, -4.6030, -2.5332, 0.5021, 0.8416],
    [3.1682, 2.0542, 2.8769, -4.6003, -2.5521, 0.5011, 0.8427],
    [6.3807, 1.9788, 2.7014, -4.8504, 0.7459, 0.4991, 0.8834],
    [5.9665, 1.9988, 2.7156, -4.8472, 0.3459, 0.5021, 0.8837]
])


regular_basic_estimated_values_ELBO = np.array([
    [3.1974, 2.0572, 2.8766, -0.6991, -0.1952],
    [3.1957, 2.0594, 2.8773, -0.6983, -0.2013],
    [3.1961, 2.0513, 2.8749, -0.6991, -0.1987],
    [3.1944, 2.0498, 2.8766, -0.6989, -0.1964],
    [3.1993, 2.0579, 2.8744, -0.6993, -0.1936],
    [3.1985, 2.0564, 2.8745, -0.7022, -0.1968],
    [3.1985, 2.0564, 2.8745, -0.7022, -0.1968],
    [3.1992, 2.0556, 2.8744, -0.6983, -0.1952],
    [3.1921, 2.0568, 2.8795, -0.7037, -0.2057],
    [3.1981, 2.0477, 2.8743, -0.6954, -0.1923]
])

regular_basic_estimated_values_SAEM = np.array([
    [3.1800, 2.0707, 2.8926, 0.5024, 0.8868],
    [3.1823, 2.0761, 2.8921, 0.5013, 0.8868],
    [3.1800, 2.0592, 2.8825, 0.5041, 0.8829],
    [3.1842, 2.0670, 2.8862, 0.5021, 0.8806],
    [3.1873, 1.9885, 2.7070, 0.4978, 0.8812],
    [3.1964, 1.9857, 2.7066, 0.5024, 0.8851],
    [3.1853, 2.0555, 2.8758, 0.5021, 0.8916],
    [3.1682, 2.0542, 2.8769, 0.5011, 0.8927],
    [3.1807, 1.9788, 2.7014, 0.4991, 0.8834],
    [3.1865, 1.9988, 2.7156, 0.5021, 0.8837]
])

regular_deltaS_estimated_values_ELBO = np.array([
    [3.1874, 2.0572, 2.8766, -0.6991, -0.1952, -4.6001],
    [3.1857, 2.0594, 2.8773, -0.6983, -0.2013, -4.6001],
    [3.1861, 2.0513, 2.8749, -0.6991, -0.1987, -4.6000],
    [3.1844, 2.0498, 2.8766, -0.6989, -0.1964, -4.5999],
    [3.1893, 2.0579, 2.8744, -0.6993, -0.1936, -4.6002],
    [3.1885, 2.0564, 2.8745, -0.7022, -0.1968, -4.5995],
    [3.1885, 2.0564, 2.8745, -0.7022, -0.1968, -4.5995],
    [3.1892, 2.0556, 2.8744, -0.6983, -0.1952, -4.6006],
    [3.1821, 2.0568, 2.8795, -0.7037, -0.2057, -4.5980],
    [3.1881, 2.0477, 2.8743, -0.6954, -0.1923, -4.6006]
])

regular_deltaS_estimated_values_SAEM = np.array([
    [3.1800, 2.0707, 2.8926, -4.5927, 0.5024, 0.8868],
    [3.1823, 2.0761, 2.8921, -4.5983, 0.5013, 0.8868],
    [3.1900, 2.0592, 2.8825, -4.5970, 0.5041, 0.8929],
    [3.1842, 2.0670, 2.8862, -4.5988, 0.5021, 0.8906],
    [3.1973, 1.9885, 2.8070, -4.5928, 0.4978, 0.8812],
    [3.1864, 1.9857, 2.8066, -4.6086, 0.5024, 0.8851],
    [3.1853, 2.0555, 2.8758, -4.6030, 0.5021, 0.8816],
    [3.1882, 2.0542, 2.8769, -4.6003, 0.5011, 0.8927],
    [3.1807, 1.9788, 2.8014, -4.5904, 0.4991, 0.8834],
    [3.1965, 1.9988, 2.8156, -4.5972, 0.5021, 0.8837]
])

deltaAb_new_order = [0, 1, 2, 5, 6, 3, 4]
deltaS_new_order = [0, 1, 2, 5, 3, 4]
deltaAb_estimated_values_ELBO_corrected = regular_deltaAb_estimated_values_ELBO[:, deltaAb_new_order].copy()
deltaS_estimated_values_ELBO_corrected = regular_deltaS_estimated_values_ELBO[:, deltaS_new_order].copy()

deltaAb_estimated_values_ELBO_corrected[:, 5] = np.exp(deltaAb_estimated_values_ELBO_corrected[:, 5])  # omega_theta
deltaAb_estimated_values_ELBO_corrected[:, 6] = np.exp(deltaAb_estimated_values_ELBO_corrected[:, 6])  # omega_F2
deltaS_estimated_values_ELBO_corrected[:, 4] = np.exp(deltaS_estimated_values_ELBO_corrected[:, 4])
deltaS_estimated_values_ELBO_corrected[:, 5] = np.exp(deltaS_estimated_values_ELBO_corrected[:, 5])
regular_basic_estimated_values_ELBO[:, 3] = np.exp(regular_basic_estimated_values_ELBO[:, 3])  # omega_theta
regular_basic_estimated_values_ELBO[:, 4] = np.exp(regular_basic_estimated_values_ELBO[:, 4])  # omega_theta

VAE_estimate_noise_variances = [0.12, 0.11, 0.095, 0.10, 0.11, 0.1, 0.12, 0.099, 0.098, 0.12]
deltaAb_estimated_values_ELBO_corrected = np.column_stack((deltaAb_estimated_values_ELBO_corrected, VAE_estimate_noise_variances)) 
deltaS_estimated_values_ELBO_corrected = np.column_stack((deltaS_estimated_values_ELBO_corrected, VAE_estimate_noise_variances)) 
basic_estimated_values_ELBO = np.column_stack((regular_basic_estimated_values_ELBO, VAE_estimate_noise_variances))

regular_deltaAb_estimated_values_SAEM = np.column_stack((regular_deltaAb_estimated_values_SAEM, VAE_estimate_noise_variances)) 
regular_deltaS_estimated_values_SAEM = np.column_stack((regular_deltaS_estimated_values_SAEM, VAE_estimate_noise_variances)) 
regular_basic_estimated_values_SAEM = np.column_stack((regular_basic_estimated_values_SAEM, VAE_estimate_noise_variances)) 

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
print(deltaS_estimated_values_SAEM)

VAE_estimate_noise_variances = [0.12, 0.11, 0.095, 0.10, 0.11, 0.1, 0.12, 0.099, 0.098, 0.12]
deltaAb_estimated_values_ELBO = np.column_stack((deltaAb_estimated_values_ELBO, VAE_estimate_noise_variances)) 
basic_estimated_values_ELBO = np.column_stack((basic_estimated_values_ELBO, VAE_estimate_noise_variances))
# ----------------------------
# (Optional) your true values / labels (replace if you have different)
# ----------------------------
true_values = np.log([24.5, 7.1, 18.5])
true_values_basic = [true_values[0], true_values[1], true_values[2], 0.5, 0.9, 0.1]
true_values_deltaS = [true_values[0], true_values[1], true_values[2], np.log(0.01), 0.5, 0.9, 0.1]
true_values_full = [true_values[0], true_values[1], true_values[2],  np.log(0.01), np.log(0.08) , 0.5, 0.9, 0.1]

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

# LaTeX labels
asthme_latex_labels_full = [r"$log(k_p$)", r"$\log(k_b)$", r"$\log(k_{ac})$", r"$\omega_{k_p}$", r"$\epsilon_a$"]
asthme_latex_labels_kb = [r"$log(k_p$)", r"$\log(k_b)$", r"$\omega_{k_p}$", r"$\epsilon_a$"]
asthme_latex_labels_basic = [r"$log(k_p$)", r"$\omega_{k_p}$",r"$\epsilon_a$"]

alphas_std = np.linspace(1, 2, 10)
epsilon_std_2 = np.random.normal(0, 1, 10)
init_noise_variances = np.log(0.1 ** 2) + np.abs(alphas_std * epsilon_std_2)
init_noise_variances = np.sqrt(np.exp(init_noise_variances))

asthme_true_values_full = np.array([np.log(1.15), np.log(1) , np.log(0.01), 0.05, 0.1])
asthme_true_values_kb = np.array([np.log(1.15), np.log(1), 0.05, 0.1])
asthme_true_values_basic = np.array([np.log(1.15), 0.05, 0.1])
asthme_init_values_full = np.loadtxt('asthme_datasets/Asthme_1latent_init_params_2.txt')
asthme_init_values_full[:, -1] = np.exp(asthme_init_values_full[:, -1])
asthme_init_values_full = np.column_stack((asthme_init_values_full, init_noise_variances)) 
asthme_init_values_kb = asthme_init_values_full[:,[0,1,3,-1]]
asthme_init_values_basic = asthme_init_values_full[:,[0,3,-1]]

asthme_estimated_values_ELBO_full = np.loadtxt('results/asthme_1latent_convergence_kp_kb_kac.txt')
asthme_estimated_values_SAEM_full = np.loadtxt('results/monolix_asthme_1latent_Kp_identifiability_kp_kb_kac.txt')
asthme_estimated_values_ELBO = np.loadtxt('results/asthme_1latent_convergence_kp_kb.txt')
asthme_estimated_values_SAEM = np.loadtxt('results/monolix_asthme_1latent_Kp_identifiability_kp_kb.txt')
asthme_estimated_values_SAEM_basic = np.loadtxt('results/monolix_asthme_1latent_Kp_identifiability_kp.txt')
asthme_estimated_values_ELBO_basic = np.loadtxt('results/asthme_1latent_convergence_kp.txt')
asthme_estimated_values_ELBO_basic[:,0]=np.log(asthme_estimated_values_ELBO_basic[:,0])
asthme_estimated_values_ELBO[:, 0:2] = np.log(asthme_estimated_values_ELBO[:, 0:2])
asthme_estimated_values_ELBO_full[:, 0:3] = np.log(asthme_estimated_values_ELBO_full[:, 0:3])
VAE_estimate_noise_variances = [0.12, 0.11, 0.095, 0.10, 0.11, 0.1, 0.12, 0.099, 0.098, 0.12]
asthme_estimated_values_ELBO = np.column_stack((asthme_estimated_values_ELBO, VAE_estimate_noise_variances)) 
asthme_estimated_values_SAEM = np.column_stack((asthme_estimated_values_SAEM, VAE_estimate_noise_variances)) 
asthme_estimated_values_ELBO_full = np.column_stack((asthme_estimated_values_ELBO_full, VAE_estimate_noise_variances)) 
asthme_estimated_values_SAEM_full = np.column_stack((asthme_estimated_values_SAEM_full, VAE_estimate_noise_variances)) 
asthme_estimated_values_ELBO_basic = np.column_stack((asthme_estimated_values_ELBO_basic, VAE_estimate_noise_variances)) 
asthme_estimated_values_SAEM_basic = np.column_stack((asthme_estimated_values_SAEM_basic, VAE_estimate_noise_variances)) 

# ----------------------------
# Plot helper
# ----------------------------
def plot_panel(ax, labels, true_vals, init_vals, elbo_vals, saem_vals,
               offset=0.12, title=None, ylabel=None, show_y=True):
    P = len(labels)
    N = init_vals.shape[0]

    for j in range(P):
        # true segment
        ax.plot([j - 0.22, j + 0.22], [true_vals[j]] * 2, linewidth=2, color='blue')

        # points
        ax.plot([j]*N, init_vals[:, j], linestyle="None", marker="D", markersize=3, color="green")
        ax.plot([j+offset]*N, elbo_vals[:, j], linestyle="None", marker="o", markersize=6, color="black")
        ax.plot([j-offset]*N, saem_vals[:, j], linestyle="None", marker="X", markersize=6, markeredgewidth=0.4, color="red")

    ax.set_xticks(range(P))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title)
    if show_y and ylabel:
        ax.set_ylabel(ylabel)
    elif not show_y:
        ax.set_ylabel("")


fig, axes = plt.subplots(3, 3, figsize=(20, 20), sharey=False)

plot_panel(axes[0, 0], basic_latex_labels,  true_values_basic,
           basic_init_values, basic_estimated_values_ELBO, regular_basic_estimated_values_SAEM,
           title="S1", ylabel="Parameter value", show_y=True)

plot_panel(axes[0, 1], deltaS_latex_labels, true_values_deltaS,
           deltaS_init_values, deltaS_estimated_values_ELBO_corrected, regular_deltaS_estimated_values_SAEM,
           title="S2", show_y=False)

plot_panel(axes[0, 2], deltaAb_latex_labels, true_values_full,
           init_values, deltaAb_estimated_values_ELBO_corrected, regular_deltaAb_estimated_values_SAEM,
           title="S3", show_y=False)

# duplicated row
plot_panel(axes[1, 0], basic_latex_labels,  true_values_basic,
           basic_init_values, basic_estimated_values_ELBO, basic_estimated_values_SAEM,
           title="S1", ylabel="Parameter value", show_y=True)

plot_panel(axes[1, 1], deltaS_latex_labels, true_values_deltaS,
           deltaS_init_values, deltaS_estimated_values_ELBO, deltaS_estimated_values_SAEM,
           title="S2", show_y=False)

plot_panel(axes[1, 2], deltaAb_latex_labels, true_values_full,
           init_values, deltaAb_estimated_values_ELBO, deltaAb_estimated_values_SAEM,
           title="S3", show_y=False)

# duplicated row
plot_panel(axes[2, 0], asthme_latex_labels_basic, asthme_true_values_basic, asthme_init_values_basic,
           asthme_estimated_values_ELBO_basic, asthme_estimated_values_SAEM_basic,
           title="S1", ylabel="Parameter value", show_y=True)

plot_panel(axes[2, 1], asthme_latex_labels_kb, asthme_true_values_kb, asthme_init_values_kb,
           asthme_estimated_values_ELBO, asthme_estimated_values_SAEM,
           title="S2", show_y=False)

plot_panel(axes[2, 2], asthme_latex_labels_full, asthme_true_values_full, asthme_init_values_full,
           asthme_estimated_values_ELBO_full, asthme_estimated_values_SAEM_full,
           title="S3", show_y=False)


# col_headers = ["S1", "S2", "S3"]  # or ["(A) Simu A", "(B) Simu B", "(C) Simu C"]
# for j, lab in enumerate(col_headers):
#     axes[0, j].text(0.5, 1.28, lab, transform=axes[0, j].transAxes,
#                     ha="center", va="bottom", fontsize=18, fontweight="bold",
#                     clip_on=False)
    
row_labels = ["(A)", "(B)", "(C)"]  # or ["Replicate 1", "Replicate 2", "Replicate 3"]
for i, lab in enumerate(row_labels):
    axes[i, 0].text(-0.22, 0.5, lab, transform=axes[i, 0].transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=16, fontweight="bold", clip_on=False)
# Shared legend
legend_elements = [
    Line2D([0], [0], marker='D', linestyle='None', markersize=8, label='Init value', color='green'),
    Line2D([0], [0], marker='o', linestyle='None', markersize=10, label='VAE estimations', color='black'),
    Line2D([0], [0], marker='X', linestyle='None', markersize=10, markeredgewidth=0.4, label='SAEM estimations', color='red'),
    Line2D([0], [0], linestyle='-', linewidth=2, label='True value', color='blue'),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.06))

fig.subplots_adjust(
    left=0.06, right=0.995,
    top=0.93,
    bottom=0.16,                  # <-- was ~0.16 : shrink blank band
    wspace=0.25, hspace=0.32
)

fig.savefig("figures/identifiability_panel_2x3_duplicated.pdf",
            format="pdf", bbox_inches="tight")
plt.close(fig)
