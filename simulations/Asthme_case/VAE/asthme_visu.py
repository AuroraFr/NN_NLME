import numpy as np
asthme_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_1.npy')
nonoise_asthme_data = np.load('asthme_datasets/20_400_50_1latent_Kp/dataset_nonoise_1.npy')

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
indices = list(range(10))
t_eval = np.linspace(0, 400, 20)
for i in indices:
    plt.scatter(t_eval[:], asthme_data[i,:], alpha=0.6)
    plt.plot(t_eval[:], np.log10(nonoise_asthme_data[i, 1, :]))
    # plt.plot(t_eval[:], nonoise_asthme_data[i, 0, :])
plt.xlabel("Time (Days)")
plt.ylabel(r"log$_{10}$ $a(t)$-number of active TGF-$\beta$ per unit volume")
plt.tight_layout()
plt.savefig('figures/asthme_1latent_Kp.pdf')
