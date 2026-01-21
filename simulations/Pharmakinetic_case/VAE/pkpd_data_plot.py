import numpy as np
import matplotlib.pyplot as plt

# simulation setup
num_points = 5

# ----------------------------------------------------
# 4. PLOT (one panel per subject)
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

data = np.load('PKPD_datasets/'+str(num_points)+'_irregular_1latent_theta1/dataset_2.npy', allow_pickle=True)
rng = np.random.default_rng()
nums = rng.integers(0, 50, size=10)
count = 0
for i in nums:
    t = data[i]["t"]
    X1 = data[i]["X1"]
    Y = data[i]["Y"]

    # true line (thin)
    plt.plot(t, X1, alpha=0.7)

    # noisy points
    plt.scatter(t, Y, s=20, alpha=0.9)


plt.xlabel("Time")
plt.ylabel("APV Plasma Concentration (ng/mL)")
plt.tight_layout()
plt.savefig('figures/pkpd_'+str(num_points)+'_irregular.pdf')