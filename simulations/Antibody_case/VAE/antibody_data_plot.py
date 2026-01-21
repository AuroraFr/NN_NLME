import numpy as np
import matplotlib.pyplot as plt
irregular = False

subjects = 5
duration = 400
measurements = 15

# data_folder = "antibody_datasets/scipy_antibody_3dose_1/"

if not irregular:
    data_folder = "antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/"

    dataset = np.load(data_folder+"dataset_0.npy")
    dataset_nonoise = np.load(data_folder+"dataset_nonoise_0.npy")

    # data_folder_2 = "antibody_datasets/scipy_antibody_3dose_15_400_50/"
    # dataset_2 = np.load(data_folder_2+"dataset0.npy")
    # print(dataset_2.shape)
    
    timepoints = np.linspace(0, 400, measurements)
    # timepoints = np.array([0, 20, 45, 85, 130, 200, 260, 310, 350, 400])
    for i in range(5):
        # print(dataset[i, 1, :].shape, dataset_nonoise[i, :, 1].shape)
        plt.scatter(timepoints, 10 ** dataset[i, :, 1])
        # plt.scatter(timepoints, dataset_2[i, :])
        plt.plot(timepoints, 10 ** dataset_nonoise[i, :, 1])
    
    plt.axvline(30, color='grey', alpha=0.3, linestyle='--', linewidth=1)
    plt.axvline(250, color='black', alpha=0.8, linestyle='--', linewidth=1)
    plt.xlabel("Time (Days)")
    plt.ylabel("Anti-S IgG (BAU/ml)")
    plt.tight_layout()

    plt.savefig('figures/antibody_'+str(measurements)+'_50_400_3doses.pdf')

else:
    
    data_folder = "antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/"
    dataset = np.load(data_folder+"dataset_1.npy")
    dataset_nonoise = np.load(data_folder+"dataset_nonoise_1.npy")
    timepoints = np.load(data_folder+"dataset_timepoints_1.npy")
    rng = np.random.default_rng()
    nums = rng.integers(0, 50, size=5)

    for i in nums:
        print(dataset[i, 1, :].shape, dataset_nonoise[i, :, 1].shape)
        plt.scatter(timepoints[i], 10 ** dataset[i, :, 1])
        plt.plot(timepoints[i], 10 **  dataset_nonoise[i, :, 1])
    
    plt.axvline(30, color='grey', alpha=0.3, linestyle='--', linewidth=1)
    plt.axvline(250, color='black', alpha=0.8, linestyle='--', linewidth=1)
    plt.xlabel("Time (Days)")
    plt.ylabel("Anti-S IgG (BAU/ml)")
    plt.tight_layout()

    plt.savefig('figures/antibody_irregular_'+str(measurements)+'_50_400_3doses.pdf')
    