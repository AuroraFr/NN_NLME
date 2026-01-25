import numpy as np
from pathlib import Path

for i in range(100):
    # Path to your txt file
    txt_path = "/beegfs/zli/workspace/monolix/antibody_datasets/15p_400d_50s_3dose_2latents_theta_F2/"+str(i)+".txt"

    # 1. Load the numeric data (skip header, use ; as separator)
    data = np.loadtxt(txt_path, delimiter=";", skiprows=1)

    # data shape is (50 * 15, 5)
    # Columns: 0=Id, 1=Time, 2=second_dose, 3=third_dose, 4=Observation

    # 2. Take the Observation column
    obs = data[:, 4]   # last column

    # 3. Reshape to (50, 15)
    n_subjects = 50
    n_times = 15
    assert obs.size == n_subjects * n_times, "Size mismatch, check file length or n_subjects/n_times."

    obs_matrix = obs.reshape(n_subjects, n_times)

    # 4. Save to .npy
    dataset_folder = "antibody_datasets/scipy_antibody_3doses_1"
    Path(dataset_folder).mkdir(parents=True, exist_ok=True)
    np.save(dataset_folder+"/dataset"+str(i)+".npy", obs_matrix)
