#Pipeline to run VAE on real antibody kinetic data
import pandas as pd
import real_ab_train
from FIM_antibody_multidose_real import variance_estimation
import jax.numpy as jnp
from pathlib import Path
import numpy as np
from scipy.stats import norm
from real_ab_prediction import prediction_interval

timepoints = 17
b_sample_size = 8000

#Run Monolix to save the estimation results

df = pd.read_csv('antibody_datasets/real_ab_data_origin.csv')
df["id"] = df["id"].astype(int)
df["Time"] = df["Time"].astype(float)
df["ED50_BAU"] = df["ED50_BAU"].astype(float)
df = df.dropna()
data = real_ab_train.preprocess_data(df)
N_subjects = data['padded_y'].shape[0]
print(N_subjects)
#train the model and return the estimated params
model, VAE_params = real_ab_train.main(data)
print(VAE_params)

#variance
VAE_variance_matrix = variance_estimation(data, jnp.array(VAE_params), N_subjects, b_sample_size=b_sample_size)
result_folder = 'EXPs/variance_real_ab_'+str(b_sample_size)

Path(result_folder).mkdir(parents=True, exist_ok=True)
np.save(result_folder+'/variance_matrix', VAE_variance_matrix)

#CI
alpha = 0.05
# Standard errors
se = np.sqrt(np.diag(VAE_variance_matrix))

# Normal quantile
z = norm.ppf(1 - alpha / 2)
lower = VAE_params - z * se
upper = VAE_params + z * se
CI_df = pd.DataFrame({
        "param": VAE_params,
        "lower": lower,
        "upper": upper,
        "se": se
    })
CI_df.to_csv("result_summaries/CI_real_ab.csv", index=False)

#prediction interval
SAEM_params = jnp.array([3.62, 1.35, 2.62, -4.72, -1.97]) 
SAEM_stds = jnp.array([0.19, 0.14, 0.12, 0.067, 0.23])
prediction_interval(model, df, VAE_params, VAE_variance_matrix, SAEM_params, SAEM_stds)