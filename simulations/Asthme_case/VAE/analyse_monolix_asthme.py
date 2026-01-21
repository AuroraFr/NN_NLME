import numpy as np
import pandas as pd
df = pd.DataFrame(columns=['parameter', 'mean','RRMSE', 'RBias','Variance'])
folder = '/beegfs/zli/workspace/monolix/asthme_1latent_identifiability_kp_kb_kac/'
folder = '/beegfs/zli/workspace/monolix/asthme_1latent_results/'
folder = '/beegfs/zli/workspace/monolix/asthme_1latent_identifiability_kp_kb/'
folder = '/beegfs/zli/workspace/monolix/asthme_1latent_results_kp/'
folder = '/beegfs/zli/workspace/monolix/asthme_1latent_identifiability_kp/'
true_Kp = np.log(1.15)
true_Kb = 1
true_omega_Kp = 0.05

Kps = []
Kbs = []
Kacs = []
omega_Kps = []
parameters = []

for i in range(1, 10):
    data = pd.read_csv(folder+'asthme_param_'+str(i)+'.csv')
    print(data)
    Kp = data.iloc[0].values[0]
    # Kb = data.iloc[1].values[0]
    # Kac = data.iloc[2].values[0]
    omega_Kp = data.iloc[3].values[0]
    parameters.append([Kp, omega_Kp])

    Kps.append(Kp)
    # Kbs.append(Kb)
    # Kacs.append(Kac)
    omega_Kps.append(omega_Kp)
parameters = np.array(parameters)
np.savetxt('results/monolix_asthme_1latent_Kp_identifiability_kp.txt', parameters)
# for i in range(1, 100):
#     data = pd.read_csv(folder+'asthme_param_'+str(i)+'.csv')
#     print(data)
#     Kp = data.iloc[0].values[0]
#     Kb = data.iloc[1].values[0]
#     # Kac = data.iloc[2].values[0]
#     omega_Kp = data.iloc[3].values[0]
#     parameters.append([Kp, Kb, omega_Kp])

#     Kps.append(Kp)
#     # Kbs.append(Kb)
#     # Kacs.append(Kac)
#     omega_Kps.append(omega_Kp)

# parameters = np.array(parameters)
# print(parameters)
# Kps = np.array(Kps)
# # Kbs = np.array(np.exp(Kbs))
# omega_Kps = np.array(omega_Kps)

# rrmse_Kp = np.sqrt(np.mean((Kps - true_Kp)**2)) / true_Kp
# # rrmse_Kb = np.sqrt(np.mean((Kbs - true_Kb)**2)) / true_Kb
# rrmse_omega_Kp = np.sqrt(np.mean((omega_Kps - true_omega_Kp)**2)) / true_omega_Kp

# bias_Kp = np.mean(Kps - true_Kp)/true_Kp 
# # bias_Kb = np.mean(Kbs - true_Kb)/true_Kb
# bias_omega_Kp = np.mean(omega_Kps - true_omega_Kp)/true_omega_Kp

# variance_Kp = np.var(Kps)
# # variance_Kb = np.var(Kbs)
# variance_omega_Kp = np.var(omega_Kps)

# df.loc[len(df)] = {'parameter':'pop_Kp', 'mean':np.mean(Kps),'RRMSE':rrmse_Kp, 'RBias': bias_Kp, 'Variance':variance_Kp}
# # df.loc[len(df)] = {'parameter':'pop_Kb', 'mean':np.mean(Kbs),'RRMSE':rrmse_Kb, 'RBias': bias_Kb, 'Variance':variance_Kb}
# df.loc[len(df)] = {'parameter':'omega_Kp', 'mean':np.mean(omega_Kps),'RRMSE':rrmse_omega_Kp, 'RBias': bias_omega_Kp, 'Variance':variance_omega_Kp}
# print(df)
# np.savetxt('results/monolix_asthme_1latent_Kp_50s_kp.txt', parameters)
# df.to_csv('results/monolix_asthme_1latent_Kp_quality_kp')
