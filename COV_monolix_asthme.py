import numpy as np
import pandas as pd

folder = '/beegfs/zli/workspace/monolix/asthme_1latent_results_kp/'
parameters = []
std_dfs = []
omega_Kp_count = 0
Kp_count = 0
Kb_count = 0


for i in range(100):

    std_df = pd.read_csv(folder+'asthme_std_'+str(i)+'.csv')['stochasticApproximation.se']
    param_df = pd.read_csv(folder+'asthme_param_'+str(i)+'.csv')
    parameter = param_df.values
    Kp = parameter[0]
    Kb = parameter[1]
    omega_Kp = parameter[3]
    std = std_df.values
    
    if Kp - 1.96 * std[0] <= np.log(1.15) <= Kp + 1.96 * std[0]:
        Kp_count += 1
    
    
    # if Kb - 1.96 * std[1] <= np.log(1) <= Kb + 1.96 * std[1]:
    #     Kb_count += 1
    # else:
    #     print(Kb, Kb - 1.96 * std[1], Kb + 1.96 * std[1])
    
    if omega_Kp - 1.96 * std[1] <= 0.05 <= omega_Kp + 1.96 * std[1]:
        omega_Kp_count += 1

    std_dfs.append(std)

print(Kp_count/100, omega_Kp_count/100)
print(np.mean(np.array(std_dfs)**2, axis=0))