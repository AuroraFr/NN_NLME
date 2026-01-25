import pandas as pd
import numpy as np

def convert_data_format_monolix(L, timepoints, n_subjects, monolix_data_dir, datafile=None, data=None, full=True, doses=1, irregular=False):
    if datafile:
        file_type = datafile.split('.')[-1]
        if file_type == 'txt':
            data = pd.read_csv(datafile, sep=" ", header=None).values.reshape(n_subjects, timepoints, -1)[:,:,0]
        
        elif file_type == 'npy':
            if full:
                data = np.load(datafile).reshape(n_subjects, timepoints, 2)[:,:,1]
            else:
                data = np.load(datafile)

    dataframes = []
    for i in range(data.shape[0]):
        if doses == 1:
            dataframe = pd.DataFrame(columns=['Id', 'Time','Observation'])
        elif doses == 2:
            dataframe = pd.DataFrame(columns=['Id', 'Time', 'second_dose','Observation'])
        elif doses == 3:
            dataframe = pd.DataFrame(columns=['Id', 'Time', 'second_dose','third_dose','Observation'])

        # t_early = np.linspace(0, 20, 20)  # More points in the early stage
        # t_late = np.linspace(20, 100, 10)  # Fewer points in the later stage
        # times = np.unique(np.concatenate((t_early, t_late)))
        if irregular:
            for time_index, time in enumerate(timepoints[i]):
                if doses == 1:
                    list_row = [i+1, time, data[i, time_index]]
                elif doses == 3:
                    list_row = [i+1, time, 30, 250, data[i, time_index]]
                dataframe.loc[len(dataframe)] = list_row
        else:
            for time_index, time in enumerate(timepoints):
                if doses == 1:
                    list_row = [i+1, time, data[i, time_index]]
                elif doses == 3:
                    list_row = [i+1, time, 30, 250, data[i, time_index]]
                dataframe.loc[len(dataframe)] = list_row
        
        dataframes.append(dataframe)

    df = pd.concat(dataframes)
    df.to_csv(monolix_data_dir+'/'+str(L)+'.txt', sep=';', index=False)


if __name__ == "__main__":

    irregular = True

    for L in range(100):
        n_subjects = 100
        timepoints = 11
        T = 400

        if not irregular:
            datafile = 'antibody_datasets/scipy_antibody_3dose/10_400_50_2latent_theta_F2/dataset_'+str(L)+'.npy'
            times = np.linspace(0,T,timepoints)
            times= np.array([0, 20, 45, 85, 130, 200, 260, 310, 350, 400])
            times= np.array([0, 20, 45, 65, 85, 110, 130, 160, 200, 240, 270, 310, 350, 380, 400])
        else:
            # datafile = 'PKPD_datasets/'+str(timepoints)+'_irregular_1latent_theta1/dataset_'+str(L)+'.npy'
            # timefile = 'PKPD_datasets/irregular_'+str(timepoints)+'_400_50_1latent_Kp/dataset_timepoints_'+str(L)+'.npy'
            # times = np.load(timefile)
            # print("times shape", times.shape)

            data = np.load('PKPD_datasets/'+str(timepoints)+'_irregular_1latent_theta1/dataset_'+str(L)+'.npy', allow_pickle=True)
            df = pd.DataFrame(data.tolist())
            y = np.stack(df['Y'].to_numpy())
            times = np.stack(df['t'].to_numpy())
        
        data_folder = '/beegfs/zli/workspace/monolix/PKPD_datasets/irregular_'+str(timepoints)+'_10_100_1latent_theta1/'
        from pathlib import Path
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        convert_data_format_monolix(L, times, n_subjects, data_folder, datafile=None, data=y, full=False, doses=1, irregular=irregular)
