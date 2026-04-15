import pandas as pd
import numpy as np
import os
import h5py

dir_osd_output = '/d/cdac Dropbox/Wolfgang Ganglberger/WolfgangGanglberger/CLAS_Teun_Project/OSD_predictions/'
dir_prepared = '/d/cdac Dropbox/Wolfgang Ganglberger/WolfgangGanglberger/CLAS_Teun_Project/prepared_data/'

files = [f for f in os.listdir(dir_osd_output) if f.endswith('.csv')]
fileids = [f.replace('.csv','') for f in files]

for fileid in fileids:
    osd_output = f'/d/cdac Dropbox/Wolfgang Ganglberger/WolfgangGanglberger/CLAS_Teun_Project/OSD_predictions/{fileid}.csv'
    df_osd = pd.read_csv(osd_output)
    df_osd = df_osd[['OSD']]

    # describe OSD values
    print(fileid, df_osd.describe())
    
    eeg = f'/d/cdac Dropbox/Wolfgang Ganglberger/WolfgangGanglberger/CLAS_Teun_Project/prepared_data/{fileid}.h5'
    eeg = h5py.File(eeg, 'r')
    len_eeg = len(eeg['signals']['c4-m1'][:])

    # OSD gives a value each 600 samples (i.e. 3 seconds at 200 Hz)
    # scale df_osd accordingly, i.e. repeat each value 600 times
    df_osd_expanded = np.repeat(df_osd['OSD'].values, 600)
    # if shape is off by less than 600, fill with 0s
    if len_eeg > len(df_osd_expanded):
        diff = len_eeg - len(df_osd_expanded)
        assert diff < 600
        df_osd_expanded = np.concatenate([df_osd_expanded, np.zeros(diff)])
    assert len_eeg == len(df_osd_expanded)
    
    # save as csv
    df_osd_final = pd.DataFrame({'OSD': df_osd_expanded})
    df_osd_final.to_csv(f'/d/cdac Dropbox/Wolfgang Ganglberger/WolfgangGanglberger/CLAS_Teun_Project/OSD_predictions/{fileid}.csv', index=False)
    print(f'All good, saved.')