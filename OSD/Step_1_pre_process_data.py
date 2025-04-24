import mne
from mne.preprocessing import EOGRegression
import os 
from os.path import join as opj
import h5py as h5
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from utils.pre_processing.prepare_data import process_file


os.environ["CUDA_VISIBLE_DEVICES"]="0"

##############################
######## definitions #########
##############################

def pre_process_data(data):

    #add zeros for referencing
    data_extra = np.zeros((data.shape[0]+1,data.shape[1]))
    data_extra[:data.shape[0],:] = data

    ####################
    # filter EKG 
    ####################

    # create mne file 
    info  = mne.create_info(ch_names=['F3','F4','C3','C4','O1','O2','ECG','M1'],
                            sfreq=200,
                            ch_types=['eeg','eeg','eeg','eeg','eeg','eeg','ecg','eeg']
                            )
    raw = mne.io.RawArray(data_extra, info, first_samp=0, verbose=None)

    #reference with zeros, (already referenced)
    raw.set_eeg_reference(ref_channels=['M1'])
    #filter
    raw = raw.copy().filter(l_freq=0.3, h_freq=30)
    #make montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)

    #set epochs
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    #We need to explicitly specify that we want to average the EOG channel too.
    ecg_evoked = ecg_epochs.average('all')

    # perform regression on the evoked blink response
    model_evoked = EOGRegression(picks='eeg', picks_artifact='ecg').fit(ecg_evoked)
    ecg_evoked_clean = model_evoked.apply(ecg_evoked)
    ecg_evoked_clean.apply_baseline()
    raw_clean = model_evoked.apply(raw)

    #back to numpy
    data = raw_clean.get_data()

    #delete zeros/reference channel
    data = data[:-2,:]

    return data


#@ray.remote
def create_OSD_FILES(file,write_path):
    try:
        
        file_path = write_path+file.split('/')[-1].split('.')[0]+'/'+file.split('/')[-1]
        file_path = file_path.replace('_task-psg_eeg','')
        if not os.path.exists(file_path):

            #define variables
            channel_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1','ecg']
            
            #read in data
            f = h5.File(file, 'r')
            data = np.zeros((len(channel_names),len(f['signals'][channel_names[0]])))
            for i,c in enumerate(channel_names):
                data[i,:] = np.squeeze(np.array(f['signals'][c]))
            f.close()
                
            #define label names
            channel_names = ['stage','arousal']

            #read in data
            f = h5.File(file, 'r')
            label = np.zeros((len(channel_names),len(f['annotations'][channel_names[0]])))
            for i,c in enumerate(channel_names):
                label[i,:] = np.squeeze(np.array(f['annotations'][c]))
            f.close()

            data_pre_prob = pre_process_data(data)
            data_pre = np.vstack((data_pre_prob,label))

            #write pre-processed data
            pat = file.split('/')[-1]
            dtypef = 'float32'
            #write data
            if not os.path.exists(write_path+pat.split('.')[0]+'/'):
                os.makedirs(write_path+pat.split('.')[0]+'/')

            hf = h5.File(write_path+pat.split('.')[0]+'/'+pat, 'w')
            hf.attrs['sample_rate']=200
            hf.attrs['Unit_size']='V'
            #write datae
            hf.create_dataset('channels/F3-M2', data=data_pre[0,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('channels/F4-M1', data=data_pre[1,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('channels/C3-M2', data=data_pre[2,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('channels/C4-M1', data=data_pre[3,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('channels/O1-M2', data=data_pre[4,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('channels/O2-M1', data=data_pre[5,:],dtype=dtypef, compression="gzip")
            hf.create_dataset('annotations/stage_expert_0', data=data_pre[6,:],dtype='uint32', compression="gzip")
            hf.create_dataset('annotations/arousal_expert_0', data=data_pre[7,:],dtype='uint32', compression="gzip")
            #hf.create_dataset('annotations/arousal-shifted_converted_0', data=data_pre[8,:],dtype='uint32', compression="gzip")
            hf.close()
            
            print(f'Finished {file}')
        return None


    except:

        return None
    
def find_edf_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.edf'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files


#######################
# Define Files
#######################  

#TODO: CHANGE WORKING_DIRECTORY
WORKING_DIRECTORY = '/your/path/to/OSD_repo'
Process_ALL = True
subset_CSV_path =f'{WORKING_DIRECTORY}/ODS/utils/data_split/train_split.csv'

#define paths
write_path_edf = f'{WORKING_DIRECTORY}/OSD_data/prepared/'
write_path_h5 = f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/'

#######################
#convert to basic H5
#######################
edf_files = find_edf_files(WORKING_DIRECTORY)
[process_file(path_input.replace('.edf',''),opj(write_path_edf,path_input.split('/')[-1].replace('_task-psg_eeg','')).replace('.edf','.h5'), add_existing_annotations=True) for path_input in edf_files]


#grab files
files = glob(f'{write_path_edf}*.h5')

#FOR SUBSET 
if Process_ALL:
    files_To_process = files
else:
    df_test_sub = pd.read_csv(subset_CSV_path)
    test_subjects = df_test_sub['fileid'].values
    files_To_process = [x for x in files if x.split('/')[-1][:-3] in test_subjects]

#######################
# RUN preprocessing specific for OSD
#######################  

f = [create_OSD_FILES(file,write_path_h5) for file in tqdm(files_To_process)]
