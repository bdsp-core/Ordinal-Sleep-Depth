import os
from os.path import join as opj
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py as h5
import gc
import tensorflow as tf
from utils.models.OSD_architecture import OSD_architecture
from utils.pre_processing.prepare_data import process_file
import mne
from mne.preprocessing import EOGRegression

# ------------- CONFIGURATION ----------------
WORKING_DIRECTORY = '/home/wolfgang/repos/Ordinal-Sleep-Depth'
PROCESS_ALL = True
SPLIT = 'test'  # can be train, val, or test
BATCH_SIZE = 32
MODEL_VERSION = 'OSD_MODEL'
CUDA_DEVICE = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

WRITE_PATH_EDF = f'{WORKING_DIRECTORY}/OSD_data/prepared/'
WRITE_PATH_H5 = WRITE_PATH_EDF # f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/'
WEIGHTS_PATH = f'{WORKING_DIRECTORY}/OSD/utils/models/weights_scaler/OSD_weigths.h5'
CSV_SPLIT_PATH = f'{WORKING_DIRECTORY}/OSD/utils/data_split/{SPLIT}.csv'
PREDICTION_WRITE_PATH = f'{WORKING_DIRECTORY}/OSD_predictions/{MODEL_VERSION}/{SPLIT}/'

os.makedirs(WRITE_PATH_EDF, exist_ok=True)
os.makedirs(WRITE_PATH_H5, exist_ok=True)
os.makedirs(PREDICTION_WRITE_PATH, exist_ok=True)

# ------------- STEP 1: EDF to Basic H5 ----------------
def find_edf_files(directory):
    return [os.path.join(root, file)
            for root, dirs, files in os.walk(directory)
            for file in files if file.endswith('.edf')]

def convert_edf_to_h5():
    edf_files = find_edf_files(WORKING_DIRECTORY)
    for path_input in edf_files:
        output_path = opj(WRITE_PATH_EDF, path_input.split('/')[-1].replace('_task-psg_eeg','')).replace('.edf','.h5')
        process_file(path_input.replace('.edf',''), output_path, add_existing_annotations=True)

# ------------- STEP 2: Preprocess for OSD ----------------
def pre_process_data(data):
    data_extra = np.zeros((data.shape[0]+1, data.shape[1]))
    data_extra[:data.shape[0], :] = data
    info = mne.create_info(
        ch_names=['F3','F4','C3','C4','O1','O2','ECG','M1'],
        sfreq=200,
        ch_types=['eeg']*6 + ['ecg', 'eeg']
    )
    raw = mne.io.RawArray(data_extra, info, verbose=None)
    raw.set_eeg_reference(ref_channels=['M1'])
    raw = raw.copy().filter(l_freq=0.3, h_freq=30)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    # Skip ECG correction if fake ECG
    try:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
        ecg_evoked = ecg_epochs.average('all')
        model_evoked = EOGRegression(picks='eeg', picks_artifact='ecg').fit(ecg_evoked)
        raw_clean = model_evoked.apply(raw)
    except Exception as e:
        raw_clean = raw
        
    data = raw_clean.get_data()
    return data[:-2, :]

def create_OSD_FILES(file, write_path):
    # try:
    if 1:
        out_file = write_path + file.split('/')[-1].split('.')[0] + '/' + file.split('/')[-1]
        out_file = out_file.replace('_task-psg_eeg', '')
        if not os.path.exists(out_file):
            channel_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1','ecg']
            with h5.File(file, 'r') as f:
                signals_group = f['signals']
                data = np.stack([
                    np.squeeze(np.asarray(signals_group[ch]))
                    for ch in channel_names
                ])

                # Default labels if prepared file does not provide annotations
                label_stage = np.ones(data.shape[1], dtype=np.uint8)
                label_arousal = np.zeros(data.shape[1], dtype=np.uint8)

                annotations_group = f.get('annotations')
                if annotations_group is not None:
                    if 'stage' in annotations_group:
                        label_stage = np.squeeze(np.asarray(annotations_group['stage'])).astype(np.uint8)
                    if 'arousal' in annotations_group:
                        label_arousal = np.squeeze(np.asarray(annotations_group['arousal'])).astype(np.uint8)

                target_len = data.shape[1]

                def _match_length(arr, target, fill_value):
                    if arr.shape[0] < target:
                        pad_width = target - arr.shape[0]
                        return np.pad(arr, (0, pad_width), constant_values=fill_value)
                    return arr[:target]

                label_stage = _match_length(label_stage, target_len, fill_value=1)
                label_arousal = _match_length(label_arousal, target_len, fill_value=0)

                label = np.vstack((label_stage, label_arousal))

            print(f"Original data shape: {data.shape}")
            print(f"Original label shape: {label.shape}")
            data_pre = np.vstack((pre_process_data(data), label))
            print(f"data_pre shape: {data_pre.shape}")
            os.makedirs(write_path + file.split('/')[-1].split('.')[0], exist_ok=True)
            with h5.File(out_file, 'w') as hf:
                hf.attrs['sample_rate'] = 200
                hf.attrs['Unit_size'] = 'V'
                for i, ch in enumerate(['F3-M2','F4-M1','C3-M2','C4-M1','O1-M2','O2-M1']):
                    hf.create_dataset(f'channels/{ch}', data=data_pre[i,:], dtype='float32', compression='gzip')
                hf.create_dataset('annotations/stage_expert_0', data=data_pre[6,:], dtype='uint32', compression='gzip')
                hf.create_dataset('annotations/arousal_expert_0', data=data_pre[7,:], dtype='uint32', compression='gzip')
    # except Exception as e:
    #     print(f"Error processing {file}: {e}")

# ------------- STEP 3: Run Model ----------------
def run_model_prediction():
    params = pd.read_pickle(f'{WORKING_DIRECTORY}/OSD/utils/scaling_osd/scaling_values.pkl')
    model = OSD_architecture()
    model.load_weights(WEIGHTS_PATH)
    files = sorted(glob(os.path.join(WRITE_PATH_H5, '*/*.h5')))
    if not PROCESS_ALL:
        subset_ids = set(pd.read_csv(CSV_SPLIT_PATH)['fileid'].tolist())
        files = [
            f for f in files
            if os.path.splitext(os.path.basename(f))[0] in subset_ids
        ]
    count = 0
    for file in tqdm(files):
        file_id = os.path.splitext(os.path.basename(file))[0]
        out_csv = opj(PREDICTION_WRITE_PATH, f'{file_id}.csv')
        if os.path.exists(out_csv): continue
        try:
            with h5.File(file, 'r') as f:
                label = f['annotations']['stage_expert_0'][:]
                arousal = f['annotations']['arousal_expert_0'][:]
                eeg = np.stack([f['channels'][ch][:] for ch in ['F3-M2','F4-M1','C3-M2','C4-M1','O1-M2','O2-M1']])
            eeg_fit = eeg[:, :eeg.shape[1]//600*600]
            ordinal = model.predict(eeg_fit.reshape(6,600,-1,order='F').T)
            ordinal_ = ordinal[1]
            ord_pred = np.argmax(np.array(ordinal_), axis=1) + 1
            OSD = [x[3] for x in ordinal_]
            OSD = (OSD-params['min'])/params['max']

            label = label[::600][:len(ord_pred)]
            arousal = [np.max(arousal[x:x+600]) for x in np.arange(0, len(arousal), 600) if len(arousal)-x >= 600][:len(ord_pred)]
            df = pd.DataFrame({
                'OSD': OSD,
            })
            df.to_csv(out_csv)
            count += 1
            if count > 100:
                tf.keras.backend.clear_session()
                gc.collect()
                model = OSD_architecture()
                model.load_weights(WEIGHTS_PATH)
                count = 0
        except Exception as e:
            print(f"Error in prediction: {e}")

# ------------- MAIN PIPELINE ----------------
if __name__ == '__main__':
    convert_edf_to_h5()
    h5_files = glob(f'{WRITE_PATH_EDF}*.h5')
    files_to_process = h5_files if PROCESS_ALL else pd.read_csv(CSV_SPLIT_PATH)['fileid'].apply(lambda x: opj(WRITE_PATH_EDF, x+'.h5')).tolist()
    for file in tqdm(files_to_process):
        create_OSD_FILES(file, WRITE_PATH_H5)
    run_model_prediction()
