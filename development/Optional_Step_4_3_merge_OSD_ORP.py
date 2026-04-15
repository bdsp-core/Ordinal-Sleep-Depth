import pandas as pd
import numpy as np
import h5py as h5
import operator
import json
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join as opj


def grab_orp(filepath):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        orp_json = json.load(file)

    fullORP = orp_json["FullORPs"]
    ORP = np.zeros(len(fullORP)*10)*np.nan
    for i,orp in enumerate(fullORP):
        if len(orp['ORP1Values'])>0 and len(orp['ORP2Values'])>0:
            tmp = (np.array(orp['ORP1Values'])+np.array(orp['ORP2Values']))/2
        elif len(orp['ORP1Values'])>0:
            tmp = np.array(orp['ORP1Values'])
        elif len(orp['ORP2Values'])>0:
            tmp = np.array(orp['ORP2Values'])
        else:
            continue
        ORP[i*10:(i*10)+len(tmp)] = tmp
    return ORP

def grab_OSD(filepath):
    df = pd.read_csv(filepath)
    label = df['Label'].to_numpy()
    pred = df['ordinal_pred_w'].to_numpy()
    return df, label,pred

def grab_OSD_2(filepath):
    df = pd.read_csv(filepath)
    df = df.add_prefix('OSD_2_') 
    return  df


from glob import glob 

for c in ['test','cognitive']:
    files_ORP = glob('/media/erikjan/Expansion/OSD/Datasets/raw/predictions/*/*.csv')
    files_ORP = [f.split('/')[-1][:-4] for f in files_ORP]
    files_OSD = glob(f'/media/erikjan/Expansion/OSD/predictions/original_may_13/{c}/*.csv')
    files_OSD = [f.split('/')[-1][:-4] for f in files_OSD]

    files = [f for f in files_OSD if f in files_ORP]
    

    for pat in tqdm(files):
        try:
            pat_min = pat.split('_ses')[0]
            pat_max = pat.split('_ses')[1]

            orp_path = f'/media/erikjan/Expansion/OSD/Datasets/raw/predictions/{pat}/{pat}_Autoscoring Events.json'
            osd_path = f'/media/erikjan/Expansion/OSD/predictions/original_may_13/{c}/{pat}.csv'
            prepared_path = f'/media/erikjan/Expansion/OSD/Datasets/raw/mgh/{pat}.h5'

            write_path = f'/media/erikjan/Expansion/OSD/predictions/OSD_FINAL/{pat}.csv'

            # #predictions
            orp = grab_orp(orp_path)
            DF_to_save, label, osd = grab_OSD(osd_path)
            
            #pre-processed
            f = h5.File(prepared_path,'r')
            pp_data = f['signals']['c3-m2'][()]
            label = f['annotations']['stage'][()]

            ll = len(label)//6000*6000
            label = label[:ll]
            idx = np.arange(0,len(label),600)
            label = label[idx,0]
            orp_nonan = orp[~np.isnan(label)]
            label = label[~np.isnan(label)]

            #clip to multiple of 10
            min_length = min(len(label), len(orp_nonan), len(osd))
            if min_length>0:
                orp_nonan = orp_nonan[:min_length]
                DF_to_save = DF_to_save.iloc[:min_length]
                DF_to_save['ORP'] = orp_nonan
                DF_to_save['OSD'] = DF_to_save['ordinal_pred_w']

                #save
                DF_to_save.to_csv(opj(write_path))

        except:
            pass
