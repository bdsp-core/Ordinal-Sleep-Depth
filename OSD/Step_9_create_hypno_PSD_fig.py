# import pandas as pd
# import numpy as np
# import h5py as h5
# import operator
# import json
# import mne
# from tqdm import tqdm
# from os.path import join as opj
# from utils.definitions.Power_spectral_density import *

# import matplotlib
# # Choose a different backend, e.g., Agg or Qt5Agg
# matplotlib.use('Agg')  # or 'Agg' for non-GUI

# import matplotlib.pyplot as plt


# def grab_orp(filepath):
#     # Open and read the JSON file
#     with open(filepath, 'r') as file:
#         orp_json = json.load(file)

#     fullORP = orp_json["FullORPs"]
#     ORP = np.zeros(len(fullORP)*10)*np.nan
#     for i,orp in enumerate(fullORP):
#         if len(orp['ORP1Values'])>0 and len(orp['ORP2Values'])>0:
#             tmp = (np.array(orp['ORP1Values'])+np.array(orp['ORP2Values']))/2
#         elif len(orp['ORP1Values'])>0:
#             tmp = np.array(orp['ORP1Values'])
#         elif len(orp['ORP2Values'])>0:
#             tmp = np.array(orp['ORP2Values'])
#         else:
#             continue
#         ORP[i*10:(i*10)+len(tmp)] = tmp
#     return ORP

# def grab_OSD(filepath):
#     df = pd.read_csv(filepath)
#     label = df['Label'].to_numpy()
#     pred = df['ordinal_pred_w'].to_numpy()
#     return df, label,pred

# def grab_OSD_2(filepath):
#     df = pd.read_csv(filepath)
#     df = df.add_prefix('OSD_2_') 
#     return  df


# def ORP_30_sec(orp):
#     orp_30 = orp.rolling(window=10, center=False, min_periods=0).mean()
#     orp_30=orp_30[::10].repeat(10)
#     if len(orp_30)>len(orp):
#         orp_30 = orp_30[:len(orp)]
#     return orp_30.round(2).values.tolist()

# def OSD_30_sec(osd):
#     osd[osd>1]=1
#     osd[osd<0]=0
#     osd_30 = osd.rolling(window=10, center=True, min_periods=0).mean()
#     return osd_30.round(2).values.tolist()

# import ray
# ray.init(num_cpus=15)

# @ray.remote
# def PSD(data):
#     mt_spectrogram, stimes, sfreqs, mt_spectrogram_DB = multitaper_spectrogram(data,128,min_nfft=2048,frequency_range=[0,20],multiprocess = True ,window_params=[3,0.1],plot_on=False)
#     return mt_spectrogram, stimes, sfreqs, mt_spectrogram_DB


# from glob import glob 
# params = pd.read_pickle('/media/erikjan/Expansion/OSD/python_code/utils/fitted_spline/fitted_spline_train_mgh_fit_values_may13.pkl')
# for c in ['cognitive']:


#     files = glob('/media/erikjan/Expansion/OSD/predictions/OSD_FINAL/*.csv')
#     for pat in tqdm(files):
#         try:

#             id = pat.split('/')[-1][:-4]
#             prepared_path = f'/media/erikjan/Expansion/OSD/Datasets/raw/mgh/{id}.h5'

#             # #predictions
#             df = pd.read_csv(pat)
#             orp = df['ORP'].copy()
#             osd = df['OSD'].copy()
#             osd = (osd-params['min'])/params['max']
#             osd_30 = np.array(OSD_30_sec(osd))
#             orp_30 = np.array(ORP_30_sec(orp))

            
#             #pre-processed
#             f = h5.File(prepared_path,'r')
#             pp_data1 = f['signals']['f3-m2'][()]
#             pp_data2 = f['signals']['f4-m1'][()]
#             frontal_data = pp_data1+pp_data2
#             pp_data1 = f['signals']['c3-m2'][()]
#             pp_data2 = f['signals']['c4-m1'][()]
#             central_data = pp_data1+pp_data2
#             pp_data1 = f['signals']['o1-m2'][()]
#             pp_data2 = f['signals']['o2-m1'][()]
#             occ_data = pp_data1+pp_data2
#             label = f['annotations']['stage'][()]

#             ll = len(label)//6000*6000
#             label = label[:ll]
#             frontal_data = frontal_data[:ll]
#             frontal_data = frontal_data[~np.isnan(label)]
#             central_data = central_data[:ll]
#             central_data = central_data[~np.isnan(label)]
#             occ_data = occ_data[:ll]
#             occ_data = occ_data[~np.isnan(label)]
#             idx = np.arange(0,len(label),600)
#             label = label[idx,0]
            
#             label = label[~np.isnan(label)]
            

#             futures = PSD.remote(frontal_data)
#             mt_spectrogram, stimes, sfreqs, f_spectrogram_DB = ray.get(futures)
#             vmin_f  = np.percentile(f_spectrogram_DB,5)
#             vmax_f  = np.percentile(f_spectrogram_DB,95)

#             futures = PSD.remote(central_data)
#             mt_spectrogram, stimes, sfreqs, c_spectrogram_DB = ray.get(futures)
#             vmin_c  = np.percentile(c_spectrogram_DB,5)
#             vmax_c  = np.percentile(c_spectrogram_DB,95)

#             futures = PSD.remote(occ_data)
#             mt_spectrogram, stimes, sfreqs, o_spectrogram_DB = ray.get(futures)
#             vmin_o  = np.percentile(o_spectrogram_DB,5)
#             vmax_o  = np.percentile(o_spectrogram_DB,95)


#             # Create a wide figure with 5 rows
#             fig, axs = plt.subplots(5, 1, figsize=(18, 10))  # Adjust figsize as needed

#             # Define x-axis limits based on the length of the label
#             x_limits = [0, len(label)]

#             # Row 1: Plotting spectrogram for Frontal Electrodes
#             axs[0].imshow(f_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
#                             extent=[0, len(label), 0, 20], vmin=vmin_f, vmax=vmax_f)
#             axs[0].set_title('Average Power of Frontal Electrodes', fontsize=16)
#             axs[0].set_xlabel('Time', fontsize=14)
#             axs[0].set_ylabel('Frequency', fontsize=14)
#             axs[0].set_xlim(x_limits)  # Set x-axis limits

#             # Row 2: Plotting spectrogram for Central Electrodes
#             axs[1].imshow(c_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
#                             extent=[0, len(label), 0, 20], vmin=vmin_c, vmax=vmax_c)
#             axs[1].set_title('Average Power of Central Electrodes', fontsize=16)
#             axs[1].set_xlabel('Time', fontsize=14)
#             axs[1].set_ylabel('Frequency', fontsize=14)
#             axs[1].set_xlim(x_limits)  # Set x-axis limits

#             # Row 3: Plotting spectrogram for Occipital Electrodes
#             axs[2].imshow(o_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
#                             extent=[0, len(label), 0, 20], vmin=vmin_o, vmax=vmax_o)
#             axs[2].set_title('Average Power of Occipital Electrodes', fontsize=16)
#             axs[2].set_xlabel('Time', fontsize=14)
#             axs[2].set_ylabel('Frequency', fontsize=14)
#             axs[2].set_xlim(x_limits)  # Set x-axis limits

#             # Row 4: Plotting OSD and Hypnogram
#             ax1 = axs[3].twinx()  # Create a twin Axes sharing the x-axis

#             # Plot OSD with transparency
#             ax1.plot(osd, label='OSD', color='#003366', alpha=0.2, zorder=1)
#             ax1.plot(osd_30, label='OSD', color='#003366', alpha=0.5, zorder=1)
#             ax1.set_ylabel('OSD', fontsize=14)
#             # Set y-limits for OSD
#             ax1.set_ylim(
#                 min(osd_30) - min(osd_30) * 0.05,
#                 max(osd_30) + max(osd_30) * 0.05
#             )
#             ax1.set_yticks([0, 0.5, 1])  # Set y-ticks for OSD
#             ax1.set_xlim(x_limits)  # Set x-axis limits

#             # Plot the hypnogram as a line on the primary axis to ensure it's on top
#             rem_label = label.copy()
#             rem_label[rem_label != 4] = np.nan  # Masking other values
#             axs[3].plot(label, color='black', label='Hypnogram', alpha=0.6,zorder=3)  # Hypnogram
#             axs[3].plot(rem_label, color='red', zorder=4)  # Value = 4 in red
#             axs[3].set_title('OSD and Hypnogram', fontsize=16)
#             axs[3].set_ylabel('Hypnogram', fontsize=14)
#             # Set y-limits for hypnogram
#             axs[3].set_ylim(
#                 min(label) - min(label) * 0.05,
#                 max(label) + max(label) * 0.05
#             )
#             axs[3].set_yticks([1, 2, 3, 4, 5])  # Set y-ticks for hypnogram
#             axs[3].set_yticklabels(['N3', 'N2', 'N1', 'REM', 'W'])  # Set y-tick labels
#             axs[3].grid(True)

#             # Row 5: Plotting ORP and Hypnogram
#             ax2 = axs[4].twinx()  # Create a twin Axes sharing the x-axis

#             # Plot ORP with transparency
#             ax2.plot(orp, label='ORP', color='#003366', alpha=0.2, zorder=1)
#             ax2.plot(orp_30, label='ORP', color='#003366', alpha=0.6, zorder=1)
#             ax2.set_ylabel('ORP', fontsize=14)
#             # Set y-limits for ORP
#             ax2.set_ylim(
#                 min(orp_30) - min(orp_30) * 0.05,
#                 max(orp_30) + max(orp_30) * 0.05
#             )
#             ax2.set_xlim(x_limits)  # Set x-axis limits

#             # Plot the hypnogram as a line on the primary axis to ensure it's on top
#             axs[4].plot(label, color='black', label='Hypnogram', alpha=0.6,zorder=3)  # Hypnogram
#             axs[4].plot(rem_label, color='red', zorder=4)  # Value = 4 in red
#             axs[4].set_title('ORP and Hypnogram', fontsize=16)
#             axs[4].set_ylabel('Hypnogram', fontsize=14)
#             # Set y-limits for hypnogram
#             axs[4].set_ylim(
#                 min(label) - min(label) * 0.05,
#                 max(label) + max(label) * 0.05
#             )
#             axs[4].set_yticks([1, 2, 3, 4, 5])  # Set y-ticks for hypnogram
#             axs[4].set_yticklabels(['N3', 'N2', 'N1', 'REM', 'W'])  # Set y-tick labels
#             axs[4].grid(True)

#             # Adjust layout
#             plt.tight_layout()

#             # Save the figure
#             plt.savefig(f'/media/erikjan/Expansion/OSD/figures/OSD_FINAL/test/psd/{id}_PSD_HYPNO.png')

#             # Show the plot
#             plt.close()


#             print('a')



#         except:
#             pass


import pandas as pd
import numpy as np
import h5py as h5
import operator
import json
import mne
from tqdm import tqdm
from os.path import join as opj
from utils.definitions.Power_spectral_density import *

import matplotlib
# Choose a different backend, e.g., Agg or Qt5Agg


import matplotlib.pyplot as plt


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


def ORP_30_sec(orp):
    orp_30 = orp.rolling(window=10, center=False, min_periods=0).mean()
    orp_30=orp_30[::10].repeat(10)
    if len(orp_30)>len(orp):
        orp_30 = orp_30[:len(orp)]
    return orp_30.round(2).values.tolist()

def OSD_30_sec(osd):
    osd[osd>1]=1
    osd[osd<0]=0
    osd_30 = osd.rolling(window=10, center=True, min_periods=0).mean()
    return osd_30.round(2).values.tolist()

import ray
ray.init(num_cpus=15)

@ray.remote
def PSD(data):
    mt_spectrogram, stimes, sfreqs, mt_spectrogram_DB = multitaper_spectrogram(data,128,min_nfft=2048,frequency_range=[0,20],multiprocess = True ,window_params=[3,0.1],plot_on=False)
    return mt_spectrogram, stimes, sfreqs, mt_spectrogram_DB


from glob import glob 
params = pd.read_pickle('/media/erikjan/Expansion/OSD/python_code/utils/fitted_spline/fitted_spline_train_mgh_fit_values_may13.pkl')



files = glob('/media/erikjan/Expansion/OSD/predictions/original_may_13/development/*.csv')
for pat in tqdm(files[100:]):
    try:

        id = pat.split('/')[-1][:-4]
        prepared_path = f'/media/erikjan/Expansion/OSD/Datasets/raw/mgh/{id}.h5'


        # #predictions
        df = pd.read_csv(pat)
        osd = df['ordinal_pred_w'].copy()
        osd = (osd-params['min'])/params['max']
        osd_30 = np.array(OSD_30_sec(osd))

        
        #pre-processed
        f = h5.File(prepared_path,'r')
        pp_data1 = f['signals']['f3-m2'][()]
        pp_data2 = f['signals']['f4-m1'][()]
        frontal_data = pp_data1+pp_data2
        pp_data1 = f['signals']['c3-m2'][()]
        pp_data2 = f['signals']['c4-m1'][()]
        central_data = pp_data1+pp_data2
        pp_data1 = f['signals']['o1-m2'][()]
        pp_data2 = f['signals']['o2-m1'][()]
        occ_data = pp_data1+pp_data2
        label = f['annotations']['stage'][()]

        if len(np.unique(label))<5:
            a=b=s

        ll = len(label)//6000*6000
        label = label[:ll]
        frontal_data = frontal_data[:ll]
        frontal_data = frontal_data[~np.isnan(label)]
        central_data = central_data[:ll]
        central_data = central_data[~np.isnan(label)]
        occ_data = occ_data[:ll]
        occ_data = occ_data[~np.isnan(label)]
        idx = np.arange(0,len(label),600)
        label = label[idx,0]
        
        label = label[~np.isnan(label)]
        

        futures = PSD.remote(frontal_data)
        mt_spectrogram, stimes, sfreqs, f_spectrogram_DB = ray.get(futures)
        vmin_f  = np.percentile(f_spectrogram_DB,5)
        vmax_f  = np.percentile(f_spectrogram_DB,95)

        futures = PSD.remote(central_data)
        mt_spectrogram, stimes, sfreqs, c_spectrogram_DB = ray.get(futures)
        vmin_c  = np.percentile(c_spectrogram_DB,5)
        vmax_c  = np.percentile(c_spectrogram_DB,95)

        futures = PSD.remote(occ_data)
        mt_spectrogram, stimes, sfreqs, o_spectrogram_DB = ray.get(futures)
        vmin_o  = np.percentile(o_spectrogram_DB,5)
        vmax_o  = np.percentile(o_spectrogram_DB,95)


        # Create a wide figure with 5 rows
        fig, axs = plt.subplots(4, 1, figsize=(18, 10))  # Adjust figsize as needed

        # Define x-axis limits based on the length of the label
        x_limits = [0, len(label)]

        # Row 1: Plotting spectrogram for Frontal Electrodes
        axs[0].imshow(f_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
                        extent=[0, len(label), 0, 20], vmin=vmin_f, vmax=vmax_f)
        axs[0].set_title('Average Power of Frontal Electrodes', fontsize=16)
        axs[0].set_xlabel('Time', fontsize=14)
        axs[0].set_ylabel('Frequency', fontsize=14)
        axs[0].set_xlim(x_limits)  # Set x-axis limits

        # Row 2: Plotting spectrogram for Central Electrodes
        axs[1].imshow(c_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
                        extent=[0, len(label), 0, 20], vmin=vmin_c, vmax=vmax_c)
        axs[1].set_title('Average Power of Central Electrodes', fontsize=16)
        axs[1].set_xlabel('Time', fontsize=14)
        axs[1].set_ylabel('Frequency', fontsize=14)
        axs[1].set_xlim(x_limits)  # Set x-axis limits

        # Row 3: Plotting spectrogram for Occipital Electrodes
        axs[2].imshow(o_spectrogram_DB, aspect='auto', cmap='jet', origin='lower',
                        extent=[0, len(label), 0, 20], vmin=vmin_o, vmax=vmax_o)
        axs[2].set_title('Average Power of Occipital Electrodes', fontsize=16)
        axs[2].set_xlabel('Time', fontsize=14)
        axs[2].set_ylabel('Frequency', fontsize=14)
        axs[2].set_xlim(x_limits)  # Set x-axis limits

        # Row 4: Plotting OSD and Hypnogram
        ax1 = axs[3].twinx()  # Create a twin Axes sharing the x-axis

        # Plot OSD with transparency
        ax1.plot(osd, label='OSD', color='#003366', alpha=0.2, zorder=1)
        ax1.plot(osd_30, label='OSD', color='#003366', alpha=0.5, zorder=1)
        ax1.set_ylabel('OSD', fontsize=14)
        # Set y-limits for OSD
        ax1.set_ylim(
            min(osd_30) - min(osd_30) * 0.05,
            max(osd_30) + max(osd_30) * 0.05
        )
        ax1.set_yticks([0, 0.5, 1])  # Set y-ticks for OSD
        ax1.set_xlim(x_limits)  # Set x-axis limits

        # Plot the hypnogram as a line on the primary axis to ensure it's on top
        rem_label = label.copy()
        rem_label[rem_label != 4] = np.nan  # Masking other values
        axs[3].plot(label, color='black', label='Hypnogram', alpha=0.6,zorder=3)  # Hypnogram
        axs[3].plot(rem_label, color='red', zorder=4)  # Value = 4 in red
        axs[3].set_title('OSD and Hypnogram', fontsize=16)
        axs[3].set_ylabel('Hypnogram', fontsize=14)
        # Set y-limits for hypnogram
        axs[3].set_ylim(
            min(label) - min(label) * 0.05,
            max(label) + max(label) * 0.05
        )
        axs[3].set_yticks([1, 2, 3, 4, 5])  # Set y-ticks for hypnogram
        axs[3].set_yticklabels(['N3', 'N2', 'N1', 'REM', 'W'])  # Set y-tick labels
        axs[3].grid(True)

        
        # Adjust layout
        plt.tight_layout()

        # Save the figure
        #plt.savefig(f'/media/erikjan/Expansion/OSD/figures/OSD_FINAL/test/psd/{id}_PSD_HYPNO.png')

        # Show the plot
        plt.show()
        plt.close()


        print('a')



    except:
        pass
