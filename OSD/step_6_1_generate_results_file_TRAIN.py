import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr
from glob import glob
import os
from tqdm import tqdm

# Function to calculate Spearman and Pearson correlations between label and algorithm predictions
def calculate_correlations(df,columns=['ordinal_pred_w'],names=['OSD']):
    correlations = {}
    df_tmp = df[['Label']+list(set(columns))].dropna()
    for col,name in zip(columns,names):
        algo = (df_tmp[col]-params['min'])/params['max']
        algo[algo>1]=1
        algo[algo<0]=0
        algo_30 = algo.rolling(window=10, center=True, min_periods=0).mean()


        spearman_corr, _ = spearmanr(df_tmp['Label'], algo)
        pearson_corr, _ = pearsonr(df_tmp['Label'], algo)

        spearman_corr_30, _ = spearmanr(df_tmp['Label'], algo_30)
        pearson_corr_30, _ = pearsonr(df_tmp['Label'], algo_30)
        correlations[name] = {'spearman': spearman_corr, 'pearson': pearson_corr,'spearman_30': spearman_corr_30, 'pearson_30': pearson_corr_30}

    return correlations

def calculate_stage_metrics(df,columns=['ordinal_pred_w','OSD_2_ordinal_pred_w','ORP'],names=['OSD','OSD_2','ORP']):
    stage_metrics = {}
    df_tmp = df[['Label']+list(set(columns))].dropna()
    for col,name in zip(columns,names):
        if ('OSD' in name) & ('_fit' in name):
            algo = spline((df_tmp[col]-params['min'])/params['max'])
            algo[algo>1]=1
            algo[algo<0]=0
        elif ('OSD' in name):
            algo = (df_tmp[col]-params['min'])/params['max']
            algo[algo>1]=1
            algo[algo<0]=0
        else:
            algo = df_tmp[col]

        len_NR = len(df_tmp[(df_tmp['Label']<4)&(df_tmp['Label']>0)])
        if len_NR ==0:
            len_R = 0
            len_N1 = 0
            len_N2 = 0
            len_N3 = 0
        else:
            len_R = len(df_tmp[(df_tmp['Label']==4)])/len_NR
            len_N1 = len(df_tmp[(df_tmp['Label']==3)])/len_NR
            len_N2 = len(df_tmp[(df_tmp['Label']==2)])/len_NR
            len_N3 = len(df_tmp[(df_tmp['Label']==1)])/len_NR

        val_NR = algo[(df_tmp['Label']<4)&(df['Label']>0)].mean()
        val_R  = algo[(df_tmp['Label']==4)].mean()
        val_N1 = algo[(df_tmp['Label']==3)].mean()
        val_N2 = algo[(df_tmp['Label']==2)].mean()
        val_N3 = algo[(df_tmp['Label']==1)].mean()
        val_W = algo[(df_tmp['Label']==5)].mean()

        stage_metrics[f'{name}_NR'] = val_NR
        stage_metrics[f'{name}_R'] = val_R
        stage_metrics[f'{name}_N1'] = val_N1
        stage_metrics[f'{name}_N2'] = val_N2
        stage_metrics[f'{name}_N3'] = val_N3
        stage_metrics[f'{name}_W'] = val_W

    
    stage_metrics[f'NR_len'] = len_NR
    stage_metrics[f'R_len'] = len_R
    stage_metrics[f'N1_len'] = len_N1
    stage_metrics[f'N2_len'] = len_N2
    stage_metrics[f'N3_len'] = len_N3

    stage_metrics[f'W_time'] = len(df_tmp[(df_tmp['Label']==5)])
    stage_metrics[f'NR_time'] = len(df_tmp[(df_tmp['Label']<4)&(df_tmp['Label']>0)])
    stage_metrics[f'R_time'] = len(df_tmp[(df_tmp['Label']==4)])
    stage_metrics[f'N1_time'] = len(df_tmp[(df_tmp['Label']==3)])
    stage_metrics[f'N2_time'] = len(df_tmp[(df_tmp['Label']==2)])
    stage_metrics[f'N3_time'] = len(df_tmp[(df_tmp['Label']==1)])

    return stage_metrics


def calculate_dem_cat(id,master_df):
    dem_cat = {}
    master_row = master_df[master_df['fileid']==id]
    DEM = int(master_row['dx_dementia'].values[0])
    MCI = int(master_row['dx_mci'].values[0])
    SYM = int(master_row['dx_symptomatic'].values[0])
    NO_DEM = int(master_row['dx_no-dementia'].values[0])

    DEM_cat = np.nan
    if DEM==1:
        DEM_cat=3
    elif MCI==1:
        DEM_cat=2
    elif SYM==1:
        DEM_cat=1
    elif NO_DEM ==1:
        DEM_cat=0

    dem_cat['DEM_cat']=DEM_cat
    return dem_cat


###################################################################################################################
##############################################         START         ##############################################
###################################################################################################################


# Load the master CSV
train_csv = pd.read_csv('/media/erikjan/Expansion/OSD/data_splits/final_split/train_subjects.csv')
val_csv = pd.read_csv('/media/erikjan/Expansion/OSD/data_splits/final_split/val_subjects.csv')
master_csv = pd.concat([train_csv, val_csv], ignore_index=True).drop(['dt_no-dementia', 
                                                                            'certainty_no-dementia', 'dt_symptomatic', 
                                                                            'certainty_symptomatic', 'dt_mci', 'certainty_mci', 
                                                                            'dt_dementia', 'certainty_dementia'
                                                                            ],axis=1)

list_of_csv_files = glob('/media/erikjan/Expansion/OSD/predictions/original_may_13/development/*.csv')

#read fit
spline = pd.read_pickle('/media/erikjan/Expansion/OSD/python_code/utils/fitted_spline/fitted_spline_train_mgh_may13.pkl')
params = pd.read_pickle('/media/erikjan/Expansion/OSD/python_code/utils/fitted_spline/fitted_spline_train_mgh_fit_values_may13.pkl')
np.seterr(invalid='ignore')


# Assuming we have 5000+ prediction CSV files
for file in tqdm(list_of_csv_files):
    try:
        # Extract participant ID from the file name (assumes file name format 'ID.csv')
        participant_id = os.path.basename(file).split('.')[0]
        
        # Load the prediction CSV
        df = pd.read_csv(file)
        
        # Perform all calculations (confusion matrix, correlations, kappa)
        columns=['ordinal_pred_w']
        names=['OSD']
        corr_values = calculate_correlations(df,columns,names)
        stage_values = calculate_stage_metrics(df,columns,names)
        dem_values = {'DEM_cat': 0}

        # Store the results in the master CSV, using the 'ID' column to match participants
        for col in corr_values:
            master_csv.loc[master_csv['fileid'] == participant_id, f'spearman_{col}'] = corr_values[col]['spearman']
            master_csv.loc[master_csv['fileid'] == participant_id, f'pearson_{col}'] = corr_values[col]['pearson']
            master_csv.loc[master_csv['fileid'] == participant_id, f'spearman_{col}_30'] = corr_values[col]['spearman_30']
            master_csv.loc[master_csv['fileid'] == participant_id, f'pearson_{col}_30'] = corr_values[col]['pearson_30']
        for col in stage_values:
            master_csv.loc[master_csv['fileid'] == participant_id, f'{col}'] = stage_values[col]
        for col in dem_values:
            master_csv.loc[master_csv['fileid'] == participant_id, f'{col}'] = dem_values[col]
        
    except:
        pass

# Save the updated master CSV
master_csv.to_csv('development_RESULTS.csv', index=False)