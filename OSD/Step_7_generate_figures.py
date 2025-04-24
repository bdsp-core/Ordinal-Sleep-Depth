
from sklearn.metrics import confusion_matrix 
from scipy.stats import pearsonr, spearmanr
from collections import  Counter
from utils.evaluate.evaluation_tools import *
from statistics import mode
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import ray
import os
from scipy.stats import *
import random 
random.seed(2024)

ray.init(num_cpus=30)

########
#definitions
########


@ray.remote
def load_orp_osd(file_path):
    try:
        df = pd.read_csv(file_path)
        OSD = df['OSD'].to_numpy()
        ORP = df['ORP'].to_numpy()
        hypno = df['Label'].to_numpy()

        return hypno,OSD,ORP
    except:
        return [],[],[]

@ray.remote
def load_orp_osd_arousal(file_path):
    try:
        df = pd.read_csv(file_path)
        OSD = df['OSD'].to_numpy()
        ORP = df['ORP'].to_numpy()
        hypno = df['Label'].to_numpy()
        arousal = df['Arousal'].to_numpy()
        if max(arousal)>0:
            hypno[arousal==1]=5
            return hypno,OSD,ORP
        else:
            return [],[],[]
    except:
        return [],[],[]

@ray.remote
def cm_calc(file,label_name,pred_name,labels,mask=False):
    try:
        #read in data
        file_path = file
        df = pd.read_csv(file_path)
        label = df[label_name]
        pred = df[pred_name]

        if mask:
            diff_loc = np.where(np.diff(label)!=0)[0]
            for dl in diff_loc:
                pred[np.max((dl-5,0)):np.min((dl+5,len(pred)))]=0


        #remove nan
        pred = pred[~np.isnan(label)].to_numpy()
        label = label[~np.isnan(label)].to_numpy()

        return confusion_matrix(label,pred,labels=labels) 
    except:
        return np.zeros((len(labels),len(labels)))

def return_results(results):
    summed_data = {}

    # Iterate over the outer list
    for inner_list in results:
        # Iterate over each dictionary in the inner list
        for item in inner_list:
            if item in summed_data:
                for key, value in inner_list[item].items():
                    summed_data[item][key] += value
            else:
                    summed_data[item] = inner_list[item]

    x = []
    y = []
    for k in summed_data.keys():
        x.append(k)
        y.append(summed_data[k]['total_fives']/summed_data[k]['total_occurrences'])
    return x,y

def merge_results(results):
    all_unique_vals = set()  # To store unique OSD values across all patients
    count_dict = {}          # To accumulate counts
    len_dict = {}            # To accumulate lengths

    # Iterate over each patient's results
    for patient_result in results:
        unique_vals, count_vals, len_vals = patient_result
        
        # Update the set of all unique values
        cleaned_vals = [val for val in unique_vals if not pd.isna(val)]
        all_unique_vals.update(cleaned_vals)
        
        # Sum count and len values per unique OSD value
        for idx, val in enumerate(cleaned_vals):
            # Initialize the dictionaries if the value is new
            if val not in count_dict:
                count_dict[val] = 0
                len_dict[val] = 0
            
            # Sum the counts and lengths
            count_dict[val] += count_vals[idx]
            len_dict[val] += len_vals[idx]

    # Convert the set of unique values back to a sorted list
    sorted_unique_vals = sorted(all_unique_vals)

    # Prepare the final output lists
    summed_counts = []
    summed_lengths = []
    
    for val in sorted_unique_vals:
        summed_counts.append(count_dict[val])
        summed_lengths.append(len_dict[val])

    return [sorted_unique_vals, summed_counts, summed_lengths] 


def merge_results_new(results):
    all_unique_vals = set()  # To store unique OSD values across all patients
    count_dict = {}          # To accumulate counts
    len_dict = {}            # To accumulate lengths

    # Iterate over each patient's results
    for patient_result in results:
        unique_vals, count_vals, len_vals = patient_result
        
        # Update the set of all unique values
        cleaned_vals = [val for val in unique_vals if not pd.isna(val)]
        all_unique_vals.update(cleaned_vals)
        
        # Sum count and len values per unique OSD value
        for idx, val in enumerate(cleaned_vals):
            # Initialize the dictionaries if the value is new
            if val not in count_dict:
                count_dict[val] = []
                len_dict[val] = []
            
            # Sum the counts and lengths
            count_dict[val].append(count_vals[idx])
            len_dict[val].append(len_vals[idx])

    # Convert the set of unique values back to a sorted list
    sorted_unique_vals = sorted(all_unique_vals)

    # Prepare the final output lists
    summed_counts = []
    summed_lengths = []
    
    for val in sorted_unique_vals:
        summed_counts.append(count_dict[val])
        summed_lengths.append(len_dict[val])

    ArI_mean = []
    for s,l in zip(summed_counts,summed_lengths):
        ArI_mean.append(np.mean(np.array(s)/np.array(l)))

    return [sorted_unique_vals, ArI_mean] 

@ray.remote
def calc_ArI(OSD,label):
    
    unique_osd_values = np.unique(OSD)
    unique_val = []
    count_val = []
    len_val = []
    # Count occurrences of each unique OSD value and the corresponding '5's in the label
    for value in unique_osd_values:
        positions_with_value = np.where(OSD == value)[0] 
        list_temp = [[label[idx+1:idx+11]] for idx in positions_with_value]
        # Check if any subarray contains a 5
        contains_five = [any(5 in subarray for subarray in inner_list) for inner_list in list_temp]
        unique_val.append(value)
        count_val.append(sum(contains_five))
        len_val.append(len(contains_five))
        
    return [unique_val,count_val,len_val]

@ray.remote
def calc_ArI2(OSD_list,label_list,value):
    
    return_array = []
    for OSD, label in zip(OSD_list,label_list):
        positions_with_value = np.where(OSD == value)[0] 
        list_temp = [[label[idx+1:idx+11]] for idx in positions_with_value]
        # Check if any subarray contains a 5
        contains_five = [any(5 in subarray for subarray in inner_list) for inner_list in list_temp]
        return_array.append(contains_five)

    return return_array



@ray.remote
def find_arousal_wakening(pat_label,idx1,idx2):
    if any(pat_label[idx1:idx2]==5):
        return 1
    else:
        return 0
    
@ray.remote
def find_ArI_shell1(pat_label,pat_pred,unique_pred):
    loc = np.where(pat_pred==unique_pred)[0]
    if len(loc)>0:
        areas_of_interest = [loc+1,loc+11]
        futures = [find_arousal_wakening.remote(pat_label,areas_of_interest[0][i],areas_of_interest[1][i]) for i in range(len(areas_of_interest[0]))]
        return ray.get(futures)    
    else:
        return []

   
def find_ArI_shell2(labels_list,pred_list,unique_pred):
    
    futures_shell1 = [find_ArI_shell1.remote(pat_label,pat_pred,unique_pred) for pat_label,pat_pred in zip(labels_list,pred_list)]

    return ray.get(futures_shell1)


def find_mode(label,pred,val):
    loc = np.where(pred==val)[0]
    mode_ = mode(np.array(label)[loc])[0][0]
    return mode_

def find_indices_and_labels(osd_list, labels_list, target_value):
    osd_indices = []  # To store the indices of the sublists in osd_list
    osd_positions = []  # To store the indices of the target_value within osd_list
    label_indices = []  # To store the corresponding labels

    for sublist_index, (sublist, label_sublist) in enumerate(zip(osd_list, labels_list)):
        # Convert the sublist to a NumPy array for efficient searching
        sublist_array = np.array(sublist)
        
        # Find the indices where the target_value exists
        found_indices = np.where(sublist_array == target_value)[0]

        if found_indices.size > 0:
            label_indices.append([label_sublist[i] for i in found_indices])  # Get corresponding labels

    return calculate_mode(label_indices)

def calculate_mode(labels):
    # Flatten the list of labels
    flat_labels = [label for sublist in labels for label in sublist]
    
    # Count occurrences of each label
    label_counts = Counter(flat_labels)
    
    # Find the mode (most common label)
    if label_counts:
        mode_label, mode_count = label_counts.most_common(1)[0]
        try:
            return mode_label[-1]
        except:
            return mode_label
    else:
        return None

def ORP_30_sec(orp):
    orp = pd.Series(orp)
    orp_30 = orp.rolling(window=10, center=False, min_periods=0).mean()
    orp_30=orp_30[::10].repeat(10)
    if len(orp_30)>len(orp):
        orp_30 = orp_30[:len(orp)]
    return orp_30.round(2).values.tolist()

def OSD_30_sec(osd):
    osd = pd.Series(osd)
    osd = (osd-params['min'])/params['max']
    osd[osd>1]=1
    osd[osd<0]=0
    osd_30 = osd.rolling(window=10, center=True, min_periods=0).mean()
    return osd_30.round(2).values.tolist()


##########################################################################################################################################################################
# Fill in 
##########################################################################################################################################################################
WORKING_DIRECTORY = '/user/your/path/to/OSD_repo'
model_version ='OSD_FINAL'
evaluation_part = 'test'
#load files

pred_path = f'{WORKING_DIRECTORY}/OSD_predictions/{model_version}/'
os.makedirs(f'{WORKING_DIRECTORY}/figures/{model_version}/',exist_ok=True)

df_test_all = pd.read_csv(f'{WORKING_DIRECTORY}/OSD_predictions/Test_RESULTS.csv') 

files = glob(f'{pred_path}*.csv')
df_test_all_tmp = df_test_all.copy()
df_test_all_tmp['DEM_cat'] = df_test_all_tmp['DEM_cat'].fillna(0)
files_test =df_test_all_tmp[df_test_all_tmp['DEM_cat']<=3]['fileid'].values
files = [f for f in files if os.path.basename(f)[:-4] in files_test]

#what to calculate
calculate_boxplot = True #DONE
age_vs_sleep_depth = False #DONE
hypno_kappa = True #DONE
calculate_ArI_new = False
masked = False

target_names_class = ['N3','N2','N1','R','W']
target_names_ordinal = ['N3','N2','N1','W']
color_list = ['gold','dodgerblue','seagreen','crimson','black']
np.random.seed(2023)

params = pd.read_pickle(f'{WORKING_DIRECTORY}/OSD/utils/scaling_osd/scaling_values.pkl')


#################
# BoxPlot
#################

if calculate_boxplot == True:
    #get labels
    futures = [load_orp_osd.remote(file) for file in files]
    label_pred_list = ray.get(futures)
    labels_list=[]
    osd_list = []
    orp_list = []
    for hypno,osd,orp in label_pred_list:
        labels_list.append(hypno)
        osd_list.append((osd-params['min'])/params['max'])
        orp_list.append(orp)
    
    #append to long list
    labels = [value for patient_list in labels_list for value in patient_list]
    osd = [value for patient_list in osd_list for value in patient_list]
    orp = [value/2.5 for patient_list in orp_list for value in patient_list]

    #create dataframe
    data_w = {'Hypnogram':labels,
              'OSD':osd,
              'ORP':orp}
    df_w = pd.DataFrame(data_w)

    # Melt the DataFrame to long format
    df_melted = pd.melt(df_w, id_vars=['Hypnogram'], value_vars=['OSD', 'ORP'], 
                        var_name='alogrithm', value_name='Value')

    filtered_df = df_melted[df_melted['Hypnogram'].isin([1, 2, 3])].copy()
    filtered_df['Hypnogram']=6
    df_melted = pd.concat([df_melted, filtered_df], ignore_index=True)
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 7))
    # Define a dark blue color palette
    dark_palette = sns.color_palette("Blues", 2)  # 2 shades of blue
    # Plot with the dark blue palette
    sns.boxplot(x='Hypnogram', y='Value', hue='alogrithm', data=df_melted, 
                palette=dark_palette, showfliers=False, ax=ax1)

    # Ensure gridlines are drawn behind the plot
    ax1.set_axisbelow(True)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Customize y-axis label and tick parameters
    ax1.set_ylabel('OSD', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('ORP', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Set x-ticks and labels
    ax1.set_xticks(ticks=[0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['N3', 'N2', 'N1', 'REM', 'Wake','NR'])

    # Set y-ticks for the secondary y-axis
    ax2.set_yticks(ticks=[0, 0.5/2.5, 1/2.5, 1.5/2.5, 2/2.5, 1])
    ax2.set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5])
    ax2.set_ylim([-0.04, 1.04])
    ax1.set_ylim([-0.04, 1.04])

    # Set title and adjust layout
    plt.title('OSD and ORP Distribution per Sleep Stage', fontsize=14)
    plt.tight_layout()

    # Place the legend outside the plot on the upper right
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Category', 
            title_fontsize='13', fontsize='11')
    plt.subplots_adjust(right=0.85)

    # Show the plot
    plt.show()
    plt.savefig(f'{WORKING_DIRECTORY}/figures/{model_version}/{evaluation_part}/Box_Plot_masked_{masked}.png')
    print('a')
    plt.close()

###################################################
# Arousability index
###################################################


#@ray.remote
if calculate_ArI_new == True:
    try:
        #load bootstrapping
        pred_vals = np.load('pred_vals.npy')
        bootstrap_vals_osd = pred_vals[:,0,:101]
        bootstrap_vals_orp = pred_vals[:,1,:]
    except:
        #do bootstrapping
        pred_vals = np.zeros((1000,2,251))*np.nan
        for i in tqdm(range(1)):

            selected_files = random.choices(files, k=len(files))
            futures = [load_orp_osd_arousal.remote(file) for i,file in enumerate(selected_files)]
            label_pred_list = ray.get(futures)
            labels_list=[]
            osd_list = []
            orp_list = []
            for hypno,osd,orp in label_pred_list:
                labels_list.append(hypno)
                osd_list.append(OSD_30_sec(osd))
                orp_list.append(ORP_30_sec(orp))

            for i2, pred_list in enumerate([osd_list,orp_list]):
                #do ArI calculations
                futures = [calc_ArI.remote(algo,label) for algo,label in zip(pred_list,labels_list)]
                results = ray.get(futures)
                unique_preds,ArI_val = merge_results_new(results)
                up = [int(u*100) for u in unique_preds]
                pred_vals[i,i2,up]=ArI_val
        
        bootstrap_vals_osd = pred_vals[:,0,:101]
        bootstrap_vals_orp = pred_vals[:,1,:]
                
    futures = [load_orp_osd_arousal.remote(file) for i,file in enumerate(files)]
    label_pred_list = ray.get(futures)
    labels_list=[]
    osd_list = []
    orp_list = []
    for hypno,osd,orp in label_pred_list:
        labels_list.append(hypno)
        osd_list.append(OSD_30_sec(osd))
        orp_list.append(ORP_30_sec(orp))

    fig, axs = plt.subplots(2, 1,figsize=(10, 6))
    axs[0].set_title("Arousal Index ")
    scaler =[1,2.5]
    c = ['#4C8CC3','#2171B5']
    scaler =[1,2.5]
    bootstrap = [bootstrap_vals_osd,bootstrap_vals_orp]
    label1 = ['Chance of Arousal (OSD)','Chance of Arousal (ORP)']
    label2 = ["OSD Value","ORP Value"]
    
    for plt_idx, pred_list in enumerate([osd_list,orp_list]):
        #do ArI calculations
        futures = [calc_ArI.remote(algo,label) for algo,label in zip(pred_list,labels_list)]
        results = ray.get(futures)
        unique_preds,ArI_val = merge_results_new(results)

        # Remove columns from `bootstrapped_ArI_val` that contain only NaNs
        valid_bootstrap_columns = ~np.all(np.isnan(bootstrap[plt_idx]), axis=0)
        ArI_val = np.array(ArI_val)[valid_bootstrap_columns]
        unique_preds = np.array(unique_preds)[valid_bootstrap_columns]
        bootstrap_vals = bootstrap[plt_idx][:, valid_bootstrap_columns]

        # Calculate the 5th and 95th percentiles, ignoring NaNs in bootstrapped data
        lower_bound = np.nanpercentile(bootstrap_vals, 2.5, axis=0)
        upper_bound = np.nanpercentile(bootstrap_vals, 97.5, axis=0)

        # Identify indices where there are no NaNs across `ArI_val`, `lower_bound`, and `upper_bound`
        valid_indices = ~np.isnan(ArI_val) & ~np.isnan(lower_bound) & ~np.isnan(upper_bound)

        # Filter arrays to include only valid (non-NaN) indices
        ArI_val_clean = ArI_val[valid_indices]
        unique_preds_clean = unique_preds[valid_indices]
        lower_bound_clean = lower_bound[valid_indices]
        upper_bound_clean = upper_bound[valid_indices]

        bp_array = np.zeros((int(scaler[plt_idx]*100+1)))
        for res in results:
            for r in range(len(res[0])):
                try:
                    bp_array[int(res[0][r]*100)]+=res[2][r]
                except:
                    pass
        bp_array = bp_array/np.sum(bp_array)*100

        # Plotting
        axs[plt_idx].plot([0,scaler[plt_idx]],[0,1],'--',color='grey',alpha=0.4)
        axs[plt_idx].plot(unique_preds_clean, ArI_val_clean, label=label1[plt_idx], color=c[plt_idx])
        axs[plt_idx].fill_between(unique_preds_clean, lower_bound_clean, upper_bound_clean, color=c[plt_idx], alpha=0.2, label='95% Confidence Interval')
        axs[plt_idx].legend()
        axs[plt_idx].set_xlabel(label2[plt_idx])
        axs[plt_idx].set_ylabel("Chance of Arousal")
        
        r_value, p_value = pearsonr(unique_preds_clean, ArI_val_clean)

        # Step 1: Calculate Pearson's r for each bootstrap sample
        r_values = np.array([pearsonr(np.arange(len(data)),data)[0] for data in bootstrap_vals])

        # Step 2: Calculate the 95% confidence interval
        alpha = 0.05  # 5% significance level
        lower_bound = np.percentile(r_values, 2.5)  # 2.5 percentile
        upper_bound = np.percentile(r_values, 97.5)  # 97.5 percentile

        # Step 3: Display the results
        print(f"Pearson's r (bootstrapped): Median = {np.median(r_values):.3f}, 95% Confidence Interval: ({lower_bound:.3f}, {upper_bound:.3f})")

        # Step 1: Calculate Pearson's r for each bootstrap sample
        r_values = np.array([spearmanr(np.arange(len(data)),data)[0] for data in bootstrap_vals])

        # Step 2: Calculate the 95% confidence interval
        alpha = 0.05  # 5% significance level
        lower_bound = np.percentile(r_values, 2.5)  # 2.5 percentile
        upper_bound = np.percentile(r_values, 97.5)  # 97.5 percentile

        # Step 3: Display the results
        print(f"Spearman's r (bootstrapped): Median = {np.median(r_values):.3f}, 95% Confidence Interval: ({lower_bound:.3f}, {upper_bound:.3f})")
    
        from scipy.optimize import curve_fit
        from sklearn.metrics import mean_squared_error, r2_score
        def quadratic(x, a,b,c): 
            return  a * x**2 + b * x + c

        # 3. Fit the curve to the data using curve_fit
        r_vals = []
        for data in bootstrap_vals:
            params, covariance = curve_fit(quadratic,np.arange(len(data)),data)

            # 4. Generate fitted y values from the quadratic model
            y_fit = quadratic(np.arange(len(data)), *params)
        
            r_vals.append(r2_score(data, y_fit))
        r_vals = np.array(r_vals)
        lower_bound = np.percentile(r_vals, 2.5)  # 2.5 percentile
        upper_bound = np.percentile(r_vals, 97.5)  # 97.5 percentile

        # Step 3: Display the results
        print(f"Curve Linear's r (bootstrapped): Median = {np.median(r_vals):.3f}, 95% Confidence Interval: ({lower_bound:.3f}, {upper_bound:.3f})")
    plt.show()


if age_vs_sleep_depth:
    # Group the Age column into 10-year bins, starting at 18, with 80+ as the final bin
    bins = [18] + list(range(20, 81, 10)) + [float('inf')]  # Bins [18-19, 20-29, ..., 70-79, 80+]
    labels = ['18-19'] + [f'{i}-{i+9}' for i in range(20, 80, 10)] + ['80+']  # Corresponding labels

    df_test_all['Age_Group'] = pd.cut(df_test_all['age'], bins=bins, labels=labels, right=False)

    # Melt the dataframe to long format for both OSD and ORP values
    df_melted_osd = pd.melt(df_test_all, id_vars=['Age_Group'], value_vars=['OSD_N3', 'OSD_N2', 'OSD_N1', 'OSD_R', 'OSD_W', 'OSD_NR'],
                            var_name='Sleep_Stage', value_name='Value')
    df_melted_orp = pd.melt(df_test_all, id_vars=['Age_Group'], value_vars=['ORP_N3', 'ORP_N2', 'ORP_N1', 'ORP_R', 'ORP_W', 'ORP_NR'],
                            var_name='Sleep_Stage', value_name='Value')

    # Map the sleep stage to readable labels for both
    sleep_stage_mapping_osd = {'OSD_N3': 'N3', 'OSD_N2': 'N2', 'OSD_N1': 'N1', 'OSD_R': 'REM', 'OSD_W': 'Wake', 'OSD_NR': 'NR'}
    df_melted_osd['Sleep_Stage'] = df_melted_osd['Sleep_Stage'].map(sleep_stage_mapping_osd)

    sleep_stage_mapping_orp = {'ORP_N3': 'N3', 'ORP_N2': 'N2', 'ORP_N1': 'N1', 'ORP_R': 'REM', 'ORP_W': 'Wake', 'ORP_NR': 'NR'}
    df_melted_orp['Sleep_Stage'] = df_melted_orp['Sleep_Stage'].map(sleep_stage_mapping_orp)

    # Create a 2x6 grid of subplots (6 for OSD, 6 for ORP)
    fig, axes = plt.subplots(2, 6, figsize=(18, 12), sharey='row')

    # Plot OSD results in the first row
    sleep_stages = ['N3', 'N2', 'N1', 'REM', 'Wake', 'NR']
    for i, stage in enumerate(sleep_stages):
        sns.boxplot(x='Sleep_Stage', y='Value', hue='Age_Group', data=df_melted_osd[df_melted_osd['Sleep_Stage'] == stage],
                    ax=axes[0, i], palette='Blues', showfliers=False)
        axes[0, i].set_title(f'OSD - {stage}')
        axes[0, i].grid(True, linestyle='--', alpha=0.7)
        axes[0, i].set_axisbelow(True)  # Ensure gridlines are behind the bars
        axes[0, i].legend().set_visible(False)  # Hide individual legends
        axes[0, i].set_xticklabels([stage])  # Set x-axis label to the stage only
        axes[0, i].set_xlabel('')  # Remove x label
        axes[0, i].set_ylabel('')  # Remove y label
        if i == 0:
            axes[0, i].set_ylabel('OSD')  # Set y label for OSD only for the first column

    # Plot ORP results in the second row
    for i, stage in enumerate(sleep_stages):
        sns.boxplot(x='Sleep_Stage', y='Value', hue='Age_Group', data=df_melted_orp[df_melted_orp['Sleep_Stage'] == stage],
                    ax=axes[1, i], palette='Blues', showfliers=False)
        axes[1, i].set_title(f'ORP - {stage}')
        axes[1, i].grid(True, linestyle='--', alpha=0.7)
        axes[1, i].set_axisbelow(True)  # Ensure gridlines are behind the bars
        axes[1, i].legend().set_visible(False)  # Hide individual legends
        axes[1, i].set_xticklabels([stage])  # Set x-axis label to the stage only
        axes[1, i].set_ylabel('')  # Remove y label
        if i == 0:
            axes[1, i].set_ylabel('ORP')  # Set y label for ORP only for the first column

    # Add a single legend for both rows outside the last plot with bounding box
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Age Group', 
            title_fontsize='13', fontsize='11')

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.92)
    plt.close()

if hypno_kappa:
    # Prepare the DataFrame
    df_hypno_corr = df_test_all[['spearman_OSD',  'spearman_ORP', 'spearman_OSD_30', 'spearman_ORP_30']].copy()
    s_osd = df_test_all['spearman_OSD'].values
    s_osd_30 = df_test_all['spearman_OSD_30'].values
    s_orp = df_test_all['spearman_ORP'].values
    s_orp_30 = df_test_all['spearman_ORP_30'].values


    length = np.arange(len(df_hypno_corr))
    S_osd = []
    S_osd_30 = []
    S_orp = []
    S_orp_30 = []

    for i in range(1000):
        selected_idx = random.choices(length, k=len(length))
        S_osd.append(np.nanmedian(s_osd[selected_idx]))
        S_osd_30.append(np.nanmedian(s_osd_30[selected_idx]))
        S_orp.append(np.nanmedian(s_orp[selected_idx]))
        S_orp_30.append(np.nanmedian(s_orp_30[selected_idx]))

    print(f'median OSD {np.median(S_osd)} [{np.nanpercentile(S_osd, 2.5)},{np.nanpercentile(S_osd, 97.5)}]')
    print(f'median OSD_30 {np.median(S_osd_30)} [{np.nanpercentile(S_osd_30, 2.5)},{np.nanpercentile(S_osd_30, 97.5)}]')
    print(f'median ORP {np.median(S_orp)} [{np.nanpercentile(S_orp, 2.5)},{np.nanpercentile(S_orp, 97.5)}]')
    print(f'median ORP_30 {np.median(S_orp_30)} [{np.nanpercentile(S_orp_30, 2.5)},{np.nanpercentile(S_orp_30, 97.5)}]')

    # Melt the DataFrame to long format
    df_melted = pd.melt(df_hypno_corr, 
                        var_name='Correlation_Type', 
                        value_name='Value')

    # Map the correlation types to readable labels
    df_melted['Algorithm'] = df_melted['Correlation_Type'].apply(lambda x: 'OSD' if 'OSD' in x else 'ORP')
    df_melted['Type'] = df_melted['Correlation_Type'].apply(lambda x: 'Spearman' if 'spearman' in x else 'Pearson')

    # Create a new column to separate 3 and 30 second windows
    df_melted['Window'] = df_melted['Correlation_Type'].apply(lambda x: '3 seconds' if '_30' not in x else '30 seconds')

    # Plot with time window as an additional hue
    plt.figure(figsize=(8, 6))
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    sns.boxplot(x='Window', y='Value', hue='Algorithm', data=df_melted, palette='Blues', showfliers=False, zorder=1)

    # Set labels and title
    plt.title('Hypnogram Correlation (Spearman)', fontsize=20)
    plt.ylabel('Correlation Value', fontsize=14)
    plt.xlabel('Calculation Window', fontsize=14)

    # Legend
    plt.legend(title='Algorithm', title_fontsize='13', fontsize='11', loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and save
    plt.tight_layout()
    plt.show()
    plt.close()


    
    #calc cm features
    futures = [cm_calc.remote(file,'Label','categorical_pred',labels=np.arange(5)+1,mask=masked) for file in tqdm(files)]
    #futures = [cm_calc.remote(file,'Label','class_pred') for file in tqdm(files)]
    cm = ray.get(futures)
    CM_class = np.zeros((5,5))   
    for cm_tmp in cm:
        CM_class+=cm_tmp

    #calc cm features
    futures = [cm_calc.remote(file,'Label','ordinal_pred',labels=np.arange(5)+1,mask=masked) for file in tqdm(files)]
    #futures = [cm_calc.remote(file,'Label','ordinal_pred') for file in tqdm(files)]
    cm = ray.get(futures)
    CM_ordinal = np.zeros((5,5))   
    for cm_tmp in cm:
        CM_ordinal+=cm_tmp
        
    
    #switch rows and remove rem
    CM_ordinal[[3,4],:] = CM_ordinal[[4,3],:]
    CM_ordinal = CM_ordinal[:-1,:-1]
    #clean-up workspace
    del cm,futures,cm_tmp

    # #plot

    CM_class = CM_class[[0,1,2,3,4],:]
    CM_class = CM_class[:,[0,1,2,3,4]]
    plot_confusion_matrix(CM_class,target_names_class,title='CM classwise',cmap=None,normalize=True)
    plt.savefig(f'/media/erikjan/Expansion/OSD/figures/{model_version}/{evaluation_part}/CM_Classwise_masked_{masked}.png')


    #CM_ordinal = CM_ordinal[[0,1,2,4],:-1]
    plot_confusion_matrix(CM_ordinal,target_names_ordinal,title='CM Ordinal',cmap=None,normalize=True)
    plt.savefig(f'/media/erikjan/Expansion/OSD/figures/{model_version}/{evaluation_part}/CM_Ordinal_masked_{masked}.png')
    plt.show()
    plt.close()

    print(f'ACCURACY CLASS :{np.sum((CM_class[[0,1,2,3,4],[0,1,2,3,4]]))/np.sum(CM_class)}')
    print(f'ACCURACY ORDINAL :{np.sum((CM_ordinal[[0,1,2,3],[0,1,2,3]]))/np.sum(CM_ordinal)}')
    print('a')