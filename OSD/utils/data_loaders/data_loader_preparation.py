import numpy as np
from glob import glob
import pandas as pd

def data2loader(path_to_all_datasets,batch_size=64,max_pat=None,prefix=None):
    
    df_files = pd.read_csv(path_to_all_datasets)
    #df_files = df_files[df_files['Predicted_Stage']=='No Dementia']
    pat_paths = df_files['fileid'].to_numpy().tolist()

    if prefix:
       pat_paths =  [prefix+x+'/'+x+'.h5' for x in pat_paths]

    if max_pat!=None:
        pat_paths = pat_paths[:max_pat]

    #initialize selection change 
    #only one dataste so 1/len(files)
    chance_list = [1/len(pat_paths)]*(len(pat_paths))
    
    #create stage vector to ensure all stages are presented in batch
    start_label = np.repeat(np.array((1,2,3,4,5)),np.ceil(batch_size/5))
    if max_pat!=None:
        start_label = start_label[:max_pat]


    #pop values if necesary
    while len(start_label)>batch_size:
        rand_pop = np.random.randint(0,high=len(start_label))
        start_label = np.delete(start_label,rand_pop)
    
    dataset_params = {'patients': pat_paths,'selection_chance':chance_list, 'label_distribution_batch':start_label}
    return dataset_params