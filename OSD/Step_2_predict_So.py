from utils.models.OSD_architecture import *
from utils.data_loaders.custom_data_loader_MGH import *
from utils.data_loaders.data_loader_preparation import *
from utils.data_loaders.loader_functions import *
import os
from os.path import join as opj
from tqdm import tqdm
import gc 

from keras.backend  import clear_session
from keras.backend  import get_session

os.environ["CUDA_VISIBLE_DEVICES"]="1"

##################
# Definitions
##################

# Reset Keras Session
def reset_keras(weights_path):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    print(gc.collect()) # if it's done something you should see a number being outputted

#################
# Define vars
#################
WORKING_DIRECTORY = '/user/your/path/to/OSD_repo'

splits = ['test']
for split in splits:
    project_dir = f'{WORKING_DIRECTORY}/OSD'
    model_version = 'OSD_MODEL' 
    weights_path = f'{WORKING_DIRECTORY}/OSD/utils/models/weights/OSD_weights.h5'
    write_path = f'{WORKING_DIRECTORY}/OSD/predictions/{model_version}/{split}/'
    os.makedirs(write_path,exist_ok=True)

    #load model
    model = OSD_architecture()
    model.load_weights(weights_path)

    print(model.summary())

    #batchsize = # of 3 second epochs per label --> batchsize of 64 = 64xwake + 64xN1 ect... 
    input_dataloader = data2loader(f'/media/erikjan/Expansion/OSD/data_splits/MGH_{split}_osd (copy).csv',batch_size=64,max_pat=None,prefix='/media/erikjan/Expansion/OSD/Datasets/Pre-processed_EJ/OSD_MGH/')#TODO

    files = input_dataloader['patients']

    count = 0
    for file in tqdm(files):
        if os.path.exists(opj(write_path,file.split('/')[-1][:-3]+'.csv'))==False:
            try:
                with h5.File(file,'r') as f:
                    label = f['annotations']['STAGE'][:]
                    arousal = f['annotations']['AROUSAL'][:]

                    eeg = np.zeros((len(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']),len(f['channels']['F3-M2'])))
                    for i_chan,chan in enumerate(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']):
                        eeg[i_chan,:] = np.squeeze(np.array(f['channels'][chan]))
                
                # predict
                len_eeg = eeg.shape[1]
                eeg_fit = eeg[:,:eeg.shape[1]//600*600]
                ordinal = model.predict(eeg_fit.reshape(6,600,-1,order='F').T)

                ordinal_ = ordinal[1]

                #argmax for ordinal
                ord_pred = np.argmax(np.array(ordinal_),axis=1)+1
                ordinal_w = [x[3] for x in ordinal_]
                ordinal_n1 = [x[2] for x in ordinal_]
                ordinal_n2 = [x[1] for x in ordinal_]
                ordinal_n3 = [x[0] for x in ordinal_]

                pred = np.argmax(np.array(ordinal[0]),axis=1)+1
                categorical_w = [x[4] for x in ordinal[0]]
                categorical_r = [x[3] for x in ordinal[0]]
                categorical_n1 = [x[2] for x in ordinal[0]]
                categorical_n2 = [x[1] for x in ordinal[0]]
                categorical_n3 = [x[0] for x in ordinal[0]]


                #get correct size for label 
                label = label[np.arange(0,len(label),600)]
                label = label[:len(ord_pred)]
                arousal = [np.max(arousal[x:x+600]) for x in np.arange(0,len(arousal),600) if len(arousal)-x>=600]
                label
                #to dict
                data = {'Label':label,
                        'Arousal':arousal,
                        'ordinal_pred':ord_pred,
                        'ordinal_pred_w':ordinal_w,
                        'ordinal_pred_n1':ordinal_n1,
                        'ordinal_pred_n2':ordinal_n2,
                        'ordinal_pred_n3':ordinal_n3,
                        'categorical_pred': pred,
                        'categorical_pred_w': categorical_w,
                        'categorical_pred_r': categorical_r,
                        'categorical_pred_n1':categorical_n1,
                        'categorical_pred_n2':categorical_n2,
                        'categorical_pred_n3':categorical_n3}


                #save
                df = pd.DataFrame(data)
                df.to_csv(opj(write_path,file.split('/')[-1][:-3]+'.csv'))

                count+=1
                if count>100:
                    reset_keras(weights_path)
                    count=0
                    del model
                    model = OSD_architecture()
                    model.load_weights(weights_path)
                
            except:
                print('error')


    print('a')
