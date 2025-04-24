from utils.models.OSD_architecture import *
from utils.data_loaders.custom_data_loader import *
from utils.data_loaders.data_loader_preparation import *
from utils.data_loaders.loader_functions import *
from utils.hyperparameters.custom_callbacks import scheduler
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#################
# Define vars
#################
WORKING_DIRECTORY = '/Users/erik-janmeulenbrugge/Documents/EJ_REPOS/OSD_repo/OSD_repo'#'/user/your/path/to/OSD_repo'
TEST_RUN = False
RUN_NAME = '_test_run'
project_dir = f'{WORKING_DIRECTORY}/'
save_weights_path = project_dir+'OSD/utils/models/newly_trained/OSD'+RUN_NAME+'/'
os.makedirs(save_weights_path,exist_ok=True)


#################
# Load Model
#################
model = OSD_architecture()
print(model.summary())

#################
# Define data for loaders
#################
#batchsize = # of 3 second epochs per label --> batchsize of 64 = 64xwake + 64xN1 ect... 
if TEST_RUN:
    input_dataloader = data2loader(f'{WORKING_DIRECTORY}/OSD/utils/data_split/train.csv',batch_size=32,max_pat=64,prefix=f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/')
    input_dataloader_val = data2loader(f'{WORKING_DIRECTORY}/OSD/utils/data_split/val.csv',batch_size=32,max_pat=10,prefix=f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/')
else:
    input_dataloader = data2loader(f'{WORKING_DIRECTORY}/OSD/utils/data_split/train.csv',batch_size=32,prefix=f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/')
    input_dataloader_val = data2loader(f'{WORKING_DIRECTORY}/OSD/utils/data_split/val.csv',batch_size=32,prefix=f'{WORKING_DIRECTORY}/OSD_data/Pre-Processed/')
    
#################
# Define Dataloaders
#################
train_loader = DataGenerator_OSD(input_dataloader,batches_per_epoch=600)
val_loader = DataGenerator_OSD_val(input_dataloader)

#################
# Define callbacks
#################
callbacks = [
    tf.keras.callbacks.CSVLogger(project_dir+f'/logs/training_OSD_{RUN_NAME}.csv',separator=",", append=True),
    tf.keras.callbacks.ModelCheckpoint(save_weights_path+'OSD-{epoch:02d}-{val_loss:.4f}-.weights.h5',verbose=0,save_best_only=False,save_weights_only=True,mode="max",save_freq="epoch"),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=25,verbose=0,mode="min"),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)]

#################
# Train
#################
history = model.fit(train_loader,
                    validation_data =val_loader,
                    epochs=800,
                    callbacks=callbacks
                    ) 
