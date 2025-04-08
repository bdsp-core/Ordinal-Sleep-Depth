from utils.models.OSD_architecture import *
from utils.data_loaders.custom_data_loader_MGH import *
from utils.data_loaders.data_loader_preparation import *
from utils.data_loaders.loader_functions import *
from utils.hyperparameters.custom_callbacks import scheduler
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#################
# Define vars
#################
WORKING_DIRECTORY = '/user/your/path/to/OSD_repo'
TEST_RUN = True
RUN_NAME = ''
project_dir = f'{WORKING_DIRECTORY}/OSD'
save_weights_path = project_dir+'/models/newly_trained/OSD'+RUN_NAME+'/'
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
    input_dataloader = data2loader(f'{WORKING_DIRECTORY}/OSD/data_splits/MGH_train_osd.csv',batch_size=32,max_pat=64,prefix=f'{WORKING_DIRECTORY}/DATA_OSD/')
    input_dataloader_val = data2loader(f'{WORKING_DIRECTORY}/OSD/data_splits/MGH_val_osd.csv',batch_size=32,max_pat=10,prefix=f'{WORKING_DIRECTORY}/DATA_OSD/')
else:
    input_dataloader = data2loader(f'{WORKING_DIRECTORY}/OSD/data_splits/MGH_train_osd.csv',batch_size=32,prefix=f'{WORKING_DIRECTORY}/DATA_OSD/')
    input_dataloader_val = data2loader(f'{WORKING_DIRECTORY}/OSD/data_splits/MGH_val_osd.csv',batch_size=32,prefix=f'{WORKING_DIRECTORY}/DATA_OSD/')
    
#################
# Define Dataloaders
#################
train_loader = MGH_DataGenerator_OSD(input_dataloader,batches_per_epoch=600)
val_loader = MGH_DataGenerator_OSD_val(input_dataloader)

#################
# Define callbacks
#################
callbacks = [
    tf.keras.callbacks.CSVLogger(project_dir+f'/logs/training_OSD_{RUN_NAME}.csv',separator=",", append=True),
    tf.keras.callbacks.ModelCheckpoint(save_weights_path+'OSD-{epoch:02d}-{val_loss:.4f}-.h5',verbose=0,save_best_only=False,save_weights_only=True,mode="max",save_freq="epoch"),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=25,verbose=0,mode="min"),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)]

#################
# Train
#################
history = model.fit(train_loader,
                    validation_data =val_loader,
                    use_multiprocessing=True,
                    workers=25,
                    epochs=800,
                    callbacks=callbacks
                    ) 
