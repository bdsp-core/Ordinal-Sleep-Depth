import tensorflow as tf 
from  utils.data_loaders.loader_functions import *
import collections
from scipy import stats as st

class DataGenerator_OSD(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,batches_per_epoch=500):
        self.dataset_params = dataset_params
        self.batch_size = len(dataset_params['label_distribution_batch'])
        self.batches_per_epoch = batches_per_epoch
        self.n_channels = 6
        self.data_hz = 200
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['patients'], self.dataset_params['selection_chance'],batch_size=self.batch_size)
                
                Xdata, label = load_patients(self,pat_list,self.dataset_params['label_distribution_batch'],batch_size=self.batch_size)
                succes=True
            except:
                pass

            label = np.append(np.arange(6),label)
            label = tf.keras.utils.to_categorical(label)[6:, 1:]
            label_o = label[:,[0,1,2,4]]

        return Xdata, (label,label_o)


    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['label_distribution_batch'])
        np.random.shuffle(self.dataset_params['patients'])

class DataGenerator_OSD_val(tf.keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self,dataset_params):
        self.dataset_params = dataset_params
        self.batch_size = len(dataset_params['patients'])
        self.n_channels = 6
        self.data_hz = 200
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batch_size 

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                Xdata, label = load_patients_val(self,index,self.dataset_params['patients'])
                label = np.append(np.arange(6),label)
                label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
                label_o = label[:,[0,1,2,4]]

                return Xdata, (label,label_o)
                

            
            except:
                label = np.append((np.arange(6),0))
                label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
                print('error')
                return np.zeros((1,600,6)), (np.zeros((1,5)),np.zeros((1,4)))



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['label_distribution_batch'])
        np.random.shuffle(self.dataset_params['patients'])

