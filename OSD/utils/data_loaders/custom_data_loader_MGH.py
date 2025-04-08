import tensorflow as tf 
from  utils.data_loaders.loader_functions import *
import collections
from scipy import stats as st

def find_mode(array):
    counts = collections.Counter(array)
    maximum = max(counts.values())
    return [key for key, value in counts.items()
            if value == maximum]

class MGH_DataGenerator_OSD_h5(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset,batchsize):
        self.batchsize = batchsize
        self.dataset = dataset
        self.n_channels = 6
        self.data_hz = 200
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.dataset['X_train'])/self.batchsize)

    def __getitem__(self, index):
        succes = False
        Xdata = np.array(self.dataset['X_train'][index*self.batchsize:(index+1)*self.batchsize,:,:])       
        label = np.array(self.dataset['Y_train'][index*self.batchsize:(index+1)*self.batchsize,:])  
        label = np.squeeze(st.mode(label,axis=1)[0])
        label = np.append(np.arange(6),label)
        label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
        label_o = label[:,[0,1,2,4]]

        return Xdata, [label,label_o]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass


class MGH_DataGenerator_OSD(tf.keras.utils.Sequence):
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
            label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
            label_o = label[:,[0,1,2,4]]

        return Xdata, [label,label_o]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['label_distribution_batch'])
        np.random.shuffle(self.dataset_params['patients'])

class MGH_DataGenerator_OSD_val(tf.keras.utils.Sequence):

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

                return Xdata, [label,label_o]
            
            except:
                label = np.append((np.arange(6),0))
                label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
                print('error')
                return np.zeros((1,600,6)), [np.zeros((1,5)),np.zeros((1,4))]


    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['label_distribution_batch'])
        np.random.shuffle(self.dataset_params['patients'])


######################################
# old
#######################################

class MGH_DataGenerator_OSD_rand(tf.keras.utils.Sequence):
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
                
                Xdata, label = load_patients_rand(self,pat_list,self.dataset_params['label_distribution_batch'],batch_size=self.batch_size)
                succes=True
            except:
                pass

            label = np.append(np.arange(6),label)
            label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]
            label_o = label[:,[0,1,2,4]]

        return Xdata, [label,label_o]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['label_distribution_batch'])
        np.random.shuffle(self.dataset_params['patients'])


class MGH_DataGenerator_OSD_train_h5(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data,label,batch_size=64,batches_per_epoch=500):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.n_channels = 6
        self.data_hz = 200
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data)/self.batch_size)

    def __getitem__(self, index):

        Xdata = self.data[index*self.batch_size:index*self.batch_size+64,:,:][()]
        Xdata = Xdata*1000000
        Xdata[Xdata>500]=500
        Xdata[Xdata<-500]=-500

        label = self.label[index*self.batch_size:index*self.batch_size+64,:,0][()]

        label = tf.keras.utils.to_categorical(label,dtype='int',num_classes=6)[:,0,1:]
        label_o = label[:,[0,1,2,4]]

        return Xdata, label #[label,label_o]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
        

from multiprocessing import current_process
import time
import random

def custom_data_generator(dataset_params,hparams):
    
    i = 0
    run = 0
    while True:


        #random seed
        process_id = current_process()._identity[0] if current_process()._identity else 1
        now = int(time.time() * 1e23)
        seed = (process_id+run + now ) % (2 ** 32)
        np.random.seed(seed)

        #generate list for epoch
        pat_list =  dataset_params['patients']
        
        #loop over patients
        while i<=len(pat_list):
            
            #load Patients
            Xdata, label = load_patients_val_yield(dataset_params['patients'][i])
            
            #crop patients
            Xdata_len = Xdata.shape[0]//hparams['batch_size']*hparams['batch_size']
            Xdata = Xdata[:Xdata_len,:,:]
            label = label[:Xdata_len]
            label = np.append(np.arange(6),label)
            label = tf.keras.utils.to_categorical(label,dtype='int')[6:,1:]


            #loop over batches:
            for j in range(Xdata.shape[0]//hparams['batch_size']-2):
                go = False
                J = j*hparams['batch_size']
                Xdata_return = Xdata[J:J+hparams['batch_size'],:,:]
                label_return = label[J:J+hparams['batch_size'],:]

                if Xdata_return.shape == (hparams['batch_size'],600,6):
                    if label_return.shape == (hparams['batch_size'],5):
                        go=True

                if go:
                    yield Xdata_return , label_return
            
            #go to next patient
            i+=1 
            if  i%len(pat_list) ==0:
                i = i%len(pat_list)
                np.random.shuffle(dataset_params['patients'])
