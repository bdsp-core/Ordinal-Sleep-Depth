import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy import stats as st

def get_patients(data_list_per_set, chance_list, batch_size=64):
    pat_list_idx = np.random.choice(len(data_list_per_set),batch_size*2,replace=False,p=chance_list)
    pat_list = [data_list_per_set[int(x)] for x in pat_list_idx]
    return pat_list

def load_patients(self,pat_list,start_label,batch_size=64):
    
    #pre-allocate data
    label_start = [5,4,3,2,1]
    EEG = np.zeros((batch_size*len(label_start),600,6))
    LABEL = np.zeros((batch_size*len(label_start)))
    


    #loop though patients
    i = 0
    success = 0
    while success<len(label_start):
        
        pat =pat_list[i]

        try:
            with h5.File(pat,'r') as f:
                #load
                label = f['annotations']['STAGE'][:]
                label_idx = np.where(label==label_start[success])[0]
                if len(label_idx)==0:
                    i+=1 
                    continue

                #find start of each label
                loc = np.where(np.diff(label_idx)>1)[0]+1
                label_idx_start = label_idx[loc]
                label_idx_start = np.hstack((label_idx[0],label_idx_start))
                #find start of each label
                label_idx_end = label_idx[loc-1]
                label_idx_end = np.hstack((label_idx_end,label_idx[-1]))
                #find length of each segment
                len_1 = label_idx_end-label_idx_start
                parting = len_1//(64*600)
                parting[parting==0]=1
                #if big, cut in parts
                for idx,part in zip(label_idx_start,parting):
                    if part >1:
                        for extra in range(part-1):
                            label_idx_start = np.hstack((label_idx_start,idx+((extra+1)*64*600)))
                            label_idx_end = np.hstack((label_idx_end,idx+((extra+1)*64*600)-1))
                
                #sort
                label_idx_start = np.array(sorted(label_idx_start))
                label_idx_end = np.array(sorted(label_idx_end))

                #if segmetn smaller than batch size, assign lower selection chance
                len_2 = label_idx_end-label_idx_start
                len_2[len_2>38399]=38399
                chance = len_2/sum(len_2)

                #make sure its not close to the end
                label_idx_start[label_idx_start>(np.max(label_idx_start)-(600*self.batch_size)-1)]=(np.max(label_idx_start)-(600*self.batch_size)-1)

                #get random start index
                start_idx = np.random.choice(label_idx_start,p=chance)

                #check for nan
                if np.any(np.isnan(f['annotations']['STAGE'][start_idx:start_idx+600*self.batch_size])):
                    i+=1 
                    continue

                #create data
                data = np.zeros((6,600*self.batch_size))
                for i_chan,chan in enumerate(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']):
                    data[i_chan,:] = np.squeeze(np.array(f['channels'][chan][start_idx:start_idx+600*self.batch_size]))

                data = data.T.reshape(self.batch_size,600,6)#*1000000
                # data[data>500]=500
                # data[data<-500]=-500

                if np.any(np.isnan(data)):
                    i+=1 
                    continue
                else:
                    #load in data
                    label = st.mode(np.squeeze(np.array(f['annotations']['STAGE'][start_idx:start_idx+600*self.batch_size])).reshape(self.batch_size,600).T)[0]

                    #write data in batch matrix
                    LABEL[success*self.batch_size:(success+1)*self.batch_size] = label
                    EEG[success*self.batch_size:(success+1)*self.batch_size,:,:] = data
                    i+=1 
                    success+=1

        except:
            i+=1 

        # for j in range(6):
        #     plt.plot(EEG[280,:,j]-100*j,color='b',linewidth=0.5)
        # plt.show()

    #filter out nans from labels
    non_nan_idx_label = ~np.isnan(LABEL)
    LABEL = LABEL[non_nan_idx_label]
    EEG = EEG[non_nan_idx_label,:,:]

    #filter out 0's from label
    non_zero_idx_label = np.where(LABEL!=0)[0]
    LABEL = LABEL[non_zero_idx_label]
    EEG = EEG[non_zero_idx_label,:,:]

    #filter out nans from data
    non_nan_idx_data = np.any(np.any(~np.isnan(EEG),axis=2),axis=1)
    LABEL = LABEL[non_nan_idx_data]
    EEG = EEG[non_nan_idx_data,:,:]

    return EEG, LABEL


def load_patients_rand(self,pat_list,start_label,batch_size=64):
    
    #pre-allocate data
    label_start = [5,4,3,2,1]
    EEG = np.zeros((batch_size*len(label_start),600,6))
    LABEL = np.zeros((batch_size*len(label_start)))
    


    #loop though patients
    i = 0
    success = 0
    while success<len(label_start):
        
        pat =pat_list[i]

        try:
            with h5.File(pat,'r') as f:

                start_idx = np.random.choice(len(f['channels']['F3-M2'])-(self.batch_size*600))
                #create data
                data = np.zeros((6,600*self.batch_size))
                for i_chan,chan in enumerate(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']):
                    data[i_chan,:] = np.squeeze(np.array(f['channels'][chan][start_idx:start_idx+600*self.batch_size]))

                data = data.T.reshape(self.batch_size,600,6)*1000000
                data[data>500]=500
                data[data<-500]=-500

                if np.any(np.isnan(data)):
                    i+=1 
                    continue
                else:
                    #load in data
                    label = st.mode(np.squeeze(np.array(f['annotations']['stage_expert_0'][start_idx:start_idx+600*self.batch_size])).reshape(self.batch_size,600).T)[0]

                    #write data in batch matrix
                    LABEL[success*self.batch_size:(success+1)*self.batch_size] = label
                    EEG[success*self.batch_size:(success+1)*self.batch_size,:,:] = data
                    i+=1 
                    success+=1

        except:
            i+=1 

        # for i in range(6):
        #     plt.plot(EEG[80,:,i]-100*i,color='b',linewidth=0.5)
        # plt.show()

    #filter out nans from labels
    non_nan_idx_label = ~np.isnan(LABEL)
    LABEL = LABEL[non_nan_idx_label]
    EEG = EEG[non_nan_idx_label,:,:]

    #filter out 0's from label
    non_zero_idx_label = np.where(LABEL!=0)[0]
    LABEL = LABEL[non_zero_idx_label]
    EEG = EEG[non_zero_idx_label,:,:]

    #filter out nans from data
    non_nan_idx_data = np.any(np.any(~np.isnan(EEG),axis=2),axis=1)
    LABEL = LABEL[non_nan_idx_data]
    EEG = EEG[non_nan_idx_data,:,:]

    return EEG, LABEL

def load_patients_val(self,index,pat_list,):
    
        
    pat =pat_list[index]

    try:
        with h5.File(pat,'r') as f:

            #load label
            label = np.squeeze(np.array(f['annotations']['STAGE'][:]))
            
            #load data
            data = np.zeros((6,len(f['channels']['F3-M2'][:])))
            for i_chan,chan in enumerate(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']):
                data[i_chan,:] = np.squeeze(np.array(f['channels'][chan][:]))

        # #get data where label is not 0
        # label_idx = np.where(label!=0)[0]
        # label = label[label_idx]
        # data = data[:,label_idx]

        #clip len on k*600
        max_len = len(label)//600*600
        label = label[:max_len]
        data = data[:,:max_len]

        #reshape data 
        # data = data*1000000
        # data[data>500]=500
        # data[data<-500]=-500
        data = data.T.reshape(-1,600,6)

        #reshape labels
        label = st.mode(label.reshape(-1,600).T)[0]
        label = np.squeeze(label)

        #filter out nans from labels
        non_nan_idx_label = np.squeeze(~np.isnan(label))
        label = label[non_nan_idx_label]
        data = data[non_nan_idx_label,:,:]

        #filter out 0's from label
        non_zero_idx_label = np.where(label!=0)[0]
        label = label[non_zero_idx_label]
        data = data[non_zero_idx_label,:,:]

        #filter out nans from data
        non_nan_idx_data = np.any(np.any(~np.isnan(data),axis=2),axis=1)
        label = label[non_nan_idx_data]
        data = data[non_nan_idx_data,:,:]

        # for i in range(6):
        #     plt.plot(data[80,:,i]-100*i,color='b',linewidth=0.5)
        # plt.show()

        return data, label

    except:
        return np.zeros((1,600,6)) , 0 

        

def load_patients_val_yield(pat):
    
        
    try:
        with h5.File(pat,'r') as f:

            #load label
            label = np.squeeze(np.array(f['annotations']['stage_expert_0'][:]))
            
            #load data
            data = np.zeros((6,len(f['channels']['F3-M2'][:])))
            for i_chan,chan in enumerate(['F3-M2', 'F4-M1','C3-M2','C4-M1','O1-M2', 'O2-M1']):
                data[i_chan,:] = np.squeeze(np.array(f['channels'][chan][:]))

        # #get data where label is not 0
        # label_idx = np.where(label!=0)[0]
        # label = label[label_idx]
        # data = data[:,label_idx]

        #clip len on k*600
        max_len = len(label)//600*600
        label = label[:max_len]
        data = data[:,:max_len]

        #reshape data 
        data = data*1000000
        data[data>500]=500
        data[data<-500]=-500
        data = data.T.reshape(-1,600,6)

        #reshape labels
        label = st.mode(label.reshape(-1,600).T)[0]
        label = np.squeeze(label)

        #filter out nans from labels
        non_nan_idx_label = np.squeeze(~np.isnan(label))
        label = label[non_nan_idx_label]
        data = data[non_nan_idx_label,:,:]

        #filter out 0's from label
        non_zero_idx_label = np.where(label!=0)[0]
        label = label[non_zero_idx_label]
        data = data[non_zero_idx_label,:,:]

        #filter out nans from data
        non_nan_idx_data = np.any(np.any(~np.isnan(data),axis=2),axis=1)
        label = label[non_nan_idx_data]
        data = data[non_nan_idx_data,:,:]

        # for i in range(6):
        #     plt.plot(data[80,:,i]-100*i,color='b',linewidth=0.5)
        # plt.show()

        return data, label

    except:
        pass

        