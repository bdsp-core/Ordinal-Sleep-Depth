import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,Dropout,Flatten,Dense, Softmax
from tensorflow.keras.layers import Activation
from .custom_layers.ordistic_regression import *
from keras import backend as K


# crossentropy cut
def ct_cut(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask = tf.cast(tf.greater_equal(y_true_f, -0.5), dtype='float32')
    out = -(y_true_f * tf.math.log(y_pred_f) * mask + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f) * mask)
    out = tf.reduce_mean(out)  # updated from tf.mean to tf.reduce_mean
    return out


# crossentropy 
def ct_keras(y_true, y_pred):
    return tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

# weighted crossentropy cut
def wc_ct_cut_5(y_true, y_pred):
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')

    w0 = 2
    w1 = 1
    w2 = 12
    w3 = 2
    w4 = 4

    l0 = ct_cut(y_true[:, 0], y_pred[:, 0])  # n3
    l1 = ct_cut(y_true[:, 1], y_pred[:, 1])  # n2
    l2 = ct_cut(y_true[:, 2], y_pred[:, 2])  # n1
    l3 = ct_cut(y_true[:, 3], y_pred[:, 3])  # rem
    l4 = ct_cut(y_true[:, 4], y_pred[:, 4])  # wake

    out = (w0 * l0 + w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4) / (w0 + w1 + w2 + w3 + w4)  # set custom weights for each class
    return out


#weighted crossentropy cut
def wc_ct_cut_4(y_true,y_pred):
    y_true=tf.cast(y_true,dtype='float32')
    y_pred=tf.cast(y_pred,dtype='float32')
    
    w0=2
    w1=1
    w2=12
    w4=1

    l0=ct_cut(y_true[:,0],y_pred[:,0]) * w0# n3
    l1=ct_cut(y_true[:,1],y_pred[:,1]) * w1 # n2
    l2=ct_cut(y_true[:,2],y_pred[:,2]) * w2 # n1
    l4=ct_cut(y_true[:,3],y_pred[:,3]) * w4 # wake


    out = (l0  + l1  + l2  + l4 )/(w0+w1+w2+w4)  # set custom weights for each class
    return out

def OSD_architecture():
    
    ##################### start model #####################
    inputs = Input(shape=(600,6))

    #layer 1 (600x6)
    x1 = Conv1D(64, 2^4+1, padding='same')(inputs)
    x1 = BatchNormalization(axis=1,momentum=0.99)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=2,padding='same')(x1)
    x1 = Dropout(0.2)(x1)

    #layer 2 (300x64)
    x2 = Conv1D(96, 2^4+1, padding='same')(x1)
    x2 = BatchNormalization(axis=1,momentum=0.99)(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=2,padding='same')(x2)
    x2 = Dropout(0.1)(x2)

    #layer 3 (150x96)
    x3 = Conv1D(128, 2^4+1, padding='same')(x2)
    x3 = BatchNormalization(axis=1,momentum=0.99)(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling1D(pool_size=2,padding='same')(x3)
    x3 = Dropout(0.1)(x3)

    #layer 4 (75x128)
    x4 = Conv1D(160, 2^4+1, padding='same')(x3)
    x4 = BatchNormalization(axis=1,momentum=0.99)(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPooling1D(pool_size=2,padding='same')(x4)
    x4 = Dropout(0.1)(x4)

    #layer 5 (38x160)
    x5 = Conv1D(192, 2^4+1, padding='same')(x4)
    x5 = BatchNormalization(axis=1,momentum=0.99)(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling1D(pool_size=2,padding='same')(x5)
    x5 = Dropout(0.1)(x5)
    
    #final shared layer (19x192)
    shared = Flatten()(x5)

    # #####################  model output staging #################### 
    s = Dense(512)(shared)
    s = Dropout(0.1)(s)
    s = Dense(5)(s)

    #####################  model output ordinal staging #################### 
    so = Dense(512)(shared)
    so = Dropout(0.1)(so)
    so = Dense(4)(so)

    #output layers
    outputS = Softmax(name='S')(s)
    So_ = OrdisticRegression(4)(so)
    outputSo = Softmax(name='So')(So_)

    model = Model(inputs=inputs, outputs=[outputS,outputSo])
    losses = {'S': wc_ct_cut_5,'So' : wc_ct_cut_4}
    metrices = {'S':'accuracy','So':'accuracy'}
    opt = tf.keras.optimizers.Adam(learning_rate=2.0e-05)
    model.compile(optimizer=opt,loss=losses,metrics=metrices,loss_weights=[1,1])


    return model





