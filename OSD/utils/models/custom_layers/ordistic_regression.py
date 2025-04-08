
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
import tensorflow as tf

class OrdisticRegression(Layer):
    """
    Rennie, J.D. and Srebro, N., 2005, July.
    Loss functions for preference levels: Regression with discrete ordered labels.
    In Proceedings of the IJCAI multidisciplinary workshop on advances in preference handling (pp. 180-186)
    """
    def __init__(self, nlevel, **kwargs):
        self.nlevel = nlevel
        super(OrdisticRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        # the linear dense layer
        self.w = self.add_weight(name='w', shape=(input_shape[1], 1),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(),
                                initializer='zero', trainable=True)
        
        # mu1 = -1, mu2 = tanh(itanh_mu2), mu3 = tanh(itanh_mu2+exp(log_diff_itanh_mus3)), ..., muK = 1
        init_mus = np.linspace(-1,1,self.nlevel)
        self.mu1 = self.add_weight(name='mu1', shape=(1,),
                                initializer=Constant(value=[init_mus[0]]), trainable=False)
        self.muK = self.add_weight(name='muK', shape=(1,),
                                initializer=Constant(value=[init_mus[-1]]), trainable=False)
        self.itanh_mu2 = self.add_weight(name='itanh_mu2', shape=(1,),
                                initializer=Constant(value=[np.arctanh(init_mus[1])]), trainable=True) # itanh: inverse tanh
        if self.nlevel>=4:
            self.log_diff_itanh_mus = self.add_weight(name='log_diff_itanh_mus', shape=(self.nlevel-3,),
                                initializer=Constant(value=np.log(np.diff(np.arctanh(init_mus[1:-1])))), trainable=True)
        
        super(OrdisticRegression, self).build(input_shape)  # Be sure to call this at the end
        
    def get_mus(self):
        mus = [self.mu1, K.tanh(self.itanh_mu2)]
        if self.nlevel>=4:
            mus.append(K.tanh(self.itanh_mu2 + K.cumsum(K.exp(self.log_diff_itanh_mus))))
        mus.append(self.muK)
            
        return K.concatenate(mus)
        
    def get_mus_array(self, itanh_mu2, log_diff_itanh_mus=None):
        mus = [-1, np.tanh(itanh_mu2)]
        if self.nlevel>=4:
            mus.append(np.tanh(itanh_mu2 + np.cumsum(np.exp(log_diff_itanh_mus))))
        mus.append(1.)
            
        return np.hstack(mus).astype(float)

    def call(self, x):
        mus = K.reshape(self.get_mus(), (1,self.nlevel))
        z = K.dot(x, self.w)+self.b
        
        # changed squared error to absolute error
        # because some how square error gives unstable result
        #pi = mus*z-mus**2*0.5
        pi = -K.abs(mus-z)
        
        # do not use softmax here
        # because loss=categorical_crossentropy_with_logit
        # to save some computation and get numeric stability
        #pi = K.softmax(pi, axis=1)
        return pi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nlevel)
    
    def get_config(self):
        config = {'nlevel': self.nlevel}
        base_config = super(OrdisticRegression, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def categorical_crossentropy_with_logit(y_true, y_pred):

    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

# #crossentropy cut
# def ct_cut(y_true,y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     y_pred_f= tf.cast(tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7)),tf.float32)
#     mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
#     out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
#     out=K.mean(out)
#     return out

# #weighted crossentropy cut
# def wc_ct_cut(y_true,y_pred):
#     y_true = tf.cast(y_true,tf.float32)
#     y_pred = tf.cast(y_pred,tf.float32)

#     w0=2
#     w1=1
#     w2=2
#     w3=0
#     w4=2 
#     l0=categorical_crossentropy_with_logit(y_true[:,0],y_pred[:,0]) # n3
#     l1=categorical_crossentropy_with_logit(y_true[:,1],y_pred[:,1]) # n2
#     l2=categorical_crossentropy_with_logit(y_true[:,2],y_pred[:,2]) # n1
#     l3=categorical_crossentropy_with_logit(y_true[:,3],y_pred[:,3]) # r
#     l4=categorical_crossentropy_with_logit(y_true[:,4],y_pred[:,4]) # wake

#     out = (w0 * l0 + w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4)/(w0+w1+w2+w3+w4)  # set custom weights for each class
#     len_y = tf.cast(len(y_true),tf.float32)
#     out = out/len_y
#     return out

# #weighted crossentropy cut
# def wc_ct_cut_so(y_true,y_pred):
#     y_true = tf.cast(y_true,tf.float32)
#     y_pred = tf.cast(y_pred,tf.float32)

#     w0=2
#     w1=1
#     w2=2
#     w3=2

#     l0=categorical_crossentropy_with_logit(y_true[:,0],y_pred[:,0]) # n3
#     l1=categorical_crossentropy_with_logit(y_true[:,1],y_pred[:,1]) # n2
#     l2=categorical_crossentropy_with_logit(y_true[:,2],y_pred[:,2]) # n1
#     l3=categorical_crossentropy_with_logit(y_true[:,3],y_pred[:,3]) # W


#     out = (w0 * l0 + w1 * l1 + w2 * l2 + w3 * l3 )/(w0+w1+w2+w3)  # set custom weights for each class
#     len_y = tf.cast(len(y_true),tf.float32)
#     out = out/len_y
#     return out


if __name__ == '__main__':
    
    model = load_model('/media/cdac-user/hdd/exxactpcbackup/datasets/Sleep_ORP_ERIK/OSD_model.h5', custom_objects={'OrdisticRegression': OrdisticRegression, 'categorical_crossentropy_with_logit' : categorical_crossentropy_with_logit})
    # load weights into new model
    model.load_weights('/media/cdac-user/hdd/exxactpcbackup/datasets/Sleep_ORP_ERIK/Model.h5')
    print(model.summary())

    [pred1, pred2] = model.predict(DATA,verbose=0)

    
