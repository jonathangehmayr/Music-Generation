from time import time
import numpy as np
from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, BatchNormalization, Reshape, Flatten, Conv2DTranspose, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

from midi_conversion import pq2arr, plot_track


def sampling(args):
    '''
    Reparametrization trick
    '''
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

  
def encoder():
    '''
    Encoder architecture for the VAE
    '''
    inputs = Input(shape=(image_size, image_size, num_channels), name='encoder_input')
    
    # Block 1
    encoder = Conv2D(filters=64, kernel_size=(5,5), activation='relu', name='conv_1')(inputs)
    encoder = MaxPooling2D(pool_size=(2,2),name='pooling_1')(encoder)
    encoder = BatchNormalization(name='batch_norm_1')(encoder)
    
    # Block 2
    encoder = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv_2')(encoder)  
    encoder = MaxPooling2D(pool_size=(2,2),name='pooling_2')(encoder)
    encoder = BatchNormalization(name='batch_norm_2')(encoder)
    
    # Block 3
    encoder = Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='conv_3')(encoder)   
    encoder = MaxPooling2D(pool_size=(2,2),name='pooling_3')(encoder)
    encoder = BatchNormalization(name='batch_norm_3')(encoder)

    encoder = Flatten(name='flatten')(encoder)
    encoder = Dense(units=z_dim*2, name='dense_1')(encoder) 
    
    # Layers Representing the multivariate Gaussian
    z_mean = Dense(z_dim, name='z_mean')(encoder)
    z_log_var = Dense(z_dim, name='z_log_var')(encoder)
    
    # Sampling layer that uses the sampling function which implements the reparametrisation trick 
    z = Lambda(sampling, output_shape=(z_dim,), name='sampling_output')([z_mean, z_log_var])
    encoder_model = Model(inputs, [z,z_mean, z_log_var], name='Encoder')
    
    return encoder_model


def decoder(conv_shape):
    
    decoder_input = Input(shape=(z_dim,), name='decoder_input')
    
    decoder = Dense(units=conv_shape[1] * conv_shape[2] * conv_shape[3], name='dense_1')(decoder_input)
    decoder = Reshape(target_shape=(conv_shape[1],conv_shape[2],conv_shape[3]), name='reshape_1')(decoder)
    
    # Block 1
    decoder = UpSampling2D(size=(2,2), name='upsampling_1')(decoder)
    decoder = Conv2DTranspose(filters=64, kernel_size=(3,3), activation='relu', name='convtrans_1')(decoder)
    decoder = BatchNormalization(name='batch_norm_1')(decoder)
    
    # Block 2
    decoder = UpSampling2D(size=(2,2), name='upsampling_2')(decoder)
    decoder = Conv2DTranspose(filters=64, kernel_size=(3,3),activation='relu',name='convtrans_2')(decoder)
    decoder = BatchNormalization(name='batch_norm_2')(decoder)
    
    # Block 3
    decoder = UpSampling2D(size=(2,2), name='upsampling_3')(decoder)
    decoder_output = Conv2DTranspose(filters=1, kernel_size=(5,5), activation='relu',name='convtrans_4')(decoder)
    decoder_model = Model(decoder_input, decoder_output, name='Decoder')
    
    return decoder_model


def mse_loss(y_true, y_pred):
    '''
    Implementation of mean squared error
    '''
    r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return r_loss*image_size*image_size

def kl_loss(mean, log_var):
    '''
    Calculation of Kullback-Leibler divergence
    '''
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    return kl_loss

def elbo(y_true, y_pred, mean, log_var, omega=1):
    '''
    Elbo loss by summing up MSE and KL. KL can be additionally weighted by omega
    '''
    r_loss = mse_loss(y_true, y_pred)
    kl_div = kl_loss(mean, log_var)
    return  r_loss + kl_div*omega



class VAE(Model):
    '''
    Variational Autoencoder class adapted from:
    https://www.geeksforgeeks.org/variational-autoencoders/  
    '''
    def __init__(self, encoder, decoder, omega=1,**kwargs):#encoder, decoder,
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.omega=omega 
        
        self.elbo_loss_tracker = tf.keras.metrics.Mean(name='elbo_loss')
        self.r_loss_tracker = tf.keras.metrics.Mean(name='r_loss')
        self.kl_div_tracker = tf.keras.metrics.Mean(name='kl_div') 

    @property
    def metrics(self):
        return [
            self.elbo_loss_tracker,
            self.r_loss_tracker,
            self.kl_div_tracker,
        ]
    
    def call(self, inputs):
        '''
        By calling the VAE an input array is encoded and decoded
        '''
        z,z_mean, z_log_var, = self.encoder(inputs)    
        return self.decoder(z) 
    
    def summary(self):
        '''
        Custom model.summary function to display the architecture of the VAE
        '''
        self.encoder.summary()
        self.decoder.summary()
          
    def train_step(self, data):
        '''
        Training VAEs requires a custom train step function in order to have
        access to the learnt distribution paramters and update them 
        '''
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
                      
            z,z_mean, z_log_var, = self.encoder(data)    
            y_pred = self.decoder(z) 
                   
            r_loss = mse_loss(data, y_pred)
            kl_div = kl_loss(z_mean, z_log_var)
            elbo_loss = r_loss + kl_div*self.omega
            
            
        grads = tape.gradient(elbo_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
         
        self.elbo_loss_tracker.update_state(elbo_loss)
        self.r_loss_tracker.update_state(r_loss)
        self.kl_div_tracker.update_state(kl_div)    
        
        return {
            'elbo_loss': self.elbo_loss_tracker.result(),
            'reconstruction_loss': self.r_loss_tracker.result(),
            'kl_loss': self.kl_div_tracker.result()}
    
    
    def predict(self, data):
        '''
        Encodes and decodes a given array
        '''
        z,z_mean, z_log_var, = self.encoder(data)    
        return self.decoder(z).numpy().reshape(image_size, image_size)
    
    def generate(self):
        '''
        By calling generate() a random vector is sampled from a Gaussian
        and fed into the decoder part. This outputs a newly composed array
        '''
        z_sample = np.random.random_sample((z_dim)).round(3)
        z_sample = np.array([list(z_sample)])
        return self.decoder.predict(z_sample).reshape(image_size, image_size)
      

def experiment_512():
    '''
    In this function a VAE is trained on the dataset with individual
    arrays of size (512,512)
    '''
    arr_tracks, arr_track_lengths=pq2arr(512,(512,512))
    arr_tracks=np.reshape(arr_tracks,(len(arr_tracks), image_size, image_size, num_channels))
 
    n_epochs=[30, 50, 100, 200]
    for n_epoch in n_epochs:
        
        enc=encoder()
        conv_shape=enc.get_layer('pooling_3').output.shape
        dec=decoder(conv_shape)   
        vae = VAE(enc, dec, omega=1)
        vae.compile(optimizer='adam')
        
        #batch sizes do not always work https://github.com/davidADSP/GDL_code/issues/63
        _start=time()
        vae.fit(arr_tracks, epochs=n_epoch, batch_size=n_batch)
        _stop=time()
        print(_stop-_start)
        vae.save_weights('vae_weights_512_{}_epochs'.format(n_epoch))

    
def experiment_256():
    '''
    In this function a VAE is trained on the dataset with individual
    arrays of size (256,256)
    '''
    arr_tracks, arr_track_lengths=pq2arr(256,(256,256))
    arr_tracks=np.reshape(arr_tracks,(len(arr_tracks), image_size, image_size, num_channels))

    n_epochs=[30, 50, 100, 200]
    for n_epoch in n_epochs:
        
        enc=encoder()
        conv_shape=enc.get_layer('pooling_3').output.shape
        dec=decoder(conv_shape)
        vae = VAE(enc, dec, omega=1)
        vae.compile(optimizer='adam')
        
        #batch sizes do not always work https://github.com/davidADSP/GDL_code/issues/63
        _start=time()
        vae.fit(arr_tracks, epochs=n_epoch, batch_size=n_batch)
        _stop=time()
        print(_stop-_start)
        vae.save_weights('vae_weights_256_{}_epochs'.format(n_epoch))
        

def load_vae(filename=''):
    '''
    load_vae instantiates a VAE and loads saved weights from previous trainings
    '''
    enc=encoder()
    conv_shape=enc.get_layer('pooling_3').output.shape
    dec=decoder(conv_shape)
    vae = VAE(enc, dec, omega=1)
    vae.compile(optimizer='adam')
    vae.load_weights(filename)
    return vae


if __name__=='__main__':
    
    # define some global variables configurations
    num_channels = 1
    image_size=512 #256
    z_dim=int(image_size/2)
    n_batch=59 # divides 295
    #n_epoch=100
    
    # Loading of a track to test the VAE model
    arr_tracks, arr_track_lengths=pq2arr(image_size,(image_size,image_size))
    arr_tracks=np.reshape(arr_tracks,(len(arr_tracks), image_size, image_size, num_channels))
    arr_track=arr_tracks[0,:,:,:].reshape(1,image_size, image_size)
    
    
    # Loading of the weight of a trained VAE
    vae=load_vae(filename='models/vae_weights_512_200_epochs')
    vae.summary()
    
    # Plotting of generated music
    track=vae.generate()
    plot_track(track, filename='vae_pred')

