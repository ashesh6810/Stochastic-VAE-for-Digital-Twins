from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import netCDF4 as nc
from saveNCfile import savenc

from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
from keras.callbacks import History 
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D
import tensorflow


latent_dim=512

### Load filenames for training #####
fileList_train=[]
mylist = [1]
for k in mylist:
  fileList_train.append ('/oasis/scratch/comet/a6810/temp_project/VAE/imperfect_simulations/set'+str(k)+'_'+'test'+'/PSI_output.nc')




F=nc.Dataset('/oasis/scratch/comet/a6810/temp_project/VAE/imperfect_simulations/set3_test/PSI_output.nc')
psi=F['PSI']
psi=psi[2500:,:,:,:]
y=np.asarray(F['lat'])
x=np.asarray(F['lon'])

Nlat=np.size(psi,2);
Nlon=np.size(psi,3);

print(Nlat)
print(Nlon)


lead=1


psi_test_input = psi[0:np.size(psi,0)-lead,:,:,:]
psi_test_label = psi[0+lead:np.size(psi,0),:,:,:]



psi_test_input_Tr=np.zeros([np.size(psi,0),Nlat,Nlon,2])
psi_test_label_Tr=np.zeros([np.size(psi,0),Nlat,Nlon,2])



for k in range(0,np.size(psi_test_input,0)):
    psi_test_input_Tr[k,:,:,0] = psi_test_input[k,0,:,:]
    psi_test_input_Tr[k,:,:,1] = psi_test_input[k,1,:,:]
    psi_test_label_Tr[k,:,:,0] = psi_test_label[k,0,:,:]
    psi_test_label_Tr[k,:,:,1] = psi_test_label[k,1,:,:] 




print('Test input', np.shape(psi_test_input_Tr))
print('Test label', np.shape(psi_test_label_Tr))




def create_model(frozen):

 input_data = Input(shape=(192, 96, 2))
#cond = tensorflow.keras.layers.Input(batch_shape=(192,96,2))

#input_data = keras.layers.Concatenate(axis=-1)([input_data, cond])

 if frozen == 1:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(input_data)
 else:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(input_data)

 encoder = MaxPooling2D((2,2))(encoder)



 if frozen == 1:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(encoder)
 else:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(encoder)


 encoder = MaxPooling2D((2,2))(encoder)

 if frozen == 1:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(encoder)
 
 else:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = True)(encoder)

 encoder = MaxPooling2D((2,2))(encoder)


 if frozen == 1:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = False)(encoder)
 else:
  encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable = True)(encoder)
 
  
 encoder = MaxPooling2D((2,2))(encoder)

 encoder = Flatten()(encoder)
#encoder = Dense(16)(encoder)



 def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tensorflow.shape(distribution_variance)[0]
    random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
    return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

 if frozen == 1:
  distribution_mean = Dense(latent_dim, name='mean',trainable=False)(encoder)
  distribution_variance = Dense(latent_dim, name='log_variance',trainable=False)(encoder)
 else:
  distribution_mean = Dense(latent_dim, name='mean',trainable=True)(encoder)
  distribution_variance = Dense(latent_dim, name='log_variance',trainable=True)(encoder)

 latent_encoding = Lambda(sample_latent_features)([distribution_mean, distribution_variance])

 cond = encoder
 latent_encoding_conditioned = concatenate([latent_encoding, cond], axis = 1)

 encoder_model = Model(input_data,[latent_encoding_conditioned,encoder,distribution_mean,distribution_variance])

 encoder_model.summary()

 decoder_input = Input(shape=(4608+latent_dim,))
 if frozen == 1:
  decoder = Dense(4608,trainable = False)(decoder_input)
 else: 
  decoder = Dense(4608,trainable = False)(decoder_input)

 decoder = Reshape((12, 6, 64 ))(decoder)
 if frozen == 1:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)
 else:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)

 if frozen == 1:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)
 else:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)

 decoder = UpSampling2D((2,2))(decoder)

 if frozen == 1:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)

 else:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)


 decoder = UpSampling2D((2,2))(decoder)

 if frozen == 1:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)
 else:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)

 decoder = UpSampling2D((2,2))(decoder)

 if frozen == 1:
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = False)(decoder)
 else :
  decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable = True)(decoder)


 decoder = UpSampling2D((2,2))(decoder)

 if frozen == 1:
  decoder_output = Conv2DTranspose(2, (5,5), activation='linear',padding='same',trainable = False)(decoder)
 else:
  decoder_output = Conv2DTranspose(2, (5,5), activation='linear',padding='same',trainable = True)(decoder)


 decoder_model = Model(decoder_input, decoder_output)
 decoder_model.summary()


 encoded, dum1, dum2, dum3 = encoder_model(input_data)
 decoded = decoder_model(encoded)
 autoencoder = Model(input_data, decoded)
 return encoder_model, decoder_model, autoencoder,distribution_mean,distribution_variance


def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(distribution_variance)
        kl_loss_batch = tensorflow.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch
    
    return total_loss

encoder_model, decoder_model, autoencoder,distribution_mean,distribution_variance  = create_model(1)
autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
autoencoder.summary()
autoencoder.load_weights('best_weights_lead1.h5')

(encoder_model.get_layer('log_variance')).trainable=True
(encoder_model.get_layer('mean')).trainable=True
(encoder_model.get_layer('conv2d_3')).trainable=True
(encoder_model.get_layer('conv2d_4')).trainable=True

(decoder_model.get_layer('conv2d_transpose_6')).trainable=True
(decoder_model.get_layer('conv2d_transpose_5')).trainable=True
(decoder_model.get_layer('conv2d_transpose_4')).trainable=True

encoded, dum1, dum2, dum3 = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = Model(input_data, decoded)

autoencoder.summary()


#autoencoder,distribution_mean,distribution_variance  = create_model(0)
#autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
#autoencoder.summary()
#autoencoder.load_weights('best_weights_lead1.h5')





count=0

for loop in fileList_train:
 print('Training loop index',loop)
 File=nc.Dataset(loop)
 
 psi=File['PSI']
 psi=psi[2500:,:,:,:]
 
 trainN=7000
 lead=1

 psi_input = psi[0:trainN,:,:,:]
 psi_label = psi[0+lead:trainN+lead,:,:,:]

 psi_input_Tr=np.zeros([trainN,Nlat,Nlon,2])
 psi_label_Tr=np.zeros([trainN,Nlat,Nlon,2])

 for k in range(0,trainN):
  psi_input_Tr[k,:,:,0] = psi_input[k,0,:,:]
  psi_input_Tr[k,:,:,1] = psi_input[k,1,:,:]
  psi_label_Tr[k,:,:,0] = psi_label[k,0,:,:]
  psi_label_Tr[k,:,:,1] = psi_label[k,1,:,:]

 print('Train input', np.shape(psi_input_Tr))
 print('Train label', np.shape(psi_label_Tr))


 if (count==0): 
  autoencoder.load_weights('best_weights_lead1.h5')
#  autoencoder.fit(psi_input_Tr, psi_label_Tr, epochs=300, batch_size=64, validation_data=(psi_test_input_Tr, psi_test_label_Tr),verbose=1,keras.callbacks.ModelCheckpoint('best_weights_lead1.h5', monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1),callbacks=[history])
  hist = autoencoder.fit(psi_input_Tr, psi_label_Tr,
                       batch_size = 100,
             verbose=1,
             epochs = 200,
             validation_data=(psi_test_input_Tr, psi_test_label_Tr),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights_transfer_learning_lead1.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )

 else:
  autoencoder.load_weights('best_weights_transfer_learning_lead1.h5')
  hist = autoencoder.fit(psi_input_Tr, psi_label_Tr,
                       batch_size = 100,
             verbose=1,
             epochs = 200,
             validation_data=(psi_test_input_Tr, psi_test_label_Tr),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights_transfer_learning_lead1.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )


#  autoencoder.fit(psi_input_Tr, psi_label_Tr, epochs=300, batch_size=64, validation_data=(psi_test_input_Tr, psi_test_label_Tr),verbose=1,keras.callbacks.ModelCheckpoint('best_weights_lead1.h5', monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1),callbacks=[history])
 count=count+1

#print('training loss',history.history['loss'])
#print('testing loss',history.history['val_loss'])
autoencoder.load_weights('best_weights_transfer_learning_lead1.h5')
print('starting inference')

### Sample from std normal ################

#ens=100
#z=np.zeros([36,ens])
#for k in range (0,ens):
  
#  z[:,k] = np.random.normal(0,1,[36,])



#### start generative prediction ##########
ens=100
sig_m=0.10
#init_cond = (psi_test_input_Tr[k,:,:,:]).flatten() + np.random.normal(0,sig_m,[2*192*96,])
generator_model = decoder_model

test_time=5000
pred_mean = np.zeros([test_time, 192, 96, 2])
pred_ens = np.zeros([ens, 192, 96, 2])

initial_point = psi_test_input_Tr[0,:,:,:].flatten() + np.random.normal(0,sig_m,[2*192*96,])

for k in range(0,test_time):
 print('inference time',k)
 for N in range(0,ens):

    dummy, init_cond, Mean, var = encoder_model.predict(initial_point.reshape([1,192,96,2]))
    if (k==0 and N==0):
       print('shape of mean extracted', np.shape(Mean))
       print('shape of var extracted',np.shape(var))
    random = np.random.multivariate_normal(np.zeros(latent_dim),np.eye(latent_dim))
    z = Mean + np.exp(0.5 * var) * random
    u=np.concatenate((z.reshape([1,latent_dim]),init_cond.reshape([1,4608])), axis=1)

    pred_ens[N, :,:,:] = generator_model.predict(u)

 pred_mean[k,:,:,:] = np.mean(pred_ens,0)
 initial_point = pred_mean[k,:,:,:].reshape([1,192,96,2])

# savenc(pred_mean, y, x, 'pred_ens'+str(N)+'.nc')
 savenc(pred_ens, y, x, 'pred_ens_imperfect_Trained_noisyIC_time'+str(k)+'.nc')

savenc(pred_mean, y, x, 'pred_mean_imperfect_trained_noisyIC_ens'+str(ens)+'.nc')
#savenc(psi_test_label_Tr, y, x, 'pred_truth.nc')












