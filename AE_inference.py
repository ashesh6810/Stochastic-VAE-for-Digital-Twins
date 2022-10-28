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


latent_dim=128

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




input_data = Input(shape=(192, 96, 2))
#cond = tensorflow.keras.layers.Input(batch_shape=(192,96,2))

#input_data = keras.layers.Concatenate(axis=-1)([input_data, cond])


encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable=False)(input_data)
encoder = MaxPooling2D((2,2))(encoder)

encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable=False)(encoder)
encoder = MaxPooling2D((2,2))(encoder)

encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable=False)(encoder)
encoder = MaxPooling2D((2,2))(encoder)


encoder = Conv2D(64, (5,5), activation='relu',padding='same',trainable=False)(encoder)
encoder = MaxPooling2D((2,2))(encoder)

encoder = Flatten()(encoder)
#encoder = Dense(16)(encoder)



encoder_model = Model(input_data,encoder)

encoder_model.summary()

decoder_input = Input(shape=(4608,))
decoder = Dense(4608,trainable=False)(decoder_input)
decoder = Reshape((12, 6, 64 ))(decoder)
decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable=False)(decoder)

decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable=False)(decoder)
decoder = UpSampling2D((2,2))(decoder)

decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable=False)(decoder)
decoder = UpSampling2D((2,2))(decoder)

decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable=False)(decoder)
decoder = UpSampling2D((2,2))(decoder)

decoder = Conv2DTranspose(64, (3,3), activation='relu',padding='same',trainable=False)(decoder)
decoder = UpSampling2D((2,2))(decoder)

decoder_output = Conv2DTranspose(2, (5,5), activation='linear',padding='same',trainable=False)(decoder)
decoder_model = Model(decoder_input, decoder_output)
decoder_model.summary()


encoded = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = Model(input_data, decoded)



def get_loss(a):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch
    

    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        
        return reconstruction_loss_batch     
    return total_loss


autoencoder.compile(loss=get_loss(0), optimizer='adam')
autoencoder.summary()
autoencoder.load_weights('best_weights_AE_lead1.h5')

(encoder_model.get_layer('conv2d_3')).trainable=True
(encoder_model.get_layer('conv2d_4')).trainable=True

(decoder_model.get_layer('conv2d_transpose_6')).trainable=True
(decoder_model.get_layer('conv2d_transpose_5')).trainable=True
(decoder_model.get_layer('conv2d_transpose_4')).trainable=True



encoded = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = Model(input_data, decoded)


autoencoder.compile(loss=get_loss(0), optimizer='adam')
autoencoder.summary()

count=0

for loop in fileList_train:
 print('Training loop index',loop)
 File=nc.Dataset(loop)
# F=nc.Dataset('PSI_output.nc')
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
                       keras.callbacks.ModelCheckpoint('best_weights_AE_transfer_lead1.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )

 else:
  autoencoder.load_weights('best_weights_AE_transfer_lead1.h5')
  hist = autoencoder.fit(psi_input_Tr, psi_label_Tr,
                       batch_size = 100,
             verbose=1,
             epochs = 200,
             validation_data=(psi_test_input_Tr, psi_test_label_Tr),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights_AE_transfer_lead1.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )


#  autoencoder.fit(psi_input_Tr, psi_label_Tr, epochs=300, batch_size=64, validation_data=(psi_test_input_Tr, psi_test_label_Tr),verbose=1,keras.callbacks.ModelCheckpoint('best_weights_lead1.h5', monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1),callbacks=[history])
 count=count+1

#print('training loss',history.history['loss'])
#print('testing loss',history.history['val_loss'])

autoencoder.load_weights('best_weights_AE_transfer_lead1.h5')
print('starting inference')

### Sample from std normal ################

#ens=100
#z=np.zeros([36,ens])
#for k in range (0,ens):
  
#  z[:,k] = np.random.normal(0,1,[36,])



#### start generative prediction ##########

generator_model = decoder_model

test_time=5000
pred_mean = np.zeros([test_time, 192, 96, 2])
#pred_ens = np.zeros([ens, 192, 96, 2])
sig_m=0.10
initial_point = psi_test_input_Tr[0,:,:,:].flatten()+np.random.normal(0,sig_m,[2*192*96,])

for k in range(0,test_time):
 print('inference time',k)
 

 init_cond = encoder_model.predict(initial_point.reshape([1,192,96,2]))
 u = init_cond.reshape([1,4608])

 pred_mean[k, :,:,:] = generator_model.predict(u)

 initial_point = pred_mean[k,:,:,:].reshape([1,192,96,2])



savenc(pred_mean, y, x, 'pred_mean_imperfect_trained_noisy_AE.nc')
#savenc(psi_test_label_Tr, y, x, 'pred_truth.nc')












