from keras.layers import Dense,Conv2D,Multiply,Concatenate,Input,Dropout,Flatten
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

import regularizers
import params

default_loss = params.loss
default_optimizer = params.optimizer
default_shape = params.shape
default_epochs = params.epochs

def denoiser(shape=default_shape,loss=default_loss,optimizer=default_optimizer,multi_scale_dropout=0,load=False,load_name='denoiser_1'):
    image_input = Input(shape=shape,name='d_input')

    ms_1 = Conv2D(32,(1,1),padding='same',activation='relu',name='d_l1_1')(image_input)
    ms_3 = Conv2D(40,(3,3),padding='same',activation='relu',name='d_l1_3')(image_input)
    ms_5 = Conv2D(48,(5,5),padding='same',activation='relu',name='d_l1_5')(image_input)
    ms_7 = Conv2D(56,(7,7),padding='same',activation='relu',name='d_l1_7')(image_input)
    ms_9 = Conv2D(64,(9,9),padding='same',activation='relu',name='d_l1_9')(image_input)
    multi_scale_total = Concatenate(3)([ms_1,ms_3,ms_5,ms_7,ms_9])
    multi_scale_dropout = Dropout(multi_scale_dropout)(multi_scale_total)

    denoising_layer_1 = Conv2D(240,(3,3),padding='same',activation='relu',name='d_l2_1')(multi_scale_dropout)
    denoising_layer_2 = Conv2D(240,(1,1),padding='same',activation='relu',name='d_l2_2')(denoising_layer_1)
    denoising_layer_3 = Conv2D(240,(1,1),padding='same',activation='sigmoid',name='d_l2_3')(denoising_layer_2)

    hadamard = Multiply()([denoising_layer_3,multi_scale_total])

    reconstruction_layer_1 = Conv2D(64,(3,3),padding='same',kernel_regularizer = regularizers.lp(),activation='relu',name='r_l2_1')(hadamard)
    reconstruction_layer_2 = Conv2D(32,(1,1),padding='same',kernel_regularizer = regularizers.lp(),activation='relu',name='r_l2_2')(reconstruction_layer_1)
    reconstruction_layer_3 = Conv2D(1,(1,1),padding='same',activation='sigmoid',name='r_l2_3')(reconstruction_layer_2)

    output_layer = Flatten()(reconstruction_layer_3)
    denoiser = Model(input=image_input,output=output_layer)
    denoiser.compile(loss = loss, optimizer = optimizer,metrics=['accuracy'])
    if load:
        denoiser.load_weights(load_name)
    return denoiser

def train_clean_to_clean(xtrain,epochs=default_epochs,batch_size=32,save_name='denoiser_1_ctc',image_number=5000,verbose=1):
    labels = np.zeros((image_number,4096))
    for i in range(image_number):
        labels[i]=np.reshape(xtrain[i],(4096,))
    denoiser_1 = denoiser(shape=(64,64,1),multi_scale_dropout=0.7,load=False)
    history=denoiser_1.fit(xtrain,labels,epochs=30,batch_size=32,verbose=verbose)
    denoiser.save(save_name)
    return history

def train_noisy_to_clean(xtrain,ytrain,epochs=30,batch_size=32,save_name='denoiser_1_ntc',verbose=1):
    labels = np.zeros((image_number,4096))
    for i in range(image_number):
        labels[i]=np.reshape(ytrain[i],(4096,))
    denoiser_1 = denoiser(shape=(64,64,1),load=True,load_name='denoiser_1_ctc')
    history=denoiser_1.fit(xtrain,labels,epochs=30,batch_size=32,verbose=verbose)
    denoiser.save(save_name)
    return history
