#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from tensorflow.keras.preprocessing.image import load_img ,img_to_array
from tensorflow.keras import *
from tensorflow.keras.regularizers import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices()), tf.config.experimental.list_physical_devices())
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.85
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



# Regression network model as in given in the paper - Image Colorization with Deep Convolutional Neural Networks

def reg_network_schematic():
    inputs = tf.keras.Input(shape=(64,64,1))
    c1=Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu",name='0')(inputs)
    c2=Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu",name='1')(c1)
    b1=BatchNormalization()(c2)
    m1=MaxPool2D(pool_size=(2,2),strides=(2,2),name='2')(b1)

    c3=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name='3')(b1)
    c4=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name='4')(c3)
    b2=BatchNormalization()(c4)
    m2=MaxPool2D(pool_size=(2,2),strides=(2,2),name='5')(b2)
    
    c5=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='6')(m2)
    c6=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='7')(c5)
    c7=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='8')(c6)
    b3=BatchNormalization()(c7)
    m3=MaxPool2D(pool_size=(2,2),strides=(2,2),name='9')(m3)
    
    c8=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='10')(b3)
    c9=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='11')(c8)
    c10=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='12')(c9)
    
    ex1=Conv2DTranspose(filters=256, kernel_size=(1,1),strides=(2,2), padding="same", activation="relu",name='13')(c10)
    a1=Add(name='14')([c7, ex1])
    
    ex2=Conv2DTranspose(filters=128, kernel_size=(3,3),strides=(2,2), padding="same", activation="relu",name='15')(a1)
    a2=Add(name='16')([ex2,c4])
    
    ex3=Conv2DTranspose(filters=64, kernel_size=(3,3),strides=(2,2), padding="same", activation="relu",name='17')(a2)
    a3=Add(name='18')([ex3,c2])
    
    outputs=Conv2D(filters=2, kernel_size=(3,3), padding="same", activation="relu",name='19')(a3)
    model=Model(inputs=inputs,outputs=outputs)
    return model



# A simple UNet model as proposed in U-Net: Convolutional Networks for Biomedical Image Segmentation

def UNet():
    inputs = Input([64, 64, 1])
    c0 = Conv2D(64, (3, 3), padding='same')(inputs)
    b0 = BatchNormalization()(c0)
    a0 = LeakyReLU(alpha=0.2)(b0)    
    c1 = Conv2D(64, (3, 3), strides=1, padding='same')(a0)
    b1 = BatchNormalization()(c1)
    a1 = LeakyReLU(alpha=0.2)(b1)    
    
    m1 = MaxPool2D((2, 2), strides=2)(a1)    
    c2 = Conv2D(128, (3, 3), padding='same')(m1)
    b2 = BatchNormalization()(c2)
    a2 = LeakyReLU(alpha=0.2)(b2)
    c3 = Conv2D(128, (3, 3), padding='same')(a2)
    b3 = BatchNormalization()(c3)
    a3 = LeakyReLU(alpha=0.2)(b3)    
    
    m2 = MaxPool2D((2, 2), strides=2)(a3)    
    c4 = Conv2D(256, (3, 3), padding='same')(m2)
    b4 = BatchNormalization()(c4)
    a4 = LeakyReLU(alpha=0.2)(b4)
    c5 = Conv2D(256, (3, 3), padding='same')(a4)
    b5 = BatchNormalization()(c5)
    a5 = LeakyReLU(alpha=0.2)(b5)    
    
    m3 = MaxPool2D((2, 2), strides=2)(a5)    
    c6 = Conv2D(512, (3, 3), padding='same')(m3)
    b6 = BatchNormalization()(c6)
    a6 = LeakyReLU(alpha=0.2)(b6)
    c7 = Conv2D(512, (3, 3), padding='same')(a6)
    b7 = BatchNormalization()(c7)
    a7 = LeakyReLU(alpha=0.2)(b7)    
    
    m4 = MaxPool2D((2, 2), strides=2)(a7)    
    c8 = Conv2D(1024, (3, 3), padding='same')(m4)
    b8 = BatchNormalization()(c8)
    a8 = LeakyReLU(alpha=0.2)(b8)
    c9 = Conv2D(1024, (3, 3), padding='same')(a8)
    b9 = BatchNormalization()(c9)
    a9 = LeakyReLU(alpha=0.2)(b9)    
    
    t0 = Conv2DTranspose(512, (2, 2), strides=2)(a9)    
    cc0 = Concatenate()([a7, t0])     
    c10 = Conv2D(512, (3, 3), padding='same')(cc0)
    b10 = BatchNormalization()(c10)
    a10 = Activation('relu')(b10)
    c11 = Conv2D(512, (3, 3), padding='same')(a10)
    b11 = BatchNormalization()(c11)
    a11 = Activation('relu')(b11)    
    
    t1 = Conv2DTranspose(256, (2, 2), strides=2)(a11)    
    cc1 = Concatenate()([a5, t1])     
    c12 = Conv2D(256, (3, 3), padding='same')(cc1)
    b12 = BatchNormalization()(c12)
    a12 = Activation('relu')(b12)
    c13 = Conv2D(256, (3, 3), padding='same')(a12)
    b13 = BatchNormalization()(c13)
    a13 = Activation('relu')(b13)    
    
    t2 = Conv2DTranspose(128, (2, 2), strides=2)(a13)    
    cc2 = Concatenate()([a3, t2])     
    c14 = Conv2D(128, (3, 3), padding='same')(cc2)
    b14 = BatchNormalization()(c14)
    a14 = Activation('relu')(b14)
    c15 = Conv2D(128, (3, 3), padding='same')(a14)
    b15 = BatchNormalization()(c15)
    a15 = Activation('relu')(b15)   
    
    t3 = Conv2DTranspose(64, (2, 2), strides=2)(a15)    
    cc3 = Concatenate()([a1, t3])    
    c16 = Conv2D(64, (3, 3), padding='same')(cc3)
    b16 = BatchNormalization()(c16)
    a16 = Activation('relu')(b16)
    c17 = Conv2D(64, (3, 3), padding='same')(a16)
    b17 = BatchNormalization()(c17)
    a17 = Activation('relu')(b17)    
    
    outputs = Conv2D(2, (1, 1), strides=1)(a17) 
    model = Model(inputs=inputs, outputs=outputs)
    return model

