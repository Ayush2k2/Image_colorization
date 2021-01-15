#!/usr/bin/env python
# coding: utf-8


from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.layers import *
import cv2
import numpy as np
import PIL
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import imageio
from PIL import Image,ImageOps
from scipy import ndimage
from scipy.io import loadmat
from tensorflow.keras import backend as K
from sys import argv
import imageio
import os
from skimage.io import imread
from skimage.transform import resize

if len(argv) != 3:
    print("usage:./Image_colorizer.py model filename")
    print("model=0 for regression and model=1 for unet")
    exit(4)
    

class colorize_reg():
    def __init__(self,path):
        self.path=path
        name=path.split('/')
        self.name=name[-1]
        
    def load_model(self):
        json_file = open('reg_imcolor/reg_imcolor.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_reg = model_from_json(loaded_model_json)
        # load weights into new model
        model_reg.load_weights("reg_imcolor/reg_imcolor.h5")
        
        return model_reg
    
    def colorize(self):
        from PIL import Image
        # Open the image form working directory
        test_0 = Image.open(self.path)
        test_0=np.array(test_0)
        shape=test_0.shape
        test_0 = resize(test_0, (224, 224)).astype('float32')
        test_0 = cv2.cvtColor(test_0,cv2.COLOR_RGB2Luv)
        test_0l=test_0[:,:,0]
        test=test_0l.reshape(1,224,224,1)
        # load model
        model_reg=self.load_model()
        pred=model_reg.predict(test)
        pred_im=np.dstack([test[0],pred[0]])
        pred_im=pred_im.astype('float32')
        rgbpred = cv2.cvtColor(pred_im,cv2.COLOR_Luv2RGB)
        rgbpred = resize(rgbpred, shape).astype('float32') 
        return rgbpred
        
    def run(self):
        im = self.colorize()
        im = Image.fromarray((im*255).astype(np.uint8))
        im.show(im)
        im.save('colorized_reg_'+self.name )
        K.clear_session()
        pass
        
class colorize_unet():
    def __init__(self,path):
        self.path=path
        name=path.split('/')
        self.name=name[-1]
        
    def load_model(self):
        json_file = open('unet_imcolor/unet_imcolorf.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_reg = model_from_json(loaded_model_json)
        # load weights into new model
        model_reg.load_weights("unet_imcolor/unet_imcolorf.h5")
        
        return model_reg
    
    def colorize(self):
        from PIL import Image
        # Open the image form working directory
        test_0 = Image.open(self.path)
        test_0=np.array(test_0)
        shape=test_0.shape
        test_0 = resize(test_0, (512, 512)).astype('float32')
        test_0 = cv2.cvtColor(test_0,cv2.COLOR_RGB2Luv)
        test_0l=test_0[:,:,0]
        test=test_0l.reshape(1,512,512,1)
        
        # load model
        model_reg=self.load_model()
        pred=model_reg.predict(test)
        pred_im=np.dstack([test[0],pred[0]])
        pred_im=pred_im.astype('float32')
        rgbpred = cv2.cvtColor(pred_im,cv2.COLOR_Luv2RGB)
        rgbpred = resize(rgbpred, shape).astype('float32')
        plt.figure(figsize=(12,12))
        plt.imshow(rgbpred) 
        return rgbpred
        
    def run(self):
        im = self.colorize()
        im = Image.fromarray((im*255).astype(np.uint8))
        im.show(im)
        im.save('colorized_unet_'+self.name ) 
        K.clear_session()
        pass
        
def predict(model_num,filename):
    if int(model_num)==0:
        colorize_reg(filename).run()
    elif int(model_num)==1:
        colorize_unet(filename).run()
    K.clear_session()
    pass

predict(argv[1],argv[2])