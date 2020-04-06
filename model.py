import keras
import os
import cv2
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, AveragePooling2D, Activation, ZeroPadding2D,Flatten,Input,Add,concatenate
from keras.models import Sequential,Model
from keras.utils import plot_model
from utils import *
def VGG16(input_shape=(224,224,3),n_classes=10):
    '''
        Hàm khởi tạo model VGG16.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model VGG16 với softmax 
    '''
    In=Input(shape=input_shape)
    #block 1 starts here
    x=Conv2D(64,(3,3),padding='same')(In)
    x=Activation('relu')(x)
    x=Conv2D(64,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 2 starts here
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 3 starts here
    x=Conv2D(256,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(256,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 3 starts here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 4 starts here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #Flatten and fully connected layer start here
    x=Flatten()(x)
    x=Dense(units=25088,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=n_classes,activation='softmax')(x)
    model=Model(In,x)
    plot_model(model,to_file='./Images/VGG16.png',show_shapes=True,show_layer_names=False)
    return model 

def VGG19(input_shape=(224,224,3),n_classes=10):
    '''
        Hàm khởi tạo model VGG19.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model VGG19 với softmax 
    '''
    In=Input(shape=input_shape)
    #block 1 starts here
    x=Conv2D(64,(3,3),padding='same')(In)
    x=Activation('relu')(x)
    x=Conv2D(64,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 2 starts here
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(128,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 3 starts here
    x=Conv2D(256,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(256,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 3 starts here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #block 4 starts here
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(512,(3,3),padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #Flatten and fully connected layer start here
    x=Flatten()(x)
    x=Dense(units=25088,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=4096,activation='relu')(x)
    x=Dense(units=n_classes,activation='softmax')(x)
    model=Model(In,x)
    plot_model(model,to_file='./Images/VGG19.png',show_shapes=True,show_layer_names=False)
    return model 

def InceptionNet(input_shape=(224,224,3),n_classes=10):
    '''
        Hàm khởi tạo model inception net v1.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model inception với softmax
    '''
    
    In=Input(shape=input_shape)
    
    x=Conv2D(64,(7,7),strides=2,padding='same')(In)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2, padding='same')(x)
    

    x=Conv2D(192,(3,3),strides=1,padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)

    x=inception_block(x,256) #inception module 3a
    x=inception_block(x,480) #inception module 3b
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    x=inception_block(x,512) #inception module 4a
    
    #auxiliary loss 1 goes here
    #=======================================================================
    aux1=AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='same')(x)
    aux1=Conv2D(128,(1,1))(aux1)
    aux1=Activation('relu')(aux1)
    aux1=Flatten()(aux1)
    aux1=Dense(units=1024,activation='relu')(aux1)
    aux1=Dropout(0.7)(aux1)
    aux1_out=Dense(units=n_classes,activation='softmax')(aux1)
    #=======================================================================

    x=inception_block(x,512) #inception module 4b
    x=inception_block(x,512) #inception module 4c
    x=inception_block(x,528) #inception module 4d
    
    #auxiliary loss 2 goes here
    #=======================================================================
    aux2=AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='same')(x)
    aux2=Conv2D(128,(1,1))(aux2)
    aux2=Activation('relu')(aux2)
    aux2=Flatten()(aux2)
    aux2=Dense(units=1024,activation='relu')(aux2)
    aux2=Dropout(0.7)(aux2)
    aux2_out=Dense(units=n_classes,activation='softmax')(aux2)
    #=======================================================================
    
    x=inception_block(x,832) #inception module 4e
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    x=inception_block(x,832) #inception module 5a
    x=inception_block(x,1024) #inception module 5b
    x=AveragePooling2D(pool_size=(7,7),strides=1)(x)
    x=Dropout(0.4)(x)
    x=Dense(units=1000)(x)
    x=Dense(units=n_classes)(x)
    out=Activation('softmax')(x)

    inception_model=Model(inputs=In,outputs=[aux1_out,aux2_out,out])
    plot_model(inception_model,to_file='./Images/inception.png',show_shapes=True,show_layer_names=False)
    return inception_model

def ResNet50(input_shape=(224,224,3), n_classes=10):
    '''
        Hàm khởi tạo model ResNet50.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model ResNet50 với softmax
    '''
    In=Input(shape=input_shape)
    x=Conv2D(64,(7,7),strides=2,padding='same')(In)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    #stage 1 starts here
    x=conv_block(input_tensor=x,stage=1,output_channel=256)
    x=identity_block(input_tensor=x,stage=1,output_channel=256)
    x=identity_block(input_tensor=x,stage=1,output_channel=256)
    
    #stage 2 starts here
    x=conv_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)
    x=identity_block(input_tensor=x,output_channel=512,stage=2)

    #stage 3 starts here 
    x=conv_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    x=identity_block(input_tensor=x,output_channel=1024,stage=2)
    
    #stage4 starts here
    x=conv_block(input_tensor=x,output_channel=2048,stage=2)
    x=identity_block(input_tensor=x,output_channel=2048,stage=2)
    x=identity_block(input_tensor=x,output_channel=2048,stage=2)

    x=AveragePooling2D(pool_size=(2,2))(x)
    
    #Fully connected
    x=Flatten()(x)
    x=Dense(units=n_classes,activation='softmax')(x)
    
    resnet50=Model(inputs=In,outputs=x)
    plot_model(resnet50,to_file='./Images/ResNet50.png',show_shapes=True,show_layer_names=False)

    return resnet50

def DenseNet(input_shape=(224,224,3), n_classes=10):
    '''
        Hàm khởi tạo model Densenet.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model Densenet với softmax
    '''
    In=Input(shape=input_shape)
    x=Conv2D(64,(7,7),strides=2,padding='same')(In)
    x=MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x)
    x=dense_block(x,6)
    x=transition_block(x)

    x=dense_block(x,12)
    x=transition_block(x)

    x=dense_block(x,24)
    x=transition_block(x)

    x=dense_block(x,16)

    x=keras.layers.GlobalAveragePooling2D()(x)
    x=Dense(n_classes,activation='softmax')(x)
    densenet=Model(inputs=In,outputs=x)
    plot_model(densenet,to_file='./Images/DenseNet121.png',show_shapes=True,show_layer_names=False)
    
    return densenet

def MobileNet(input_shape=(224,224,3), n_classes=10,alpha=1.0):
    '''
        Hàm khởi tạo model MobileNetv1.
        đầu vào: kích thước của ảnh.(width,height,channel)
        đầu ra: model MobileNetv1 với softmax
    '''
    In=Input(shape=input_shape)
    x=mobilenet_conv_block(In,32,(7,7),(2,2),alpha)

    layer=[ (64, (1, 1)), (128, (2, 2)), (128, (1, 1)), 
            (256, (2, 2)),  (256, (1, 1)), (512, (2, 2)), 
            (512, (1, 1)), (512, (1, 1)), (512, (1, 1)), 
            (512, (1, 1)), (512, (2, 2)), (1024, (1, 1)), 
            (1024, (1, 1))]
    for filter,stride in layer:
        x=depthwise_sep_conv(x,filter,strides=stride)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=Dense(units=1000)(x)
    x=Activation('relu')(x)
    x=Dense(units=n_classes,activation='softmax')(x)

    mobilenet=Model(inputs=In,outputs=x)
    plot_model(mobilenet,to_file='./Images/MobileNetv1.png',show_shapes=True,show_layer_names=False)
    
    return mobilenet








