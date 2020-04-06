import keras
import os
import cv2
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, AveragePooling2D, BatchNormalization
from keras.layers import Activation, ZeroPadding2D, Flatten, Input, Add, concatenate,Concatenate,DepthwiseConv2D
from keras.models import Sequential, Model
from keras.utils import plot_model

#ResNet Util
def identity_block(input_tensor, output_channel,stage=1):
    re_c=64*(2**(stage-1))
    x=Conv2D(re_c,(1,1),strides=1,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    x=Conv2D(re_c,(3,3),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)

    x=Conv2D(output_channel,(1,1),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    
    x=Add()([x,input_tensor])
    x=Activation('relu')(x)
    return x
    

def conv_block(input_tensor, output_channel, stage=1):
    
    shortcut=input_tensor
    shortcut=Conv2D(output_channel,(1,1),strides=1,padding='same')(shortcut)
    shortcut=BatchNormalization(axis=3)(shortcut)
    


    re_c=64*(2**(stage-1))
    x=Conv2D(re_c,(1,1),strides=1,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    x=Conv2D(re_c,(3,3),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)

    x=Conv2D(output_channel,(1,1),strides=1,padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    
    x=Add()([x,shortcut])
    x=Activation('relu')(x)
    return x

#inceptionNet Utils
def inception_block(input_tensor, dimension_out):
    reduction = dimension_out//4
    remain = dimension_out-reduction*4
    conv1 = Conv2D(reduction+remain, (1, 1), strides=1,
                   padding='same')(input_tensor)
    conv1 = Activation('relu')(conv1)

    conv3 = Conv2D(reduction, (1, 1), strides=1)(input_tensor)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(reduction, (3, 3), strides=1, padding='same')(conv3)
    conv3 = Activation('relu')(conv3)

    conv5 = Conv2D(reduction, (1, 1), strides=1, padding='same')(input_tensor)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(reduction, (5, 5), strides=1, padding='same')(conv5)
    conv5 = Activation('relu')(conv5)

    maxpool = MaxPool2D(pool_size=(3, 3), strides=1,
                        padding='same')(input_tensor)
    maxpool = Conv2D(reduction, (1, 1), strides=1, padding='same')(maxpool)

    output = concatenate([conv1, conv3, conv5, maxpool], axis=3)

    return output

#DenseNet utils
def dense_block(input_tensor,nb_layers):
    """
        a block of dense layers
    """
    for i in range(nb_layers):
        input_tensor=dense_layer(input_tensor,32)
    return input_tensor

def transition_block(input_tensor):
    filter=K.int_shape(input_tensor)[3]
    x=BatchNormalization(axis=3)(input_tensor)
    x=Activation('relu')(x)
    x=Conv2D(filter,(1,1))(x)
    x=AveragePooling2D(pool_size=(2,2),strides=2,padding='same')(x)

    return x

def dense_layer(input_tensor,filter):
    """
       BN-RELU-Conv(1,1) _ BN-RELU-Conv(3,3)
    """
    out=BatchNormalization(axis=3)(input_tensor)
    out=Activation('relu')(out)
    out=Conv2D(filter*4,(1,1),strides=1,use_bias=False)(out)

    out=BatchNormalization(axis=3)(input_tensor)
    out=Activation('relu')(out)
    out=Conv2D(filter,(3,3),strides=1,padding='same')(out)

    output=Concatenate(axis=3)([out,input_tensor])

    return output


def mobilenet_conv_block(input_tensor,filter,kernel_size,strides,alpha=1.0):
    x=Conv2D(int(filter*alpha),kernel_size,strides=strides,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    return x

def depthwise_sep_conv(input_tensor,n_filters,strides=(1,1),Dk=3,alpha=1.0):
    x=DepthwiseConv2D(kernel_size=(Dk,Dk),strides=strides,padding='same')(input_tensor)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    x=Conv2D(int(n_filters*alpha),(1,1),strides=1,use_bias=False)(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)

    return x