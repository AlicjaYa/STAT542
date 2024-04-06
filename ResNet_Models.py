import keras
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

weight_decay = 1e-4


def residual_block(x, o_filters_1, o_filters_2, increase=False):

    stride = (1, 1)
    if increase:
        stride = (2, 2)

    conv_1 = Conv2D(o_filters_1, kernel_size=(1, 1), strides=stride, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(x)

    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))

    conv_2 = Conv2D(o_filters_1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o1)
    o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_2))

    conv_3 = Conv2D(o_filters_2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o2)

    o3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_3))
    if increase:
        projection = Conv2D(o_filters_2, kernel_size=(1, 1), strides=stride, padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(weight_decay))(x)
        block = add([o3, projection])
    else:
        block = add([o3, x])
    return block


def residual_network(img_input, classes_num=10):
    # input: 32x32x3 output: 32x32x64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x64 output: 16x16x256
    x = residual_block(x, 64, 256, True)
    for _ in range(2):
        x = residual_block(x, 64, 256, False)

    # input: 16x16x256 output: 8*8*512
    x = residual_block(x, 128, 512, True)
    for _ in range(3):
        x = residual_block(x, 128, 512, False)

    # input: 8*8*512 output: 4*4*1024
    x = residual_block(x, 256, 1024, True)
    for _ in range(5):
        x = residual_block(x, 256, 1024, False)

    # input: 4*4*1024 output: 2*2*2048
    x = residual_block(x, 512, 2048, True)
    for _ in range(2):
        x = residual_block(x, 512, 2048, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

