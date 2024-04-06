import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.initializers import he_normal
from keras.layers import Dense, Input, add, Activation, Lambda, concatenate
from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from Dense_Models import densenet
import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
growth_rate = 12
depth = 100
compression = 0.5

img_rows, img_cols = 32, 32
img_channels = 3
num_classes = 10
batch_size = 64  # 64 or 32 or other
epochs = 80
iterations = 782
weight_decay = 1e-4

log_filepath = '/data0/jinhaibo/Remote/STAT/logs/densenet_121_regular'

def conv(x, out_filters, k_size):
    return Conv2D(filters=out_filters,
                  kernel_size=k_size,
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  use_bias=False)(x)


def dense_layer(x):
    return Dense(units=10,
                 activation='softmax',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(x)


def bn_relu(x):
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x


def bottleneck(x):
    channels = growth_rate * 4
    x = bn_relu(x)
    x = conv(x, channels, (1, 1))  # 48
    x = bn_relu(x)
    x = conv(x, growth_rate, (3, 3))  # 12
    return x


# feature map size and channels half
def transition(x, inchannels):
    outchannels = int(inchannels * compression)
    x = bn_relu(x)
    x = conv(x, outchannels, (1, 1))
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, outchannels


def dense_block(x, blocks, nchannels):
    concat = x
    for i in range(blocks):
        x = bottleneck(concat)
        concat = concatenate([x, concat], axis=-1)
        nchannels += growth_rate
    return concat, nchannels


def densenet(img_input, classes_num):
    nchannels = growth_rate * 2
    x = conv(img_input, nchannels, (3, 3))

    x, nchannels = dense_block(x, 6, nchannels)
    x, nchannels = transition(x, nchannels)

    x, nchannels = dense_block(x, 12, nchannels)
    x, nchannels = transition(x, nchannels)

    x, nchannels = dense_block(x, 24, nchannels)
    x, nchannels = transition(x, nchannels)

    x, nchannels = dense_block(x, 16, nchannels)

    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)  # 342 to 10
    return x

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

img_input = Input(shape=(img_rows, img_cols, img_channels))
output = densenet(img_input, 10)
model = Model(img_input, output)
print(model.summary())

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

# set data augmentation
'''
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)
datagen.fit(x_train)
'''
# start training
'''
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
                    '''
model.fit(x_train, y_train, batch_size=batch_size,
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
model.save('/data0/jinhaibo/Remote/STAT/models/densenet121-regular.h5')
