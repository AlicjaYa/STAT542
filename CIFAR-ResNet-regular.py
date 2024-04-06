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
from ResNet_Models import residual_network
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

stack_n = 5
layers = 6 * stack_n + 2
num_classes = 10
batch_size = 128
epochs = 80
iterations = 50000 // batch_size + 1
weight_decay = 1e-4

log_filepath = '/data0/jinhaibo/Remote/STAT/logs/ResNet50-regular'

def residual_block(x, o_filters_1, o_filters_2, increase=False):

    stride = (1, 1)
    if increase:
        stride = (2, 2)

    conv_1 = Conv2D(o_filters_1, kernel_size=(1, 1), strides=stride, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(x)

    o1 = Activation('relu')(conv_1)
#    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))

    conv_2 = Conv2D(o_filters_1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o1)
    o2 = Activation('relu')(conv_2)
#    o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_2))

    conv_3 = Conv2D(o_filters_2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o2)

    o3 = Activation('relu')(conv_3)
#    o3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_3))
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

#    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
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
    if epoch < 30:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

img_input = Input(shape=(32, 32, 3))
output = residual_network(img_input, 10)  # 5
resnet = Model(img_input, output)

# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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
resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     validation_data=(x_test, y_test))
'''
resnet.fit(x_train, y_train, batch_size=batch_size,
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     validation_data=(x_test, y_test))

resnet.save('/data0/jinhaibo/Remote/STAT/models/resnet50-regular.h5')
