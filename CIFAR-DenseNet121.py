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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

log_filepath = '/data0/jinhaibo/Remote/STAT/logs/densenet_121'


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
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)
datagen.fit(x_train)

# start training
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
model.save('/data0/jinhaibo/Remote/STAT/models/densenet121.h5')
