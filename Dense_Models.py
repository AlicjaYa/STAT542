import math
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
growth_rate = 12
depth = 100
compression = 0.5

img_rows, img_cols = 32, 32
img_channels = 3
num_classes = 10
batch_size = 64
epochs = 300
iterations = 782
weight_decay = 1e-4

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
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
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
