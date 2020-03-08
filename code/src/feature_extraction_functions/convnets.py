# Source: https://github.com/robintibor/arl-eegmodels/blob/master/EEGModels.py

from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Flatten, DepthwiseConv2D, SpatialDropout2D,
                                     Lambda, Dense, Activation, Dropout)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,
                                     SeparableConv2D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
K.set_image_data_format("channels_first")


def EEGNet(nb_classes=4, n_channels=22, n_samples=int(250*7.5),
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, dropoutType='Dropout', undersample=1):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024) """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, n_channels, n_samples))

    # Temporal convolution to learn frequency filters (linear activation)
    block1 = Conv2D(F1//undersample, (1, kernLength//undersample), padding='same',
                    input_shape=(1, n_channels, n_samples),
                    use_bias=False)(input1)

    # BN w.r.t. temporal axis
    block1 = BatchNormalization(axis=1)(block1)

    # Depthwise convolution to learn spatial filters (linear activation)
    block1 = DepthwiseConv2D((n_channels, 1),
                             use_bias=False,
                             # Defines nbr of output filters (D*F1//undersample)
                             depth_multiplier=D,
                             padding='valid',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)

    # Temporal average pooling
    block1 = AveragePooling2D((1, 4//undersample))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Separable convolution (linear activation)
    block2 = SeparableConv2D(F2//undersample,
                             (1, 16),
                             use_bias=False,
                             padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)

    # Temporal average pooling
    block2 = AveragePooling2D((1, 8//undersample))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes,
                  name='dense',
                  kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(nb_classes=4, n_channels=22, n_samples=1875,
                dropoutRate=0.5, undersample=1, l2_reg=0.01):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping."""

    # start the model
    input_main = Input((1, n_channels, n_samples))
    block1 = Conv2D(25, (1, 10//undersample),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2.))(input_main)
    block1 = Conv2D(25, (n_channels, 1), kernel_constraint=max_norm(
        2.), kernel_regularizer=l2(l2_reg))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 10//undersample), kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 10//undersample), kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 10//undersample), kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def ShallowConvNet(nb_classes=4, n_channels=22, n_samples=1125, dropoutRate=0.5, l2_reg=0.01, undersample=1):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping."""
    def square(x):
        return K.square(x)

    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))

    # start the model
    input_main = Input((1, n_channels, n_samples))

    # Temporal filtering
    block1 = Conv2D(40, (1, 50//undersample),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=l2(l2_reg),
                    name='conv1')(input_main)

    # Spatial filtering
    block1 = Conv2D(40, (n_channels, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=l2(l2_reg),
                    name='conv2')(block1)

    # Normalization before non-linearity
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)

    # Similar to power band feature extraction (log-variance)
    block1 = Lambda(lambda x: K.square(x))(block1)
    block1 = AveragePooling2D(pool_size=(1, 150 // undersample),
                              strides=(1, 30//undersample))(block1)
    block1 = Lambda(lambda x: K.log(
        K.clip(x, min_value=1e-7, max_value=10000)))(block1)

    # Classifier
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes,
                  kernel_constraint=max_norm(0.5),
                  name='fc')(flatten)

    # Probabilities
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DevConvNet(nb_classes=4, n_channels=22, n_samples=1125, dropoutRate=0.5, l2_reg=0.01, undersample=1, activation='elu', initializer='glorot_uniform', bn=True, dropout=Dropout):
    def square(x):
        return K.square(x)

    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))

    # start the model
    input_main = Input((1, n_channels, n_samples))

    # Temporal filtering
    block1 = Conv2D(40, (1, 25//undersample),
                    kernel_initializer=initializer,
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=l2(l2_reg),
                    use_bias=False,  # New
                    name='conv1')(input_main)

    # Spatial filtering
    block1 = Conv2D(40, (n_channels, 1),
                    kernel_initializer=initializer,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=l2(l2_reg),
                    use_bias=False,  # New
                    name='conv2')(block1)

    # Normalization before non-linearity
    if bn:
        block1 = BatchNormalization(
            axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(activation)(block1)

    # Similar to power band feature extraction (log-variance)
    block1 = Lambda(lambda x: K.square(x))(block1)
    block1 = AveragePooling2D(pool_size=(
        1, 75//undersample), strides=(1, 15//undersample))(block1)
    block1 = Lambda(lambda x: K.log(
        K.clip(x, min_value=1e-7, max_value=10000)))(block1)

    # Classifier
    block1 = dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes,
                  kernel_initializer=initializer,
                  kernel_constraint=max_norm(0.5), name='fc')(flatten)

    # Probabilities
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
