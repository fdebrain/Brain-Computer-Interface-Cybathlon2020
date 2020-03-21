# Source: https://github.com/robintibor/arl-eegmodels/blob/master/EEGModels.py
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Flatten, DepthwiseConv2D,
                                     SpatialDropout2D, Lambda, Dense,
                                     Activation, Dropout)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,
                                     SeparableConv2D, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from kerastuner import HyperModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def ShallowConvNet(config: dict = {}):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping."""

    n_classes = config.get('n_classes', 4)
    n_channels = config.get('n_channels', 61)
    n_samples = config.get('n_samples', 250)
    n_filt1 = config.get('n_filt1', 50)
    k_size1 = config.get('k_size1', 10)
    n_filt2 = config.get('n_filt2', 80)
    p_size = config.get('p_size', 150)
    p_stride = config.get('p_stride', 30)
    k_reg = config.get('k_reg', 0.01)
    dropout = config.get('dropout', 0.5)
    lr = config.get('lr', 1e-3)
    label_smooth = config.get('label_smooth', 0.1)

    # Define sequential architecture & input
    model = Sequential()
    model.add(Input(shape=(n_channels, n_samples, 1)))

    # Temporal filtering
    model.add(Conv2D(filters=n_filt1,
                     kernel_size=(1, k_size1),
                     kernel_regularizer=l2(k_reg),
                     name='conv1'))

    # Spatial filtering
    model.add(Conv2D(filters=n_filt2,
                     kernel_size=(n_channels, 1),
                     kernel_regularizer=l2(k_reg),
                     name='conv2'))

    # Normalization
    model.add(tfa.layers.GroupNormalization())
    model.add(Activation('elu'))

    # Non-linearity
    model.add(Lambda(lambda x: K.square(x)))
    model.add(AveragePooling2D(pool_size=(1, p_size),
                               strides=(1, p_stride)))
    model.add(Lambda(lambda x: K.log(K.clip(x, min_value=1e-7,
                                            max_value=10000))))

    model.add(Dropout(rate=dropout))
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr),
                  loss=CategoricalCrossentropy(
        label_smoothing=label_smooth),
        metrics=['accuracy'])
    return model


class ShallowConvNetDev(HyperModel):
    def __init__(self, n_classes, n_channels):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = 250

    def build(self, hp):
        # Search space
        n_filt1 = hp.Int('n_filt1', min_value=10, max_value=100,
                         step=10, default=40)
        k_size1 = hp.Int('k_size1', min_value=10, max_value=70,
                         step=10, default=50)
        n_filt2 = hp.Int('n_filt2', min_value=10, max_value=100,
                         step=10, default=40)
        p_size = hp.Int('pool_size', min_value=30, max_value=150,
                        step=10, default=100)
        p_stride = hp.Int('pool_stride', min_value=10, max_value=50,
                          step=10, default=30)
        k_reg = hp.Choice('k_reg', values=[0.0, 1e-2, 1e-1])
        dropout = hp.Choice('dropout', values=[0.25, 0.5, 0.75])
        lr = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
        label_smooth = hp.Choice('label_smooth', values=[0.0, 0.1, 0.2])

        # Define sequential architecture & input
        model = Sequential()
        model.add(Input(shape=(self.n_channels, self.n_samples, 1)))

        # Temporal filtering
        model.add(Conv2D(filters=n_filt1,
                         kernel_size=(1, k_size1),
                         kernel_regularizer=l2(k_reg),
                         name='conv1'))

        # Spatial filtering
        model.add(Conv2D(filters=n_filt2,
                         kernel_size=(self.n_channels, 1),
                         kernel_regularizer=l2(k_reg),
                         name='conv2'))

        # Normalization
        model.add(tfa.layers.GroupNormalization())
        model.add(Activation('elu'))

        # Non-linearity
        model.add(Lambda(lambda x: K.square(x)))
        model.add(AveragePooling2D(pool_size=(1, p_size),
                                   strides=(1, p_stride)))
        model.add(Lambda(lambda x: K.log(K.clip(x, min_value=1e-7,
                                                max_value=10000))))

        model.add(Dropout(rate=dropout))
        model.add(Flatten())
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))

        model.compile(optimizer=Adam(lr),
                      loss=CategoricalCrossentropy(
                          label_smoothing=label_smooth),
                      metrics=['accuracy'])
        return model


def EEGNet(n_classes=4, n_channels=61, n_samples=250,
           dropoutRate=0.5, kernLength=128, F1=8,
           D=2, F2=16, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024) """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(n_channels, n_samples, 1))

    # Temporal convolution to learn frequency filters (linear activation)
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(n_channels, n_samples, 1),
                    use_bias=False)(input1)

    # BN w.r.t. temporal axis
    block1 = BatchNormalization(axis=1)(block1)

    # Depthwise convolution to learn spatial filters (linear activation)
    block1 = DepthwiseConv2D((n_channels, 1),
                             use_bias=False,
                             # Defines nbr of output filters (D*F1)
                             depth_multiplier=D,
                             padding='valid',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)

    # Temporal average pooling
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Separable convolution (linear activation)
    block2 = SeparableConv2D(F2,
                             (1, 16),
                             use_bias=False,
                             padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)

    # Temporal average pooling
    block2 = AveragePooling2D((1, 16))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(n_classes,
                  name='dense',
                  kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(n_classes=4, n_channels=61, n_samples=250,
                dropoutRate=0.5, l2_reg=0.01):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping."""

    # start the model
    input_main = Input((n_channels, n_samples, 1))
    block1 = Conv2D(filters=25, strides=(1, 10),
                    input_shape=(n_channels, n_samples, 1),
                    kernel_constraint=max_norm(2.))(input_main)
    block1 = Conv2D(25, (n_channels, 1),
                    kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(filters=50, kernel_size=(1, 10),
                    kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(filters=100, kernel_size=(1, 10),
                    kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(filters=200, kernel_size=(1, 10),
                    kernel_constraint=max_norm(2.),
                    kernel_regularizer=l2(l2_reg))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(n_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
