# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import h5py
import numpy as np


def get_learning_rate(cnn_type):
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_DROPOUT':
        return 0.00004
    elif cnn_type == 'VGG16_KERAS':
        return 0.00005
    elif cnn_type == 'VGG19':
        return 0.00003
    elif cnn_type == 'VGG19_KERAS':
        return 0.00005
    elif cnn_type == 'RESNET50':
        return 0.00004
    elif cnn_type == 'INCEPTION_V3':
        return 0.00003
    elif cnn_type == 'SQUEEZE_NET':
        return 0.00003
    elif cnn_type == 'DENSENET_161':
        return 0.00003
    elif cnn_type == 'DENSENET_121':
        return 0.00001
    else:
        print('Error Unknown CNN type for learning rate!!')
        exit()
    return 0.00005


def get_optim(cnn_type, optim_type, learning_rate=-1):
    from keras.optimizers import SGD
    from keras.optimizers import Adam

    if learning_rate == -1:
        lr = get_learning_rate(cnn_type)
    else:
        lr = learning_rate
    if optim_type == 'Adam':
        optim = Adam(lr=lr)
    else:
        optim = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    print('Using {} optimizer. Learning rate: {}'.format(optim_type, lr))
    return optim


def get_random_state(cnn_type):
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_DROPOUT':
        return 51
    elif cnn_type == 'VGG19_KERAS':
        return 52
    elif cnn_type == 'RESNET50':
        return 53
    elif cnn_type == 'INCEPTION_V3':
        return 54
    elif cnn_type == 'VGG16_KERAS':
        return 55
    elif cnn_type == 'VGG19':
        return 56
    elif cnn_type == 'SQUEEZE_NET':
        return 66
    elif cnn_type == 'DENSENET_161':
        return 71
    elif cnn_type == 'DENSENET_121':
        return 72
    else:
        print('Error Unknown CNN Type for random state!!')
        exit()
    return 0


def get_input_shape(cnn_type):
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'XCEPTION':
        return (299, 299)
    elif cnn_type == 'SQUEEZE_NET':
        return (227, 227)
    return (224, 224)


# Tuned for 6 GB of GPU memory
def get_batch_size(cnn_type):
    if cnn_type == 'VGG19' or cnn_type == 'VGG19_KERAS':
        return 18
    if cnn_type == 'VGG16_DROPOUT':
        return 15
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_KERAS':
        return 20
    if cnn_type == 'RESNET50':
        return 20
    if cnn_type == 'INCEPTION_V3':
        return 22
    if cnn_type == 'SQUEEZE_NET':
        return 40
    if cnn_type == 'DENSENET_161':
        return 8
    if cnn_type == 'DENSENET_121':
        return 25
    return -1


def normalize_image_vgg16(img):
    img[:, 0, :, :] -= 103.939
    img[:, 1, :, :] -= 116.779
    img[:, 2, :, :] -= 123.68
    return img


def normalize_image_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def normalize_image_densenet(img):
    img[:, 0, :, :] = (img[:, 0, :, :] - 103.94) * 0.017
    img[:, 1, :, :] = (img[:, 1, :, :] - 116.78) * 0.017
    img[:, 2, :, :] = (img[:, 2, :, :] - 123.68) * 0.017
    return img


def preprocess_input_overall(cnn_type, x):
    if cnn_type == 'INCEPTION_V3':
        return normalize_image_inception(x.astype(np.float32))
    if 'DENSENET' in cnn_type:
        return normalize_image_densenet(x.astype(np.float32))
    return normalize_image_vgg16(x.astype(np.float32))


def VGG_16(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    # VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    if 1:
        f = h5py.File('../weights/vgg16_weights.h5')
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    model.add(Dense(classes_number, activation='softmax'))

    optim = get_optim('VGG16', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# MIN: 0.98 Time: 130 sec
def VGG_16_KERAS(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg16 import VGG16
    from keras.models import Model

    base_model = VGG16(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    vgg16 = Model(input=base_model.input, output=x)

    optim = get_optim('VGG16_KERAS', optim_name, learning_rate)
    vgg16.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(vgg16.summary())
    return vgg16


# MIN: 1.00 Fast: 60 sec
def VGG_16_2_v2(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.layers import Input

    input_tensor = Input(shape=(3, 224, 224))
    base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    vgg16 = Model(input=base_model.input, output=x)

    optim = get_optim('VGG16_KERAS', optim_name, learning_rate)
    vgg16.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(vgg16.summary())
    return vgg16


def VGG_19(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    # VGG19: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    f = h5py.File('../weights/vgg19_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(classes_number, activation='softmax'))

    optim = get_optim('VGG19', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def VGG_19_KERAS(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg19 import VGG19
    from keras.models import Model

    base_model = VGG19(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    optim = get_optim('VGG19_KERAS', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


def RESNET_50(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model

    base_model = ResNet50(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    optim = get_optim('RESNET50', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


def Inception_V3(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model

    base_model = InceptionV3(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    optim = get_optim('INCEPTION_V3', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


def Xception_wrapper(classes_number, optim_name='Adam', learning_rate=-1):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.xception import Xception
    from keras.models import Model

    # Only tensorflow
    base_model = Xception(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    optim = get_optim('Xception_wrapper', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def Squeeze_Net(classes_number, optim_name='Adam', learning_rate=-1):
    from a01_squeezenet import get_squeezenet
    model = get_squeezenet(classes_number, dim_ordering='th')
    optim = get_optim('SQUEEZE_NET', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def VGG16_WITH_DROPOUTS(classes_number, dropout=0.1, optim_name='Adam', learning_rate=-1):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
    base_model = VGG_16(classes_number)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[1].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[3].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[6].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[8].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[11].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[13].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[15].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[18].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[20].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[22].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[25].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[27].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[29].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', weights=base_model.layers[32].get_weights()))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', weights=base_model.layers[34].get_weights()))
    model.add(Dropout(0.5))
    model.add(Dense(classes_number, activation='softmax'))

    optim = get_optim('VGG16', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def DenseNet161(classes_number, optim_name='Adam', learning_rate=-1):
    from a01_densenet_161 import DenseNet_161
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = DenseNet_161(reduction=0.5, weights_path='../weights/densenet161_weights_th.h5')
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    optim = get_optim('DENSENET_161', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def DenseNet121(classes_number, optim_name='Adam', learning_rate=-1):
    from a01_densenet_121 import DenseNet_121
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = DenseNet_121(reduction=0.5, weights_path='../weights/densenet121_weights_th.h5')
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    optim = get_optim('DENSENET_121', optim_name, learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_pretrained_model(cnn_type, classes_number, optim_name='Adam', learning_rate=-1):
    if cnn_type == 'VGG16':
        model = VGG_16(classes_number, optim_name, learning_rate)
    elif cnn_type == 'VGG16_DROPOUT':
        model = VGG16_WITH_DROPOUTS(classes_number, 0.1, optim_name, learning_rate)
    elif cnn_type == 'VGG19':
        model = VGG_19(classes_number, optim_name, learning_rate)
    elif cnn_type == 'VGG16_KERAS':
        model = VGG_16_KERAS(classes_number, optim_name, learning_rate)
    elif cnn_type == 'VGG19_KERAS':
        model = VGG_19_KERAS(classes_number, optim_name, learning_rate)
    elif cnn_type == 'RESNET50':
        model = RESNET_50(classes_number, optim_name, learning_rate)
    elif cnn_type == 'INCEPTION_V3':
        model = Inception_V3(classes_number, optim_name, learning_rate)
    elif cnn_type == 'SQUEEZE_NET':
        model = Squeeze_Net(classes_number, optim_name, learning_rate)
    elif cnn_type == 'DENSENET_161':
        model = DenseNet161(classes_number)
    elif cnn_type == 'DENSENET_121':
        model = DenseNet121(classes_number)
    else:
        model = None
        print('Unknown CNN type: {}'.format(cnn_type))
        exit()
    return model
