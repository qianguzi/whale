import numpy as np
import tensorflow as tf

K = tf.keras.backend
layers = tf.keras.layers
Model = tf.keras.models.Model
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers
Input = tf.keras.Input

img_shape = (64, 64, 1)  # The image shape used by the model

def subblock(x, filter, **kwargs):
    x = layers.BatchNormalization()(x)
    y = x
    y = layers.Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = layers.Add()([x, y])  # layers.Add the bypass connection
    y = layers.Activation('relu')(y)
    return y


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = optimizers.Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = layers.Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = layers.GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = layers.Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = layers.Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = layers.Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = layers.Lambda(lambda x: K.square(x))(x3)
    x = layers.Concatenate()([x1, x2, x3, x4])
    x = layers.Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = layers.Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = layers.Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = layers.Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = layers.Flatten(name='flatten')(x)

    # Weighted sum implemented as a layers.Dense layer.
    x = layers.Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    # branch_model.compile()
    # head_model.compile()
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model
