from tensorflow import keras


class _Counter:
    def __init__(self, func):
        self._count = 1
        self._func = func

    def __call__(self, x, filters, kernel_size,
                 strides=1, name=None):
        if name is not None:
            name = f'{name}-{self._count}'
        self._count += 1
        return self._func(
            x, filters, kernel_size, strides, name=name)

    def clear(self):
        self._count = 1


@_Counter
def res_block(x, filters, kernel_size, strides=1, name=None):
    """
    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param name:
    :return:

    strides != 1 res block should be put into the last layer
    """
    y = x
    l2 = 5e-5
    y = keras.layers.Conv1D(
        filters, kernel_size, strides=strides, padding='same',
        kernel_regularizer=keras.regularizers.l2(l2)
    )(y)

    # y = keras.layers.BatchNormalization()(y)

    y = keras.layers.Activation('relu')(y)

    y = keras.layers.Conv1D(
        filters, kernel_size, strides=1, padding='same',
        kernel_regularizer=keras.regularizers.l2(l2)
    )(y)

    if strides != 1:
        x = keras.layers.Conv1D(
            filters, 1, strides=strides, padding='same',
            kernel_regularizer=keras.regularizers.l2(l2)
        )(x)
    y = keras.layers.add([x, y])

    # y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu', name=name)(y)

    return y


def get_global_model():
    global_name = 'global'
    res_block.clear()

    inputs = keras.layers.Input(shape=[2001, 1])

    x = inputs

    x = res_block(x, 16, 3, 2, name=global_name)
    x = res_block(x, 16, 3, 2, name=global_name)

    x = res_block(x, 32, 3, 2, name=global_name)
    x = res_block(x, 32, 3, 2, name=global_name)

    x = res_block(x, 64, 5, 2, name=global_name)
    x = res_block(x, 64, 5, 2, name=global_name)

    x = res_block(x, 128, 5, 2, name=global_name)
    x = res_block(x, 128, 5, 2, name=global_name)

    x = res_block(x, 256, 5, 2, name=global_name)

    x = res_block(x, 512, 5, 2, name=global_name)

    # x = keras.layers.AveragePooling1D(2)(x)
    # x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Flatten()(x)
    print(f"{res_block._count - 1} global res_blocks")
    return keras.models.Model(inputs=inputs, outputs=x)


def get_local_model():
    local_name = 'local'
    res_block.clear()

    inputs = keras.layers.Input(shape=[201, 1])
    x = inputs

    x = res_block(x, 8, 3, 2, local_name)

    x = res_block(x, 16, 3, 2, local_name)

    x = res_block(x, 32, 5, 2, local_name)

    x = res_block(x, 64, 5, 2, local_name)

    x = res_block(x, 128, 7, 2, local_name)
    x = res_block(x, 128, 7, 2, local_name)
    x = res_block(x, 128, 7, 2, local_name)


    print(f"{res_block._count - 1} local res_blocks")

    # x = keras.layers.AveragePooling1D(2)(x)
    # x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Flatten()(x)

    return keras.models.Model(inputs=inputs, outputs=x)


def get_resnet_model(learning_rate=1e-3):
    g = get_global_model()
    l = get_local_model()
    # f = get_features_model()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # x1 = keras.layers.Dense(units=1, kernel_regularizer=keras.regularizers.l2(1e-4))(g.output)
    # x2 = keras.layers.Dense(units=1, kernel_regularizer=keras.regularizers.l2(1e-4))(l.output)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = keras.layers.concatenate([g.output, l.output], name='before_softmax')
    # x = keras.layers.concatenate([x1, x2], name='before_softmax')

    x = keras.layers.Dense(units=2, activation='softmax')(x)

    model = keras.models.Model(
        inputs=[g.input, l.input],
        outputs=x
    )

    # model = keras.models.Model(inputs = g.input,
    #    outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
