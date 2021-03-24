from tensorflow import keras

from .resNet import get_resnet_model

from .dense import get_dense


def get_model(learning_rate=1e-3, dense=None):
    resnet = get_resnet_model(learning_rate=learning_rate)
    if dense is None:
        dense = get_dense(learning_rate=learning_rate)
    else:
        print("setting dense model NOT trainable")
        dense.trainable = False

    x1 = resnet.get_layer("before_softmax")
    x1 = keras.layers.Dense(units=1)(x1.output)
    # add one dense layer to get only 1 output

    x2 = dense.get_layer("last_dropout")
    scale = 1.0
    x2 = keras.layers.Lambda(lambda x: x * scale)(x2.output)
    # scale the dense layer by "scale"
    x2 = keras.layers.Dense(units=1)(x2)
    # add one dense layer to get only 1 output

    x = keras.layers.concatenate([x1, x2])

    x = keras.layers.Dense(units=2, activation='softmax')(x)

    model = keras.models.Model(inputs=[resnet.input, dense.input],
                               outputs=x)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model
