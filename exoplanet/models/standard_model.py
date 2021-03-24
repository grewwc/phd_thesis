from .utils import *
from .utils import __get_optimizer  # private methods need special import



def __get_global_view_model():
    inputs = tf.keras.Input(shape=(2001, 1), name='global_input')
    x = inputs
    x = Conv1D(filters=16, kernel_size=5, padding='same',
               activation='relu', name='global-conv1-1')(x)
    x = Conv1D(filters=16, kernel_size=5, padding='same',
               name='global-conv1-2', activation='relu')(x)

    x = MaxPooling1D(pool_size=5, strides=2, name='global-maxpool-1')(x)

    x = Conv1D(filters=32, kernel_size=5,
               activation='relu', name='global-conv2-1')(x)
    x = Conv1D(filters=32, kernel_size=5,
               activation='relu', name='global-conv2-2')(x)
    x = MaxPooling1D(pool_size=5, strides=2, name='global-maxpool-2')(x)

    x = Conv1D(filters=64, kernel_size=5,
               activation='relu', name='global-conv3-1')(x)
    x = Conv1D(filters=64, kernel_size=5,
               activation='relu', name='global-conv3-2')(x)
    x = MaxPooling1D(pool_size=5, strides=2, name='global-maxpool-3')(x)

    x = Conv1D(filters=128, kernel_size=5,
               activation='relu', name='global-conv4-1')(x)
    x = Conv1D(filters=128, kernel_size=5,
               activation='relu', name='global-conv4-2')(x)
    x = MaxPooling1D(pool_size=5, strides=2, name='global-maxpool-5')(x)

    x = Conv1D(filters=256, kernel_size=5,
               activation='relu', name='global-conv5-1')(x)
    x = Conv1D(filters=256, kernel_size=5,
               activation='relu', name='global-conv5-2')(x)
    x = MaxPooling1D(pool_size=5, strides=2, name='global-maxpool-6')(x)

    x = Flatten(name='global-flatten')(x)

    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def __get_local_view_model():
    inputs = tf.keras.Input(shape=[201, 1], name='local-input')
    x = inputs

    x = Conv1D(filters=16, kernel_size=5,
               activation='relu', name='local-conv1-1')(x)
    x = Conv1D(filters=16, kernel_size=5,
               activation='relu', name='local-conv1-2')(x)
    x = MaxPooling1D(pool_size=7, strides=2, name='local-maxpool-1')(x)

    x = Conv1D(filters=32, kernel_size=3,
               activation='relu', name='local-conv2-1')(x)
    x = Conv1D(filters=32, kernel_size=3,
               activation='relu', name='local-conv2-2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='local-maxpool-2')(x)

    x = Flatten(name='local-flatten')(x)

    outputs = x
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model




def get_model(learning_rate=1e-3):
    global_model = __get_global_view_model()
    local_model = __get_local_view_model()

    combined = tf.keras.layers.concatenate(
        [global_model.output, local_model.output])

    h1, h2, h3, h4 = 256, 256, 256, 256
    # combined=global_model.output
    x = Dense(units=h1, activation='relu', name='fc1')(combined)
    x = Dropout(0.7)(x)
    x = Dense(units=h2, activation='relu', name='fc2')(x)
    x = Dropout(0.7)(x)
    x = Dense(units=h3, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(units=h4, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(units=1, activation='sigmoid', name='classifier')(x)
    outputs = x

    model = tf.keras.models.Model(inputs=[global_model.input, local_model.input],
                                  outputs=outputs)

    optimizer = __get_optimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['acc'])
    return model
