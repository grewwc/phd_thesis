from tensorflow import keras
from config import GlobalVars
from preprocess.get_more_feathres import get_more_features


# it seems adding more features is not a good idea
# feature names
def get_dense(learning_rate=1e-3):
    l1, l2 = 1e-4, 1e-4
    df = get_more_features()

    inputs = keras.layers.Input(shape=(len(df.columns),))
    h1, h2, h3, h4 = 512, 512, 512, 512

    x = inputs

    x = keras.layers.Dense(
        units=h1, activation='relu',
    )(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(
        units=h2, activation='relu',
    )(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        units=h3, activation='relu',
    )(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        units=h4, activation='relu',
        # name='last_dropout'
    )(x)
    x = keras.layers.Dropout(0.5, name='last_dropout')(x)
    # dense layers

    x = keras.layers.Dense(units=2, activation='softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model
