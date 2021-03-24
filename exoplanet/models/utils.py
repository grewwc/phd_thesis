import os

import tensorflow as tf
import tensorflow.keras.backend as K


def __get_optimizer(learning_rate):
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    return optimizer


def set_lr(model, learning_rate):
    K.set_value(model.optimizer.lr, learning_rate)
    print(f'set learning rate to {K.eval(model.optimizer.lr)}')


def save_model(model, path=None):
    path = os.path.join(os.path.dirname(__file__),
                        "train.h5") if path is None else path
    root_dir = os.path.dirname(path)

    try:
        print(f'saving to {path}')
        model.save(path)
        print(f'finished saving')
    except Exception as e:
        print(f'Error saving model to {path}')
        print(e)


def load_model(path=None):
    path = os.path.join(os.path.dirname(__file__),
                        "train.h5") if path is None else path

    try:
        print(f'loading model from {path}')
        model = tf.keras.models.load_model(path)
        print(f'finished loading')
        return model
    except Exception as e:
        print(f'Error loading model from {path}')
        print("loading error: ", e)
        os._exit(-1)
