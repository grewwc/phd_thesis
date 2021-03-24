from preprocess.kepler_io import *
from models import *
from train.utils import augment_data
import matplotlib.pyplot as plt
import sys


def train(lr=1e-4,
          epochs=100,
          num_augment=3,
          plot_hist=False,
          from_start=True,
          h5_path=None):
    
    default_h5_path = os.path.join(os.getcwd(), 'temp', 'temp.h5')

    h5_path = default_h5_path if h5_path is None else h5_path

    (train_x, train_y), (test_x, test_y) = get_train_test_data()
    if from_start:
        model = get_model(learning_rate=lr)
    else:
        try:
            model = load_model(h5_path)
        except Exception as e:
            print(f'cannot load model from {h5_path}')
            print(e)
            sys.exit(-1)
    train_x_aug, train_y_aug = augment_data(train_x, train_y, n=num_augment)
    hist = model.fit(train_x_aug, train_y_aug, epochs=epochs, verbose=1)
    if plot_hist:
        plt.plot(hist)
        plt.show()

    save_model(model, h5_path)
