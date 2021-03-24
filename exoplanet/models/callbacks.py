import tensorflow as tf
import tensorflow.keras.backend as K


class LowerLr(tf.keras.callbacks.Callback):
    def __init__(self, epoch_list, lr_list):
        """
        epoch_list: e.g.: should be [5,5,5], not [5,10,15]
        """
        assert len(epoch_list) == len(lr_list), \
            "Length of epoch list and lr_list must be the same"
        self.__epoch_list = epoch_list
        self.__lr_list = lr_list
        self.__epoch_idx = 0
        self.__prev_lr_change_idx = 0

    @property
    def epoch_list(self):
        return self.__epoch_list

    @property
    def lr_list(self):
        return self.__lr_list

    def __set_lr(self, lr):
        K.set_value(self.model.optimizer.lr, lr)
        print(f"Set Learning Rate: {lr}")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.__set_lr(self.lr_list[0])

        elif epoch >= self.epoch_list[self.__epoch_idx] + self.__prev_lr_change_idx:
            self.__epoch_idx += 1
            if self.__epoch_idx < len(self.lr_list):
                self.__set_lr(self.lr_list[self.__epoch_idx])
                self.__prev_lr_change_idx = epoch
            else:
                self.model.stop_training = True



class PrintMinMaxCallback(tf.keras.callbacks.Callback):
  def on_batch_end(self, batch, logs=None):
    m = self.model 
    m.predict(m.inputs).max()