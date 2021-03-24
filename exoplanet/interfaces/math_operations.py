from abc import ABCMeta, abstractmethod, ABC


class SingletonMeta(type):
    _instance = None

    def __new__(mcls, clsname, bases, attrs):
        return type.__new__(mcls, clsname, bases, attrs)

    def __call__(cls, *args):
        if SingletonMeta._instance is None:
            SingletonMeta._instance = type.__call__(cls)
        return SingletonMeta._instance


class ABCSingletonMeta(ABCMeta, SingletonMeta):
    pass


class MathOpsBase(metaclass=ABCSingletonMeta):
    @abstractmethod
    def mask_outliers(self, data, n_sigma, *args):
        """
        data: array-like 
              The data to be sigma clipped.
        n_sigma: float
                 The number of standard deviations to use
        """
