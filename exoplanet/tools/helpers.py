# global variables should all been put here
from .metaclass import NotInstantiable


class GlobalVars(metaclass=NotInstantiable):
    _vars = {}

    @staticmethod
    def register(name, value, overwrite=False):
        if not isinstance(name, str):
            raise TypeError(f'{name} is not str type')
        if name not in GlobalVars._vars or overwrite:
            GlobalVars._vars[name] = value

    @staticmethod
    def deregister(name):
        if not isinstance(name, str):
            raise TypeError(f'{name} is not str type')
        GlobalVars._vars.pop(name, None)

    @staticmethod
    def clear():
        GlobalVars._vars.clear()

    @staticmethod
    def has_var(name):
        if not isinstance(name, str):
            raise TypeError(f'{name} is not str type')
        return name in GlobalVars._vars

    @staticmethod
    def get_all_vars():
        return GlobalVars._vars

    @staticmethod
    def get_var(name):
        if not isinstance(name, str):
            raise TypeError(f'{name} is not str type')
        if not GlobalVars.has_var(name):
            return None
        return GlobalVars._vars[name]
