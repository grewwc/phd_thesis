class Singleton(type):
    __instance = None

    def __new__(mcls, name, bases, attrs):
        if Singleton.__instance is None:
            Singleton.__instance = super().__new__(mcls, name, bases, attrs)
        return Singleton.__instance


class NotInstantiable(Singleton):
    def __call__(cls, *args, **kwargs):
        print(f"class {NotInstantiable.__qualname__}"
              f" is not instantiable")
        return None
