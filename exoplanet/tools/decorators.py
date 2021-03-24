from contextlib import contextmanager


@contextmanager
def load_ctx(filename):
    print(f"loading data from {filename}")
    try:
        yield
    except:
        print(f'Error loading from {filename}')
    finally:
        print(f'finished loading')


@contextmanager
def save_ctx(filename):
    print(f"\nsaving to {filename}")
    try:
        yield
    except:
        print(f'Error saving to {filename}')
    finally:
        print(f'finished saving')
