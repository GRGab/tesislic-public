import contextlib
import functools
import time
from datetime import datetime
from typing import Literal, Optional

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_(%H:%M:%S.%f)")

#### Save to temporal files
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def save_to_temp(binary_string : str,
                 mode : Literal['binary', 'text'],
                 tempfile : str = 'temp/temp'):
    if mode == 'binary':
        with open(tempfile, 'wb') as f:
            # byte_arr = [int("0b" + byte, 2) for byte in chunks(binary_string, 8)]
            byte_arr = [int(byte, 2) for byte in chunks(binary_string, 8)]
            binary_format = bytearray(byte_arr)
            f.write(binary_format)
    elif mode == 'text':
        with open(tempfile, 'w') as f:
            f.write(binary_string)
    else:
        raise ValueError

#### Useful decorators
# https://code.activestate.com/recipes/577089-context-manager-to-temporarily-set-an-attribute-on/
@contextlib.contextmanager
def temp_setattr(ob, attr, new_value):
    """Temporarily set an attribute on an object for the duration of the
    context manager."""
    replaced = False
    old_value = None
    if hasattr(ob, attr):
        try:
            if attr in ob.__dict__:
                replaced = True
        except AttributeError:
            if attr in ob.__slots__:
                replaced = True
        if replaced:
            old_value = getattr(ob, attr)
    setattr(ob, attr, new_value)
    yield replaced, old_value
    if not replaced:
        delattr(ob, attr)
    else:
        setattr(ob, attr, old_value)

def timer(telegram=False):
    def inner_decorator(func):
        """Print the runtime of the decorated function"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()  
            run_time = end_time - start_time
            message = f"Finished {func.__name__!r} in {run_time:.4f} secs"
            message = 'Finished {!r} in {:.4} secs'.format(func.__name__, run_time)
            print(message)
            if telegram:
                send_telegram("Finished in {:.4} secs".format(run_time))
            return value
        return wrapper_timer
    return inner_decorator

def send_telegram(message):
    try: # this will work in my local repository
        import telegram_bot
        telegram_bot.telegram_bot_sendtext(message)
    except ModuleNotFoundError:
        pass

##### Custom InitializationError
class InitializationError(Exception):
    pass
