import subprocess
import os

from random import randint

from utils import save_to_temp

def generate_random_string(length):
    return ''.join(str(randint(0, 1)) for _ in range(length))

def compression_length(binary_string, compressor='gzip', header_length=224, tempfile='temp/temp'):
    """Returns length of compressed string in bits minus size of compressor header
    Obs: "header_length" is not really constant for gzip, but it is constant for
    pseudorandom strings of length at least 1000"""
    if binary_string == '': # Note that in this case no file will be written
        return 0
    save_to_temp(binary_string, 'binary', tempfile=tempfile)
    if compressor == 'zip':
        suffix = '.zip'
        subprocess.run(['zip', tempfile, tempfile], check=True)
    elif compressor == 'gzip':
        suffix = '.gz'
        subprocess.run(['gzip', '-kf', '--best', tempfile], check=True)
    else:
        raise ValueError
    filesize = os.path.getsize(f'{tempfile}{suffix}') # in bytes, includes header
    length =  filesize * 8 - header_length
    if length < 0:
        length = 0
    return length

### Testing

def random_vs_nonrandom(length):
    assert length % 2 == 0
    random_string = generate_random_string(length)
    not_random_string = '01' * (length//2)
    length_random_string = compression_length(random_string)
    print("Compressed size for random string:", length_random_string)
    length_not_random_string = compression_length(not_random_string)
    print("Compressed size for NON-random string:", length_not_random_string)

if __name__ == "__main__":
    for n in [504, 1000, 5000, 10000, 50000, 100000]:
        print("L =", n)
        random_vs_nonrandom(n)
    # Looking at the pseudorandom strings, we see that gzip adds 28 "header"
    # bytes
