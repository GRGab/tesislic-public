import numpy as np

from functools import partial, wraps
from numpy.random import rand
from random import choice

from syntactic_causal_discovery.string_transformations import left_shift, rev, string_function, symbol_function

# Structure function decorator
def structure_function(func):
    @wraps(func)
    def wrapper(shift, direction, *args):
        output = func(*args)
        if shift != 0:
            output = left_shift(output, shift)
        if direction == 'backward':
            output = rev(output)
        return output
    return wrapper
# Utils
def get_start_idx(n, k, i):
    """Returns starting index for the ith chunk when dividing a list of n things
    into k approximately equal chunks.
    Taken from https://stackoverflow.com/a/37414115"""
    return i * (n // k) + min(i, n % k)
# Sturcture function factories
def modularsumfunc_factory(modulus):
    @structure_function
    @string_function
    @symbol_function
    def modularsumfunc(*args):
        return sum(args) % modulus
    return modularsumfunc

#### Structure functions

@structure_function
def halvingconcatfunc(*args): # Previously known as LossyConcat
    n = len(args[0]) # We assume all args have equal length
    k = len(args) # number of arguments
    idx = partial(get_start_idx, n, k) # for easier reading
    output = ''.join(args[i][idx(i):idx(i+1)] for i in range(k))
    return output

@structure_function
def bernoullifunc(*args):
    n = len(args[0]) # We assume all args have equal length
    output = ''
    for i in range(n):
        output += choice([arg[i] for arg in args])
    return output

@structure_function
def erraticfunc(*args, switch_prob=0.01):
    k = len(args) # number of arguments
    if k == 1:
        return args[0] # nothing to do here. Else...
    n = len(args[0]) # We assume all args have equal length
    switch = rand(n) < switch_prob
    idx = np.where(switch)[0] # indices where copying process switches to another argument
    idx = np.concatenate(([0], idx, [n+1])).astype(int) # we add beginning and end of final string
    n_switches = len(idx)
    # generate array of target arguments to which the copying process will switch
    ts = np.zeros(n_switches, dtype=int)
    for j in range(1, n_switches): # process always starts at first argument (i.e. 0)
        possibilities = [i for i in range(k) if i != ts[j-1]]
        ts[j] = choice(possibilities)
    output = ''.join(args[ts[j]][idx[j]:idx[j+1]] for j in range(n_switches - 1))
    return output

xorfunc = modularsumfunc_factory(2)
sum3func = modularsumfunc_factory(3)
sum5func = modularsumfunc_factory(5)