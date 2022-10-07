import functools
from math import prod
from typing import Dict, Iterable, Optional, Union

from numpy.random import choice, random
from scipy.stats import bernoulli

# type alias
Alphabet = Iterable[str]

def random_string(length, p=0.5, alphabet=None):
    alphabet = ['0', '1'] if alphabet is None else alphabet
    if alphabet == ['0', '1']:
        # p must be float, is interpreted as Pr(X=1)
        return ''.join(str(bit) for bit in bernoulli.rvs(p, size=length))
    else:
        # p must be iterable of floats, the probabilities associated with each
        # entry in alphabet
        return ''.join(str(bit) for bit in choice(alphabet, p=p, size=length))

def xor(*args):
    """
    Giving the cycling behaviour for handling strings of different lengths,
    does this function define an associative operation????
    """
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        # identify longest and shortest string
        l = max(args, key=len)
        i_l = list(args).index(l)
        s = list(args)[(i_l + 1) % 2]
        # calculate bitwise xor, cycling as necessary over shortest
        result = ''
        for i in range(len(l)):
            bit = '0' if l[i]==s[i % len(s)] else '1'
            result = result + bit
        return result
    if len(args) >= 2:
        return xor(args[0], xor(*args[1:]))
xor.arity = -1

def neg(s):
    return xor(s, '1')
def rev(s):
    return s[::-1]
def zero_evens(s):
    out = []
    for i in range(len(s)):
        out.append(0 if i%2==0 else s[i])
    return out

def OR(x,y):
    return ''.join(['1' if xi=='1' or yi=='1' else '0' for (xi, yi) in zip(x,y)])

def left_shift(string, n):
    return string[n:] + string[:n]

#### Decorators

def string_function(func):
    """
    - Builds "symbol per symbol" function out of an "inner" function 
    Resulting function uses the i-th symbols taken from all arguments as arguments
    for `func`, and returns the result of applying `func` in this way to all
    positions of the arguments.
    - `func` must have non-fixed arity, take symbols and return a symbol
    - We require input strings to be of equal length
    """
    @functools.wraps(func)
    def wrapper(*strings):
        l = len(strings[0])
        assert all(len(string) == l for string in strings), "All inputs must have equal length"
        output = ''
        for i in range(l):
            symbols = [string[i] for string in strings]
            output += func(*symbols)
        return output
    wrapper.arity = -1 # meaning arity is not fixed
    return wrapper

def symbol_function(func):
    "Makes integer function into symbol function"
    @functools.wraps(func)
    def wrapper(*symbols):
        integers = [int(s) for s in symbols]
        value = func(*integers)
        return str(value)
    return wrapper

def noise(noise_probability : float,
          noise_distribution : Union[Alphabet, Dict[str, float]]):
    """
    Adds noise to the output of a function. With probability `noise_probability`,
    the original output of the function will be replaced for some value out of
    the keys of `noise_distribution`. Which key depends on their probabilities
    as recorded by the values they hold in `noise_distribution`. If instead
    `noise_distribution` is just a list of values (strings), uniform distribution
    is assumed.

    If the value that would have been returned by the original function is one
    of the values in `noise_distribution`, then this value is not considered
    when producing a value to replace it. The remaining probabilities are then
    normalized to 1 in order to pick from the remaining possible values.

    Use example:
        @noise(0.5, {'hola': 0.8, 'chau': 0.2})
        def f(x): return 2*x
        results = [f(3) for _ in range(1000)]
        from collections import Counter
        c = Counter(results)
        print(c)
    """
    def inner_decorator(func):
        if isinstance(noise_distribution, dict):
            possible_noise_values = list(noise_distribution.keys())
            noise_value_probabilities = list(noise_distribution.values())
        else:
            possible_noise_values = list(noise_distribution)
            n = len(possible_noise_values)
            noise_value_probabilities = [1/n for _ in range(n)]
        @functools.wraps(func)
        def wrapper(*args):
            noiseless_value = func(*args)
            if random() <= noise_probability: # i.e. with probability `noise_probability`
                vals, probs = exclude_noiseless_value(possible_noise_values, noise_value_probabilities, noiseless_value)
                return choice(vals, p=probs)
            else:
                return noiseless_value
        return wrapper
    return inner_decorator

def stringnoise(noise_probability : float,
                alphabet : Optional[Alphabet] = None,
                noise_distribution : Optional[Dict[str, float]] = None):
    """
    Adds noise to a given string. With independent probability `noise_probability`,
    each symbol will be replaced for some value.
    If `alphabet` argument is passed, then a uniform distribution is assumed over those
    symbols.
    Else, `noise_distribution` must be passed. The possible noise values will
    come from the keys of `noise_distribution` and their probabilities
    are recorded by the values they hold in `noise_distribution`.

    If the original symbol, before replacement, is one of the possible values then
    this value is not considered when producing a value to replace it.
    The remaining probabilities are then normalized to 1 in order to pick from
    the remaining possible values.
    """
    def inner_decorator(func):
        if alphabet is not None:
            possible_noise_values = alphabet
            n = len(possible_noise_values)
            noise_value_probabilities = [1/n for _ in range(n)]
        elif noise_distribution is not None:
            possible_noise_values = list(noise_distribution.keys())
            noise_value_probabilities = list(noise_distribution.values())
        else:
            raise ValueError("At least one of `alphabet` and `noise_distribution` must be passed")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            noiseless_output = func(*args, **kwargs)
            if not isinstance(noiseless_output, str):
                raise ValueError("Decorator must be applied on a string-valued function")
            noisy_output = ''
            for symbol in noiseless_output:
                if random() <= noise_probability: # i.e. with probability `noise_probability`
                    vals, probs = exclude_noiseless_value(possible_noise_values, noise_value_probabilities, symbol)
                    noisy_output += choice(vals, p=probs)
                else:
                    noisy_output += symbol
            return noisy_output
        return wrapper
    return inner_decorator


#### Auxiliary functions for decorators

def exclude_noiseless_value(possible_noise_values, noise_value_probabilities, noiseless_value):
    try:
        i = possible_noise_values.index(noiseless_value)
        possible_noise_values = possible_noise_values[:i] + possible_noise_values[i+1:]
        noise_value_probabilities = noise_value_probabilities[:i] + noise_value_probabilities[i+1:]
        z = sum(noise_value_probabilities)
        noise_value_probabilities = [x/z for x in noise_value_probabilities]
    except ValueError:
        pass
    return possible_noise_values, noise_value_probabilities

#### Modular arithmetic
# Unused
def build_modular_addition_function(n):
    assert n<=10
    @string_function
    @symbol_function
    def modular_addition(*xs): return sum(xs) % n
    return modular_addition

def build_modular_multiplication_function(n):
    assert n<=10
    @string_function
    @symbol_function
    def modular_multiplication(*xs): return prod(xs) % n
    return modular_multiplication

# innecesario, pero fue divertido pensarlo
# def make_into_modular(n):
#     assert n<=10 # arithmetic mod n for n>10 not supported
#     def inner_decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args):
#             return func(*args) % n
#         if hasattr(func, 'arity'):
#             wrapper.arity = func.arity
#         return wrapper
#     return inner_decorator
