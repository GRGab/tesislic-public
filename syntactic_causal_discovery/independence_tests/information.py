from itertools import chain, zip_longest
from typing import Literal
from utils import timestamp

import numpy as np

from augmented_lz_complexity import lz_information
from independence_tests.compression import compression_length
from independence_tests.I_complexity import calculate_I_complexity


#### Pairing functions (different ways of combining strings for MI and CI calculation)
def concatenation(*args):
    return ''.join(args)
def interleaving(*args):
    return "".join(filter(None, chain.from_iterable(zip_longest(*args))))

# Global dict
pairing_functions = {"concat": concatenation, "interleave": interleaving}

class InformationEstimator():
    def __init__(self,
                 compressor: Literal['gzip', 'lz', 'I'] = None,
                 pairing: Literal['concat', 'interleave'] = None,
                 verbose: bool = False,
                 **kwargs):
        # Default values
        compressor = 'gzip' if compressor is None else compressor # actually a bad attribute name
        pairing = "concat" if pairing is None else pairing
        self.compressor = compressor
        self.tuple = pairing_functions[pairing]
        self.verbose = verbose
        self.kwargs = kwargs
        self.tempfilename = f'temp/temp_{timestamp()}'

    def I(self, *args):
        """Complexity/Information metric"""
        if self.compressor == 'gzip':
            return compression_length(self.tuple(*args), compressor='gzip',
                                      tempfile=self.tempfilename,
                                      **self.kwargs)
        elif self.compressor == 'lz':
            return lz_information(self.tuple(*args), **self.kwargs)
        elif self.compressor == 'I':
            # non-binary alphabet functionality available, though not exposed
            return calculate_I_complexity(self.tuple(*args),
                                          tempfile=self.tempfilename)
        else:
            raise ValueError("Invalid compressor name")
    def CI(self, xs, ys):
        """Estimator for conditional algorithmic information
        Returns complexity of an arbitrary number of strings in xs conditioned on
        an arbitrary number of strings in ys, i.e. C(x_1,...,x_n|y_1,...,y_m)
        
        When xs==(x,) and ys==(,), behaviour equals InformationEstimator.I()
        
        If repeated strings are given, behaviour will not be that of conditioning
        with respect to a _set_ of strings."""
        I = self.I
        return I(*xs, *ys) - I(*ys)
        # return self.I(x, *y) - self.I(*y)
    def MI(self, x, y):
        """Estimator for mutual algorithmic information"""
        I = self.I
        I_x = I(x)
        I_y = I(y)
        I_xy= I(x, y)
        MI_xy = I_x + I_y - I_xy
        if self.verbose:
            print(f"I(x) = {I_x}\nI(y) = {I_y}\nI(x, y) = {I_xy}\nMI(x, y) = {MI_xy}")
        return MI_xy
    def CMI(self, x, y, zs):
        """Estimator for conditional mutual algorithmic information
        Return mutual information between x and y conditioned on a list of
        strings.
        
        Note: InformationEstimator.CI() receives unpacked arguments, while
        InformationEstimator.CMI() must always receive 3 arguments, the
        third one being unpacked later

        WARNING: If zs has repeated elements, behaviour will not be that of
        conditioning with respect to a _set_ of strings."""
        if len(zs) == 0:
            return self.MI(x, y) # Avoid dealing with empty stuff
        I = self.I
        I_xz = I(x, *zs)
        I_yz = I(y, *zs)
        I_xyz = I(x, y, *zs)
        I_z = I(*zs)
        CMI_xyz = I_xz + I_yz - I_xyz - I_z
        if self.verbose:
            print(f"I(x, *zs) = {I_xz}\nI(y, *zs) = {I_yz}\nI(x, y, *zs) = {I_xyz}\nI(*zs) = {I_z}\nMI(x, y|zs) = {CMI_xyz}")
        return CMI_xyz
    def NMI(self, x, y):
        """Normalized mutual algorithmic information (MI(x,y)/I(x,y))"""
        return self.MI(x, y) / self.I(x, y)
    def NCD(self, x, y):
        """Normalized Compression Distance (as in Ly & Vitányi, Section 8.4.2)"""
        ix, iy = self.I(x), self.I(y)
        return (self.I(x, y) - min(ix, iy)) / max(ix, iy)
    def f(self, x, y, zs):
        """Métrica de prueba definida en enero 2021"""
        I = self.I
        I_xz = I(x, *zs)
        I_yz = I(y, *zs)
        I_xyz = I(x, y, *zs)
        I_z = I(*zs) if len(zs) != 0 else 0
        result = 2 * ((I_xyz - I_z) / (I_xz + I_yz - 2*I_z)) - 1
        if self.verbose:
            print(f"I(x, *zs) = {I_xz})\nI(y, *zs) = {I_yz}\nI(x, y, *zs) = {I_xyz}\nI(*zs) = {I_z}\nf(x, y|zs) = {result}")
        return result

class LabelledInformationEstimator(InformationEstimator):
    def __init__(self, datadict, compressor=None, pairing=None):
        self.datadict = datadict
        super().__init__(compressor=compressor, pairing=pairing)

    def I(self, *args):
        args = [self.datadict[arg] for arg in args]
        return super().I(*args)

#### quick testing of mutual information between independent random strings
def mi_random_strings(information_metric,
                      pairing_func,
                      str_len = 20000, p=0.4, alphabet=['0', '1'], n_iter=100):
    mis = np.zeros(n_iter)
    for i in range(n_iter):
        xs = random_string(str_len, p=p, alphabet=alphabet)
        ys = random_string(str_len, p=p, alphabet=alphabet)
        mi = InformationEstimator(information_metric, pairing_func).MI(xs, ys)
        mis[i] = mi
    return np.mean(mis), np.std(mis)
## for default vals:
## 'I', 'concat' -> 271 +/- 2
## 'I', 'interleave' -> 270 +/- 4
## 'gzip', 'concat' -> 453 +/- 25
## 'gzip', 'interleave' -> 447 +/- 38

#### Legacy, remove when/if compression_tests.py is refactored)

def cond_information(x, y, compressor='gzip', pairing="concat"):
    """Estimation of the conditional algorithmic information of x given a minimal program for y"""
    K = lambda s: compression_length(s, compressor=compressor)
    pair = pairing_functions[pairing]
    return K(pair(x, y)) - K(y)

def mutual_information(x, y, compressor='gzip', pairing="concat"):
    pair = pairing_functions[pairing]
    K1 = compression_length(x, compressor=compressor)
    K2 = compression_length(y, compressor=compressor)
    K3 = compression_length(pair(x, y), compressor=compressor)
    return K1 + K2 - K3

def cond_mutual_information(x, y, z, compressor='gzip', pairing="concat"):
    """Estimation of the conditional mutual algorithmic information between x
    and y given z"""
    K = lambda x, y: cond_information(x, y, compressor=compressor)
    pair = pairing_functions[pairing]
    K1 = K(x, z)
    K2 = K(y, z)
    K3 = K((pair(x, y)), z)
    return K1 + K2 - K3

#### Conditional and mutual information matrices

def cond_information_matrix(strings, compressor='gzip', pairing="concat"):
    N = len(strings)
    output_matrix = np.zeros((N, N), dtype=int)
    K = lambda x, y: cond_information(x, y, compressor=compressor, pairing=pairing)
    for i in range(N):
        for j in range(N):
            output_matrix[i, j] = K(strings[i], strings[j])
    return output_matrix

def mutual_info_matrix(strings, compressor='gzip', pairing="concat"):
    N = len(strings)
    output_matrix = np.zeros((N, N), dtype=int)
    I = lambda x, y: mutual_information(x, y, compressor=compressor, pairing=pairing)
    for i in range(N):
        for j in range(N):
            output_matrix[i, j] = I(strings[i], strings[j])
    return output_matrix

if __name__ == "__main__":
    from pprint import pprint
    from string_transformations import random_string, neg

    ie = InformationEstimator(compressor='I', pairing='concat', verbose=False)
    s = random_string(20000, p=0.5)
    t = random_string(20000, p=0.5)
    w = neg(s)
    out = (ie.I(s), ie.I(t), ie.MI(s, t), ie.MI(s, w))
    pprint(out)