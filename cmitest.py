from __future__ import annotations

from typing import Iterable, Optional

from abstract_classes import IndependenceTest
from information import InformationEstimator
from utils import InitializationError


class CompressionTest(IndependenceTest):
    def __init__(self,
                 criterion='fixed_threshold',
                 compressor=None,
                 pairing=None,
                 threshold=None,
                 verbose=False,
                 verbose_ie=False):
        self.criterion = criterion
        self.verbose = verbose
        # Output of InformationEstimator calculations will be printed only if
        # `verbose==True` is passed as kwarg to the CompressionTest constructor
        # explicitly
        self.ie = InformationEstimator(compressor=compressor, pairing=pairing,
                                       verbose=verbose_ie)
        self.threshold = threshold
        # Attributes initialized during self.run()
        # none for now
    def run(self,
            x:str,
            y:str,
            s:Iterable[str],
            data:dict,
            verbose:bool=None,
            threshold:Optional[float]=None
            ) -> CompressionTest.TestResults:
        # Update attributes, but only if given
        self.verbose = verbose if verbose is not None else self.verbose
        self.threshold = threshold if threshold is not None else self.threshold
        if self.threshold is None:
            raise InitializationError("Threshold value not set")
        x = data[x]
        y = data[y]
        s = [data[n] for n in s]
        MI = self.ie.MI(x, y)
        CMI = self.ie.CMI(x, y, s)
        fval = self.ie.f(x, y, s)
        lengths = {'x': len(x), 'y': len(y), 's': [len(n) for n in s]}
        return CompressionTest.TestResults(MI, CMI, fval, lengths, self.criterion,
                                           self.threshold, self.verbose)

    class TestResults():
        def __init__(self, MI, CMI, fval, lengths, criterion, threshold, verbose) -> None:
            self.MI = MI
            self.CMI = CMI
            self.fval = fval
            self.lengths = lengths
            self.criterion = criterion
            self.threshold = threshold
            self.verbose = verbose

        def independent(self, verbose=None):
            self.verbose = verbose if verbose is not None else self.verbose
            try:
                verdict_function = {
                    "f_fixed_threshold" : self._independent_f_fixed_threshold,
                    'fixed_threshold'   : self._independent_fixed_threshold,
                    'length_fraction'   : self._independent_length_fraction,
                    'MI_fraction'       : self._independent_MI_fraction,
                    'mixed_fraction'    : self._independent_mixed_fraction
                }[self.criterion]
            except KeyError as e:
                raise ValueError("Invalid criterion for test") from e
            return verdict_function()

        def _independent_f_fixed_threshold(self):
            # Notar que I(x : y | zs) = 0 iff f(x, y, zs) = 1, así que se
            # invierten los resultados respecto de usar CMI
            if self.fval <= self.threshold:
                if self.verbose:
                    print(f"Dependientes: {self.fval:.3g} <= threshold = {self.threshold:.3g}")
                return False
            else:
                if self.verbose:
                    print(f"Independientes: {self.fval:.3g} > threshold = {self.threshold:.3g}")
                return True
            
        def _independent_fixed_threshold(self):
            if self.CMI <= self.threshold:
                if self.verbose:
                    print(f"Independientes (Acepto H0): {self.CMI:.3g} <= threshold = {self.threshold:.3g}")
                return True
            else:
                if self.verbose:
                    print(f"Dependientes (Rechazo H0): {self.CMI:.3g} > threshold = {self.threshold:.3g}")
                return False

        def _independent_length_fraction(self):
            return self.CMI <= self.threshold * min(self.lengths['x'],
                                                    self.lengths['y'])

        def _independent_MI_fraction(self):
            if self.MI <= 0:
                return True
            elif self.CMI < self.threshold * self.MI:
                return True
            else:
                return False

        def _independent_mixed_fraction(self):
            """OJO: en este caso threshold debe ser un par de fracciones!"""
            if self.MI <= self.threshold[0] * min(self.lengths['x'], self.lengths['y']):
                # print(self.MI)
                return True
            if self.CMI < self.threshold[1] * self.MI:
                return True
            else:
                return False
    
if __name__ == "__main__":
    def load_blackboxed_data(path):
        data = {}
        with open(path, 'r') as f:
            for line in f:
                node, value = line.split()
                data[node] = value
        return data

    the_data = load_blackboxed_data('temp/blackboxed_output')
    # Set threshold
    the_threshold = 0 #-40
    cmi = CompressionTest()
    res = cmi.run('0', '3', ['2'], the_data, the_threshold)
    print(res.independent())
