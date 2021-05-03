from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np
from scipy.stats import chi2_contingency, kstest

from abstract_classes import IndependenceTest
from fisher_exact_fromR import \
    fisher_test  # for general m x n contingenct tables. Requires R
from utils import InitializationError

# type alias
Alphabet = Iterable[str]
class KSContingencyTest(IndependenceTest):
    """
    """
    def __init__(self,
                 alphabet: Optional[Alphabet] = None,
                 kind='chi2',
                 alpha: Optional[float] = None,
                 replicate: int = 2000,
                 verbose=False,
                 n_subsamples: int = 1) -> None:
        self.alphabet = alphabet
        self.kind = kind
        self.alpha = alpha
        self.verbose = verbose
        self.n_subsamples = n_subsamples
        # Parameters for exact test
        self.replicate = replicate
        # Attributes initialized during self.run()
        self.n_rows = None
        self.data_arr = None
        self.pvals = None
        self.aggregate_pval = None

    def run(self,
            x:str,
            y:str,
            s:Iterable[str],
            data:dict,
            alpha:Optional[float] = None,
            alphabet:Optional[Alphabet] = None,
            verbose:str=None
            ) -> KSContingencyTest.TestResults:
        # Update attributes, but only if given
        self.verbose = verbose if verbose is not None else self.verbose
        self.alpha = alpha if alpha is not None else self.alpha
        self.alphabet = alphabet if alphabet is not None else self.alphabet
        if self.alpha is None:
            raise InitializationError("Alpha value not set")
        if self.alphabet is None:
            raise InitializationError("Alphabet is not set")
        # Build data ndarray from strings
        self.n_rows = len(data[x])
        self.data_arr = KSContingencyTest.build_datarray(x, y, s, data)
        # z is the name of the conditioning variable
        # (which consists of multiple symbols, one from each conditioning string)
        self.pvals = [] # p-value of test conditioning over each z value
        visited_z_values = []
        for i in range(self.n_rows):
            z_value = self.data_arr[i, 2:]
            if list(z_value) in visited_z_values:
                continue
            # Keep rows that match z_value
            mask = np.all(self.data_arr[:, 2:] == z_value, axis=1)
            conditioned_array = self.data_arr[mask, :2]
            if self.verbose:
                print('\tz* = {}'.format(z_value))
            pval = self.do_test_on_array(conditioned_array)
            self.pvals.append(pval)
            if self.verbose:
                print('\t\tn_rows={}\n\t\tp-val: {:.4g}'.format(conditioned_array.shape[0], pval))
            visited_z_values.append(list(z_value)) # mark this z_row as already seen
        # Handling of nans in `self.pvals`
        proper_pvals = np.array(self.pvals)[np.isfinite(self.pvals)].tolist()
        if len(proper_pvals) == 0:
            print("WARNING: Test failed. No pvals could be calculated for any value \
                of the conditioning variables. Returning nan.")
            self.aggregate_pval = np.nan
        elif len(proper_pvals) == 1:
            self.aggregate_pval = proper_pvals[0]
        else: # aggregate pvalues with KS test
            # IMPORTANTE: Solo usamos test KS para agregar p-valores si hubo más de un p-valor calculado
            self.aggregate_pval = KSContingencyTest.kstest_on_pvals(proper_pvals)
        if self.verbose:
            print('aggr. p-val: {:.4g}'.format(self.aggregate_pval))
        return KSContingencyTest.TestResults(self.pvals, self.aggregate_pval, self.alpha, self.verbose)

    @staticmethod
    def kstest_on_pvals(pvals: Iterable[float]) -> float:
        _, p = kstest(pvals, "uniform", mode='exact')
        return p

    @staticmethod    
    def build_datarray(x:str, y:str, s:Iterable[str], data: dict) -> np.array:
        x = data[x]
        y = data[y]
        s = [data[n] for n in s]
        # Build data ndarray from strings
        n_rows = len(x)
        data_arr = np.zeros((n_rows, 2+len(s)), dtype=str)
        for i in range(n_rows):
            data_arr[i, 0] = x[i]
            data_arr[i, 1] = y[i]
            for j in range(len(s)):
                data_arr[i, j+2] = s[j][i]
        return data_arr

    def build_contingency_table(self, array: np.array) -> np.array:
        # composite W variable has pairs of symbols (= the rows of input array) as values
        # Calculate observed frequencies of w values
        w_counts = Counter(map(tuple, array))
        n = len(self.alphabet)
        contingency_table = np.zeros((n,n), dtype=int)
        for i, s1 in enumerate(self.alphabet):
            for j, s2 in enumerate(self.alphabet):
                contingency_table[i, j] = w_counts[(s1, s2)]
        return contingency_table

    def do_test_on_array(self, array: np.array) -> float:
        test = {'exact': self._do_fisher,
                'chi2': self._do_chi2}
        p = test[self.kind](array)
        return p

    def _do_fisher(self, array: np.array) -> float:
        total_table = self.build_contingency_table(array)
        if self.verbose:
            print(total_table)
        sr = total_table.sum(axis=1)
        sc = total_table.sum(axis=0)
        c = total_table[sr > 0, :][:, sc > 0]
        nr, nc = c.shape
        if nr > 1 and nc > 1:
            # Split rows into n_subsamples as evenly as possible
            subarrays = np.array_split(array, self.n_subsamples, axis=0)
            pvals = []
            # Apply contingency test on each subsample separately, save pvals
            for arr in subarrays:
                table = self.build_contingency_table(arr)
                p = fisher_test(table, simulate_pval=True, replicate=self.replicate)
                pvals.append(p)
            # Combine p-vals through KS test UNLESS number of subsamples is 1
            if self.n_subsamples != 1:
                p = KSContingencyTest.kstest_on_pvals(pvals)
            else:
                p = pvals[0]
        else: # Polémico
            # p = np.nan
            print('WARNING: Fisher test cannot be applied, setting p=1')
            p = 1
        return p

    def _do_chi2(self, array: np.array) -> float:
        total_table = self.build_contingency_table(array)
        if self.verbose:
            print(total_table)
        # Split rows into n_subsamples as evenly as possible
        subarrays = np.array_split(array, self.n_subsamples, axis=0)
        pvals = []
        # Apply contingency test on each subsample separately, save pvals
        for arr in subarrays:
            try:
                table = self.build_contingency_table(arr)
                p = chi2_contingency(table)[1]
                pvals.append(p)
            except ValueError:
                print("WARNING: The internally computed table of expected frequencies \
                       has a zero element. p value cannot be calculated")
        # Combine p-vals through KS test UNLESS number of subsamples is 1
        if len(pvals) == 0:
            print("WARNING: No p value could be calculated at all. Returning nan")
            p = np.nan
        elif len(pvals) == 1:
            p = pvals[0]
        else:
            p = KSContingencyTest.kstest_on_pvals(pvals)
        return p

    class TestResults():
        def __init__(self, pvals:Iterable[float], aggregate_pval:float, alpha:float, verbose:bool) -> None:
            self.pvals = pvals
            self.aggregate_pval = aggregate_pval
            self.alpha = alpha
            self.verbose = verbose
        def independent(self, alternate_alpha:Optional[float]=None, verbose=None):
            alpha = self.alpha if alternate_alpha is None else alternate_alpha
            verbose = verbose if verbose is not None else self.verbose
            if np.isnan(self.aggregate_pval):
                raise ValueError("Test failed. No pvals could be calculated for any value \
                of the conditioning variables.")
            if self.aggregate_pval < alpha:
                if verbose:
                    print("Dependientes (Rechazo H0): p-val agregado =", self.aggregate_pval)
                return False
            else:
                if verbose:
                    print("Independientes (No rechazo H0), p-val agregado =", self.aggregate_pval)
                return True
            ### OLD
            # pvals = np.array(self.pvals)
            # for p in pvals:
            #     if p <= alpha: # notar que si p es nan, esto da falso como queremos
            #         if self.verbose:
            #             print("Dependientes (Rechazo H0): un p-val para z fijo =", p)
            #         return False
            # if self.aggregate_pval < alpha: # si es nan, no se ejecuta
            #     if self.verbose:
            #         print("Dependientes (Rechazo H0): p-val agregado = ", self.aggregate_pval)
            #     return False
            # else:
            #     if self.verbose:
            #         print("Independientes (No rechazo H0), min p-val =", min(*pvals, self.aggregate_pval))
            #         # si todo es nan, el mínimo es nan (para la función min, no para <)
            #     return True
