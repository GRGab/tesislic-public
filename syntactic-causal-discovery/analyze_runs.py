import dill

from collections import Counter, defaultdict


def load_multirun(filename):
    with open(f"analysis/{filename}", 'rb') as f:
        mr = dill.load(f)
    return mr

class MultiRunAnalyzer():
    def __init__(self, multirun):
        self.multirun = multirun
        self._is_single_test = self.multirun._is_single_test

    def who_blocks_this_pair(self,
                             n1,
                             n2,
                             test_number=0,
                             filter_=None):
        if self._is_single_test: # ignore test_number
            test_results = self.multirun.test_results
        else:
            test_results = self.multirun.test_results[test_number]

        independence_counts = Counter() # how many times X, Y are deemed independent given some Z
        for i, iteration_results in enumerate(test_results):
            pair = (n1, n2) if (n1, n2) in iteration_results else (n2, n1)
            X, Y = pair
            if filter_ is None or filter_(iteration_results[(X, Y)]):
                for Z in iteration_results[(X, Y)]:
                    res = iteration_results[(X, Y)][Z]
                    if res.independent(verbose=False):
                        independence_counts[tuple(Z)] += 1
        return independence_counts

    def test_values_for_this_pair(self,
                                  n1,
                                  n2,
                                  test_number=0,
                                  filter_ = None):
        if self._is_single_test: # ignore test_number
            test_results = self.multirun.test_results
        else:
            test_results = self.multirun.test_results[test_number]
        test_values = defaultdict(list)
        for i, iteration_results in enumerate(test_results):
            pair = (n1, n2) if (n1, n2) in iteration_results else (n2, n1)
            X, Y = pair
            if filter_ is None or filter_(iteration_results[(X, Y)]):
                for Z in iteration_results[(X, Y)]:
                    res = iteration_results[(X, Y)][Z]
                    if hasattr(res, 'pvals'): # meaning, if test is KSContingencyTest
                        test_values[tuple(Z)].append(res.aggregate_pval)
                    elif hasattr(res, "CMI"): # meaning it's a CMITest
                        test_values[tuple(Z)].append(res.CMI)
        return test_values

def only_detected(x):
    """
    x : Iteration results for a given pair of nodes
    Filter for `analyze_test_results` that selects only iterations for which
    an edge has been detected between the two variables under consideration
    """
    return all(not res.independent(verbose=False) for res in x.values())

def only_undetected(x):
    """
    x : Iteration results for a given pair of nodes
    Filter for `analyze_test_results` that selects only iterations for which
    an edge has NOT been detected between the two variables under consideration
    """
    return any(res.independent(verbose=False) for res in x.values())