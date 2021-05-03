import itertools
from collections import Counter
from typing import List, Literal, Union

import dill
import networkx as nx

from abstract_classes import IndependenceTest
from functional_models import FunctionalModel
from ic_algorithm import IC
from utils import timestamp
from visualization import visualize_graph, visualize_multirun


class ICMultiRun():
    """
    test_results = [
                        [
                            {
                                (n1, n2) : {set : indep_test_results(n1, n2, set)}
                            for (n1, n2) in pairs_of_nodes}
                        for ic_object in iterations]
                    for indep_test in independence_tests]
    """
    def __init__(
            self,
            model : FunctionalModel,
            independence_test : Union[IndependenceTest, List[IndependenceTest]],
            n_repetitions : int = 10, # per test
            visualize : bool = True,
            verbose : bool = False,
            save_everything : bool = False,
            save_test_results : bool = True,
            send_telegram : bool = False,
            save_to_disk : bool = False,
            savepath : str = None
        ) -> None:
        self.model = model
        self.independence_test = independence_test
        self.n_repetitions = n_repetitions
        self.visualize = visualize
        self.verbose = verbose
        self.save_everything = save_everything
        self.save_test_results = save_test_results
        self.send_telegram = send_telegram
        self.save_to_disk = save_to_disk
        # Handle Union type for `self.independence_test`
        self._is_single_test = isinstance(independence_test, IndependenceTest)
        # Attributes to be initialized by `_handle_independence_tests`
        self.edge_counts = None
        # Only used if save_everything:
        self.ic_instances = None
        self.data_instances = None
        self.timestamp = None
        # Only used if save_test_results:
        self.test_results = None
        self.inferred_edges = None
        # Only used if save_to_disk:
        self.savepath = savepath

    def run(self):
        self._handle_independence_tests('open') # handle union type for independence test, initialize some attributes
        for k in range(self.n_repetitions):
            if self.verbose and self.n_repetitions > 1:
                print("\n")
                print('#######################', f'\nIteration #{k}', '\n#######################')
            # Generate model state
            self.model.generate_inputs()
            data = self.model.state
            # Run IC algorithm
            for i, independence_test in enumerate(self.independence_test):
                if self.verbose and not self._is_single_test:
                    print('=======================', f'\nUsing test #{i}', '\n=======================')
                ic = IC(independence_test, verbose=self.verbose, save_test_results=self.save_test_results)
                ic.run(data, visualize=False)
                edges = ic.inferred_edges
                print(f'{k:{len(str(self.n_repetitions-1))}d}|{i:{len(str(len(self.independence_test)-1))}d}', "Inferred edges:", edges)
                for edge in edges:
                    self.edge_counts[i][edge] += 1
                if self.save_test_results:
                    self.test_results[i].append(ic.test_results)
                    self.inferred_edges[i].append(edges)
                if self.save_everything:
                    self.ic_instances[i].append(ic)
                    self.data_instances[i].append(data)
        # only show figures if self.visualize is True
        only_save = not self.visualize
        # Save pngs only if savepath is given (and save_to_disk is True)
        if self.save_to_disk and self.savepath is not None:
            visualize_multirun(self, savepath=self.savepath, only_save=only_save)
        else:
            visualize_multirun(self, only_save=only_save)
            self.model.visualize()
        self._handle_independence_tests('close') # handle union type for independence test
        if self.save_to_disk:
            if self.savepath is None:
                self.timestamp = timestamp()
                with open(f"analysis/multiruns/{self.timestamp}.pkl", 'wb') as f:
                    dill.dump(self, f)
            else:
                with open(f"{self.savepath}.pkl", 'wb') as f:
                    dill.dump(self, f)
        if self.send_telegram:
            try: # this will work in my local repository
                import telegram_bot
                telegram_bot.telegram_bot_sendtext("Done!")
            except ModuleNotFoundError:
                pass

    def _handle_independence_tests(self, action : Literal['open', 'close']):
        """
        The idea: "internally", `self.independence_test` is always treated as a
        list. So if it is passed as a single IndependenceTest object, it will
        be converted to a singleton list before processing and reconverted to
        a single object after processing.

        Results of the MultiRun (ic_instances, data_instances, test_results,
        inferred_edges) will be given as lists where each element corresponds
        to a single independence test. After processing, if there was only one
        IndependenceTest object, the top level of the list structure is removed."""
        if action == 'open':
            if self._is_single_test:
                # Wrap in a list
                self.independence_test = [self.independence_test]
            # Initialize output attributes with one "slot" per independence test
            # to be performed
            l = len(self.independence_test)
            self.edge_counts = [Counter() for _ in range(l)]
            if self.save_everything:
                self.ic_instances = [[] for _ in range(l)]
                self.data_instances = [[] for _ in range(l)]
            if self.save_test_results:
                self.test_results = [[] for _ in range(l)]
                self.inferred_edges = [[] for _ in range(l)]
        if action == 'close':
            if self._is_single_test:
                # Extract objects from singleton lists
                self.independence_test = self.independence_test[0]
                self.edge_counts = self.edge_counts[0]
                if self.save_everything:
                    self.ic_instances = self.ic_instances[0]
                    self.data_instances = self.data_instances[0]
                if self.save_test_results:
                    self.test_results = self.test_results[0]
                    self.inferred_edges = self.inferred_edges[0]