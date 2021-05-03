from collections import defaultdict
from itertools import chain, combinations
from typing import Optional

import networkx as nx
import numpy as np

from independence_tests.cmitest import CompressionTest
from functional_models import load_functional_model
from utils import InitializationError
from visualization import visualize_graph


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def load_blackboxed_data(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            node, value = line.split()
            data[node] = value
    return data

class IC():
    def __init__(self, independence_test, labeldict=None, verbose=True, save_test_results=True):
        self.independence_test = independence_test
        self.labels = labeldict
        self.verbose = verbose
        self.save_test_results = save_test_results
        # Results of the algorithm
        self._undirected_graph = None
        self._directed_graph = None
        self.blocking_sets = {}
        self.test_results = defaultdict(dict) # for step one
        self.inferred_edges = [] # for step one

    def run(self, data, visualize=True, verbose=None, save_test_results=None):
        # Update attributes, but only if given
        self.verbose = verbose if verbose is not None else self.verbose
        self.save_test_results = save_test_results if save_test_results is not None else self.save_test_results
        # Perform IC algorithm
        nodes = list(data.keys())
        self._undirected_graph = nx.Graph()
        self._undirected_graph.add_nodes_from(nodes)
        self._directed_graph = nx.DiGraph()
        self._directed_graph.add_nodes_from(nodes)
        if self.labels is None:
            self.labels = {n : n for n in nodes} # identity dict
        self.step_one(data, nodes)
        if visualize:
            visualize_graph(self._undirected_graph, labeldict=self.labels, title="Step 1")
        # self.step_two(nodes)
        # if True: # if visualize:
        #     visualize_graph(self._directed_graph, labeldict=self.labels, title="Step 2")
        # Step three not implemented
    
    def step_one(self, data, nodes):
        if self.save_test_results and len(self.test_results) != 0:
            print('WARNING: keeping test_results from previous run. This may lead to confusion')
        for node_pair in combinations(nodes, 2):
            n1, n2 = node_pair
            if self.verbose:
                print('-----------------------', '\nMirando par', node_pair, '\n-----------------------')
            for set_of_nodes in powerset([n for n in nodes if n not in [n1, n2]]):
                if self.verbose:
                    print("Z = ", set_of_nodes)
                # Run independence test
                test_results = self.independence_test.run(n1, n2, set_of_nodes, data, verbose=self.verbose)
                they_are_independent = test_results.independent()
                if self.save_test_results:
                    self.test_results[(n1, n2)][tuple(set_of_nodes)] = test_results
                if they_are_independent:
                    self.blocking_sets[node_pair] = set_of_nodes
                    if self.verbose:
                        print('SÍ hay blocking set para ({}, {}): {}.'.format(self.labels[n1], self.labels[n2], [self.labels[n] for n in set_of_nodes]))
                    break
            else: # if no blocking set is found, create edge
                self.blocking_sets[node_pair] = None
                self._undirected_graph.add_edge(n1, n2)
                self.inferred_edges.append(node_pair)
                if self.verbose:
                    print('NO hay blocking sets para ({}, {}). Se agrega arista'.format(self.labels[n1], self.labels[n2]))

    def step_two(self, nodes):
        for (a, b) in combinations(nodes, 2):
            s_ab = self.blocking_sets[(a, b)]
            if s_ab is not None: # n1, n2 are non-adjacent in undirected graph
                for c in nodes:
                    if self._undirected_graph.has_edge(a, c) and \
                    self._undirected_graph.has_edge(b, c) and \
                    c not in s_ab:
                        if self.labels is not None:
                            print("Agregando v-estructura {}->{}<-{}".format(self.labels[a], self.labels[c], self.labels[b]))
                        self._directed_graph.add_edges_from([(a, c), (b, c)])

    def step_three(self, data, nodes):
        raise NotImplementedError
        # def are_adjacent(n1, n2):
        #     return undirected_graph.has_edge(n1, n2) or directed_graph.has_edge(n1, n2) or directed_graph.has_edge(n2, n1)
        # for (b, c) in undirected_graph.edges:
        #     # R1
        #     for a in [n for n in nodes if n not in [b, c]]:
        #         if not are_adjacent(a, c) and directed_graph.has_edge(a, b):
        #             directed_graph.add_edge(b, c)
        #             undirected_graph.remove_edge(b, c)
        #         elif not are_adjacent(a, b) and directed_graph.has_edge(a, c):
        #             directed_graph.add_edge(c, b)
        #             undirected_graph.remove_edge(b, c)
        #     # R2

if __name__ == "__main__":
    # This will run the IC algorithm on whatever is stored in temporal files
    # Load black-boxed input from temporal files
    data = load_blackboxed_data('temp/blackboxed_output')
    # Load true model from temporal files for comparison
    true_model = load_functional_model('temp/model.pkl')
    # Run IC algorithm
    test = CompressionTest(criterion='fixed_threshold', compressor=None, pairing=None)
    ic = IC(test, labeldict=true_model.randomized_names)
    
    ic.run(data)
