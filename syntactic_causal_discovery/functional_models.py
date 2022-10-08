from functools import partial
from numpy.random import random, dirichlet
from random import sample as randomsample
from typing import Callable, Dict, Iterable, List, Literal, Optional, Union

import dill
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from syntactic_causal_discovery.graphs import WGraph, generate_random_DAG
from syntactic_causal_discovery.string_transformations import random_string, stringnoise
from syntactic_causal_discovery.utils import InitializationError
from syntactic_causal_discovery.visualization import visualize_graph

plt.ion()

# type alias
Alphabet = Iterable[str]
Direction = Literal['Forward', 'Backward']

class FunctionalModel():
    """
    Right now, ordering of inputs for each node function is given by lexicographic
    order of labels of parent nodes.

    Possible improvement (TODO?): add explicit ordering to parent-sets for evaluation of node functions
    Inputs to constructor

    Possible refactor: move funcdict structure inside networkx.DiGraph object.
    This can be done through something like
    `setfuncs = lambda G, dic: nx.set_node_attributes(G, dic, name='function')`

    ---------------------
    G : nx.DiGraph
        Must be a DAG
    funcdict : dict
        (node : func) with func a function to associate w/ each node
        root nodes: func determines probability distribution on inputs
        non-root nodes: func determines what is done to inputs of the node
    """
    def __init__(self,
                 graph : nx.DiGraph,
                 funcdict : dict,
                 alphabet : Optional[Alphabet] = None
                 ):
        self.alphabet = alphabet
        if isinstance(graph, nx.DiGraph):
            self.graph = graph
            self.N = graph.order()
        else:
            self.graph = nx.DiGraph(graph)
            self.N = self.graph.order()
        # Make sure graph is a DAG
        assert nx.is_directed_acyclic_graph(graph)
        # Assign function to nodes or set them as input (input = graph root)
        self.funcdict = funcdict
        self.nodes = list(self.graph.nodes)
        self.root_nodes = [node for node, in_degree in graph.in_degree if in_degree == 0]
        self.alphabet = alphabet
        # Initialize empty state
        self.input_values = {n : None for n in self.root_nodes}
        self.state = {n : None for n in self.graph.nodes}
        self.state_is_nonempty = False
        # Assign "randomized names" to nodes (for black-boxing model output)
        self.randomized_names = {str(i) : n for i, n in zip(range(self.N), randomsample(self.graph.nodes, self.N))}
        # Calculate a topological ordering of nodes for updating state
        self.topological_ordering = list(nx.topological_sort(self.graph))

    # TODO: Extender este método a uno que maneje intervenciones
    # Es decir, ahora mismo solo se pueden "tocar" los valores de nodos raíz
    # Usually one should call update_state() after calling this method
    def modify_input_value(self, node, value):
        self.input_values[node] = value

    def set_input_values(self, input_values):
        assert list(input_values.keys()) == self.root_nodes
        self.input_values = input_values
        self.update_state()
        self.state_is_nonempty = True

    def update_state(self):
        # This uses the topological ordering on the graph
        for node in self.topological_ordering:
            parents = list(self.graph.predecessors(node))
            if len(parents) == 0:
                # The node is root; value is given as input
                self.state[node] = self.input_values[node]
            else:
                # reorder lexicographically
                parents = sorted(parents)
                parent_values = [self.state[n] for n in parents]
                value = self.funcdict[node](*parent_values)
                self.state[node] = value

    def generate_inputs(self):
        """
        Wrapper method. Sets input values randomly according to functions in
        funcdict and updates the state of the network.
        """
        input_values = {}
        for node in self.root_nodes:
            try:
                input_values[node] = self.funcdict[node]()
            except KeyError as e:
                msg = f"Node {node} doesn't have an associated function in funcdict"
                raise InitializationError(msg) from e
        self.set_input_values(input_values)
    
    def visualize(self, labels='name', savepath=None, title=None):
        """
        Notes
        -----
        If labels == 'state', method will fail for states having too long strings
        """
        if labels == 'name':
            labeldict = None
        elif labels == 'state': # This will fail if state values are too long
            labeldict = {n : self.state[n] for n in self.graph.nodes}
        else:
            raise ValueError
        visualize_graph(self.graph, savepath=savepath, labeldict=labeldict, title=title)

    def blackboxed_output(self):
        """Returns a dict with items (randomized name : node value)"""
        return {i : self.state[n] for i, n in self.randomized_names.items()}

    def save_blackboxed_output(self, filename=None):
        """Saves to a file all pairs (randomized name, node value)"""
        filename = 'temp/blackboxed_output' if filename is None else filename
        with open(filename, "w") as outfile:
            text_lines = ["{} {}".format(i, self.state[n]) for i, n in self.randomized_names.items()]
            outfile.write("\n".join(text_lines))

    def save_to_disk(self, filename=None):
        filename = 'temp/model.pkl' if filename is None else filename
        with open(filename, 'wb') as f:
            dill.dump(self, f)

class ParametricModel(FunctionalModel):
    """
    """
    def __init__(self,
                 *,
                 graph : nx.DiGraph,
                 structure_function : Callable,
                 alphabet : Alphabet,
                 len_strings : int,
                 probs : dict,
                 noise_level : float = 0.2,
                 shift : Union[int, dict] = 0,
                 direction : Union[Direction, dict]):
        self.shift = shift
        self.direction = direction
        funcdict = {}
        for node, in_degree in graph.in_degree:
            if in_degree == 0:
                funcdict[node] = partial(random_string, len_strings,
                                         probs[node], alphabet)
            else:
                shift_to_apply = self.shift if isinstance(self.shift, int) else self.shift[node]
                direction_to_apply = self.direction if isinstance(self.direction, str) else self.direction[node]
                funcdict[node] = stringnoise(noise_level,
                                             alphabet=alphabet)(partial(structure_function,
                                                                        shift_to_apply, direction_to_apply))
        super().__init__(graph, funcdict, alphabet=alphabet)

class ParametricModelUniformProbs(ParametricModel):
    def __init__(self,
                 *,
                 graph: nx.DiGraph,
                 structure_function,
                 alphabet: Alphabet,
                 len_strings : int,
                 noise_level : float = 0.2,
                 shift : Union[int, dict] = 0,
                 direction : Union[Direction, dict] = "Forward"):
        # Hago que funcione con la función random_string() tal como está definida
        # actualmente (feo)
        if alphabet == ['0', '1']:
            probs = {node: 0.5 for node in graph.nodes if graph.in_degree(node) == 0}
        else:
            pp = [1/len(alphabet) for _ in range(len(alphabet))]
            probs = {node: pp for node in graph.nodes if graph.in_degree(node) == 0}
        super().__init__(graph = graph,
                         structure_function = structure_function,
                         alphabet = alphabet,
                         len_strings = len_strings,
                         probs = probs,
                         noise_level = noise_level,
                         shift = shift,
                         direction = direction)
            
class ParametricModelRandomProbs(ParametricModel):
    def __init__(self,
                 *,
                 graph: nx.DiGraph,
                 structure_function,
                 alphabet: Alphabet,
                 len_strings : int,
                 noise_level : float = 0.2,
                 shift : Union[int, dict] = 0,
                 direction : Union[Direction, dict] = "Forward"):
        # Hago que funcione con la función random_string() tal como está definida
        # actualmente (feo)
        if alphabet == ['0', '1']:
            p = random() # equivalent to the dirichlet distribution with 2 parameters 
            probs = {node: p for node in graph.nodes if graph.in_degree(node) == 0}
        else:
            l = len(alphabet)
            # mean_values = np.ones(l) / l
            pp = dirichlet(np.ones(l)).tolist()
            # import pdb; pdb.set_trace()
            probs = {node: pp for node in graph.nodes if graph.in_degree(node) == 0}
        super().__init__(graph = graph,
                         structure_function = structure_function,
                         alphabet = alphabet,
                         len_strings = len_strings,
                         probs = probs,
                         noise_level = noise_level,
                         shift = shift,
                         direction = direction)
        
class ParametricModelNonUniformProbs(ParametricModel):
    """
    Difference with ParametricModel: `probs` is just a list with probabilities
    for symbols, and it is applied uniformly to all root nodes.
    """
    def __init__(self,
                 *,
                 graph: nx.DiGraph,
                 structure_function,
                 alphabet: Alphabet,
                 probs: List[float],
                 len_strings : int,
                 noise_level : float = 0.2,
                 shift : Union[int, dict] = 0,
                 direction : Union[Direction, dict] = "Forward"):
        probdict = {node: probs for node in graph.nodes if graph.in_degree(node) == 0}
        super().__init__(graph = graph,
                         structure_function = structure_function,
                         alphabet = alphabet,
                         len_strings = len_strings,
                         probs = probdict,
                         noise_level = noise_level,
                         shift = shift,
                         direction = direction)

##### Loading models from disk
def load_functional_model(filename):
    with open(filename, 'rb') as f:
        model = dill.load(f)
    return model

##### Model construction

## Function constructors

def build_xorfunc(arity, keyword_length=1):
    """
    TODO: Re-implement for working with arguments of different lengths simultaneously
    """
    xor_word = random_string(keyword_length) # internal xor keyword
    def xorfunc(*args):
        assert len(args) == arity, "arity of this function is {}, not {}".format(arity, len(args))
        l = len(args[0])
        k_repeated = (xor_word * ((l // keyword_length) + 1)) [:l]
        value = int(k_repeated, 2) 
        for x in args:
            value = value ^ int(x, 2)
        return '{1:0{0}b}'.format(l, value)
    xorfunc.arity = arity
    return xorfunc

### legacy, probably delete
def build_concatfunc(arity):
    f = lambda *x: ''.join(x)
    f.arity = arity # necesario para que no tire error, aunque en este caso es redundante
    return f

## Funcdict constructors

### legacy, probably delete
def build_uniform_funcdict(graph, function_constructor, **kwargs):
    """
    function_constructor : function
        given arity, returns a function of that arity
    """
    funcdict = {}
    for node, deg in graph.in_degree():
        if deg != 0:
            funcdict[node] = function_constructor(deg, **kwargs)
    return funcdict

## Functional model subclasses, that make a graph into a prespecified model
class ConcatenationModel(FunctionalModel):
    def __init__(self, graph):
        funcdict = build_uniform_funcdict(graph, build_concatfunc)
        super().__init__(graph, funcdict)

# More specific functional models
class WConcatModel(ConcatenationModel):
    def __init__(self):
        super().__init__(WGraph())

def generate_random_xor_model(n, p, k):
    """
    k : int
        Length of words used when building xor functions
    TODO: Testear
    """
    G = generate_random_DAG(n, p)
    funcdict = build_uniform_funcdict(G, build_xorfunc, keyword_length=k)
    model = FunctionalModel(G, funcdict)
    return model


if __name__ == "__main__":
    # # Minimal model with a v-structure and concatenation as function
    # graph = nx.DiGraph([('A', 'C'), ('B', 'C')])
    # funcdict = build_uniform_funcdict(graph, build_concatfunc)
    # model_1 = FunctionalModel(graph, funcdict)
    # # Choose one between big_inputs and small_inputs
    # small_inputs = {'A': '000', 'B': '111'}
    # big_inputs = {'A': '01101'*int(1e4), 'B': '11010110001'*int(1e4)}
    # inputs = big_inputs
    # model_1.set_input_values(inputs)
    # model_1.visualize()
    # model_1.save_to_disk()
    # model_1.save_blackboxed_output()

    # More complicated word-concatenation model
    graph = nx.DiGraph([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('E', 'D'), ('G', 'H')])
    funcdict = build_uniform_funcdict(graph, build_concatfunc)
    model_2 = FunctionalModel(graph, funcdict)
    model_2.set_input_values({'A': '01101'*int(1e4), 'E': '11010110001'*int(1e4), 'G': '1'*int(1e4)})
    model_2.visualize()
    model_2.save_to_disk()
    model_2.save_blackboxed_output()


    # # Pseudorandomly generated model where functions apply xor of inputs and an
    # # additional small keyword (which is repeated to match input lengths)
    # model_3 = generate_random_xor_model(10, 0.3, 10)
    # model_3.set_random_inputs(100)
    # model_3.visualize()
    # model_3.save_blackboxed_output()
