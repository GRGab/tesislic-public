import itertools
import pydot

import networkx as nx

from graphs import (graph_1, graph_2, graph_3, graph_4, graph_5)
from visualization import build_pydot_graph, draw_pydot_graph, \
                          get_graphviz_string, to_pydot

graphs = {
    1 : graph_1,
    2 : graph_2,
    3 : graph_3,
    4 : graph_4,
    5 : graph_5
}

###############################################################################
# Crear imágenes con layout de nodos fijo, a partir de archivos .pkl
###############################################################################

# Generar imágenes de grafos estructurales
def draw_the_graphs_pydot():
    titles = [f"Grafo {i}" for i in ['Y', 'Instrumental', 'Tridente',
                                    'Tridente Instrumental', 'W']]
    for i, title in zip([1, 2, 3, 4, 5], titles):
        # visualize_graph(g, savepath=f'pictures/graphs/{title.replace(" ", "_")}.png', title=None)
        path=f'pictures/graphs/{title.replace(" ", "_")}'
        g = graphs[i]
        nodes_kwargs = {n : {'pos' : positions_dict_pydot[i][n],
                             'fontsize' : 25}
                        for n in g.nodes}
        edges_kwargs = {e : {'arrowsize' : 1.5,
                             'penwidth' : 2,
                             'fontsize' : 30} for e in g.edges}
        pydot_graph = build_pydot_graph(g.nodes, g.edges,
                                        kind='digraph',
                                        nodes_kwargs=nodes_kwargs,
                                        edges_kwargs=edges_kwargs)
        draw_pydot_graph(pydot_graph, path, fmt='pdf')

def draw_multirun_pydot(multirun, path, fmt='pdf', **kwargs):
    pydot_graph = build_multirun_pydot_graph(multirun, **kwargs)
    draw_pydot_graph(pydot_graph, path, fmt=fmt)
    return pydot_graph

def build_multirun_pydot_graph(multirun,
                               node_fontsize = 25,
                               edge_fontsize = 30,
                               draw_underlying_dotted = True,
                               lighter_lines_level = 0.1):
    nodes = multirun.model.nodes
    graph_number = get_underlying_graph_number(multirun)
    nodes_kwargs = {n : {
                            'pos' : positions_dict_pydot[graph_number][n],
                            'fontsize' : node_fontsize
                        }
                    for n in nodes}
    # only draw edges with positive count
    # we assume there's only one indep test in this multirun (so we set i=0)
    edges = [e for e in itertools.combinations(multirun.model.nodes, 2)
            if multirun.edge_counts[0][e] != 0]
    edge_frequencies = {e : multirun.edge_counts[0][e]/multirun.n_repetitions for e in edges}
    # mark edges that should be drawn more lightly
    lighter_edges = [e for e in edges if edge_frequencies[e] <= lighter_lines_level]
    edges_kwargs = {e : {
                            'label' : f'{edge_frequencies[e]:.3g}',
                            'fontsize' : edge_fontsize,
                            'penwidth' : 0.5 if e in lighter_edges else 3
                        }
                    for e in edges}
    
    pydot_graph = build_pydot_graph(nodes, edges,
                                    nodes_kwargs=nodes_kwargs,
                                    edges_kwargs=edges_kwargs)    
    if draw_underlying_dotted:
        for e in graphs[graph_number].edges:
            if e not in edges and (e[1], e[0]) not in edges:
                pydot_graph.add_edge(pydot.Edge(e[0], e[1], style='dashed'))
    return pydot_graph

def build_positions_dict_pydot():
    positions_dict = {
        1: {
            'A': "1,5!",
            'B': "5,5!",
            'X': "3,3!",
            'Y': "3,1!"
        },
        2: {
            'A': "1,5!",
            'B': "5,5!",
            'X': "3,3!",
            'Y': "5,1!"
        },
        3: {
            'A': "1,5!",
            'B': "3,5!",
            'C': "5,5!",
            'X': "3,3!",
            'Y': "3,1!"
        },
        4: {
            'A': "1,5!",
            'B': "3,5!",
            'C': "5,5!",
            'X': "3,3!",
            'Y': "3,1!"
        },
        5: {
            'A': "1,5!",
            'B': "3,5!",
            'C': "5,5!",
            'X': "2,1!",
            'Y': "4,1!"
        },
    }
    return positions_dict
positions_dict_pydot = build_positions_dict_pydot()

def get_underlying_graph_number(multirun):
    underlying_graph = multirun.model.graph
    graphs = [graph_1, graph_2, graph_3, graph_4, graph_5]
    for i, graph in enumerate(graphs):
        if nx.is_isomorphic(graph, underlying_graph):
            return i+1
    # If nothing is returned...
    raise ValueError("Underlying graph of input multirun is not valid")

def generate_graphviz_strings():
    graphs = [graph_1, graph_2, graph_3, graph_4, graph_5]
    graphviz_strings = []
    for g in graphs:
        p = to_pydot(g)
        graphviz_string = get_graphviz_string(p)
        graphviz_strings.append(graphviz_string)
    return graphviz_strings