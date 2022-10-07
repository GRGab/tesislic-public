import itertools
import subprocess
from typing import Dict, Iterable, Optional, Tuple
import matplotlib.image as mpimg  # for displaying .png files
import matplotlib.pyplot as plt
import networkx as nx
import pydot

def visualize_graph(graph,
                    savepath=None,
                    labeldict=None,
                    title=None,
                    edge_labels=None,
                    only_save=False):
    if labeldict is not None:
        if edge_labels is not None: # Update keys in edge_labels according to node labels
        # this must happen before relabelling nodes
            edge_labels = {(labeldict[e[0]], labeldict[e[1]]): edge_labels[e] for e in graph.edges}
        graph = nx.relabel_nodes(graph, labeldict)
    p = to_pydot(graph, edge_labels)
    savepath = 'temp/temp.png' if savepath is None else savepath
    p.write_png(savepath)
    if not only_save:
        img = mpimg.imread(savepath)
        plt.figure()
        _ = plt.imshow(img)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    return p

# function used in ICMultiRun class
def visualize_multirun(multirun, savepath=None, only_save=False):
    for i in range(len(multirun.independence_test)):
        g = nx.Graph()
        g.add_nodes_from(multirun.model.nodes)
        edges = [e for e in itertools.combinations(multirun.model.nodes, 2)
            if multirun.edge_counts[0][e] != 0]
        g.add_edges_from(edges)
        # 2021-04-29: Just realized line below is fragile -- depends on order of itertools.combinations
        # Because multirun.edge_counts[i][(a, b)] is not equal to multirun.edge_counts[i][(b, a)]
        edge_labels = {e: f'{multirun.edge_counts[i][e]/multirun.n_repetitions:.3g}' for e in g.edges}
        title = None if multirun._is_single_test else f'Independence test #{i}'
        svpth = f'{savepath}_test_{i}.png' if savepath is not None else None
        visualize_graph(g, edge_labels=edge_labels, title=title, savepath=svpth,
                        only_save=only_save)

###############################################################################
# Utilities for working with pydot and graphviz
###############################################################################

def to_pydot(nx_graph, edge_labels=None):
    if edge_labels is not None:            
        for e in nx_graph.edges():
            nx_graph[e[0]][e[1]]['label'] = edge_labels[e]
    p = nx.drawing.nx_pydot.to_pydot(nx_graph)
    return p

def get_graphviz_string(pydot_graph):
    output_graphviz_dot = pydot_graph.create_dot()
    graphviz_string = output_graphviz_dot.decode('utf-8')
    return graphviz_string

def build_pydot_graph(nodes : Iterable[str],
                      edges : Iterable[Tuple[str, str]],
                      kind  : str = 'graph',
                      nodes_kwargs: Optional[Dict[str, dict]] = None,
                      edges_kwargs: Optional[Dict[str, dict]] = None):
    p = pydot.Dot('my_graph', graph_type=kind, engine='neato')
    pydot_nodes = {}
    for node in nodes:
        n_kwargs = nodes_kwargs[node] if node in nodes_kwargs else {}
        pydot_nodes[node] = pydot.Node(node, **n_kwargs)
        p.add_node(pydot_nodes[node])    
    pydot_edges = {}
    for e in edges:
        e_kwargs = edges_kwargs[e] if e in edges_kwargs else {}
        pydot_edges[e] = pydot.Edge(pydot_nodes[e[0]], pydot_nodes[e[1]],
                                    **e_kwargs)
        p.add_edge(pydot_edges[e])
    return p

def draw_pydot_graph(pydot_graph, path, fmt='pdf'):
    pydot_graph.write_raw('temp/temp_output_graphviz.dot')
    subprocess.run(['neato', f'-T{fmt}', '-Gsplines=true',
                    'temp/temp_output_graphviz.dot', '-o', f'{path}.{fmt}'],
                   check=True)

if __name__ == "__main__":
    G = nx.complete_graph(5)
    word = 'abcde'
    the_labels={i: word[i] for i in range(5)}
    the_edge_labels = {e : 'a' for e in G.edges}
    p = visualize_graph(G, labeldict=the_labels, edge_labels=the_edge_labels)
