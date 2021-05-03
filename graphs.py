import networkx as nx
from histogram import histogram
from visualization import visualize_graph

class WGraph(nx.DiGraph):
    def __init__(self, **attr):
        data = [('A', 'C'), ('B', 'C'), ('B', 'D'), ('E', 'D')]
        super().__init__(incoming_graph_data=data, **attr)

class InstrumentalGraph(nx.DiGraph):
    def __init__(self, **attr):
        data = [('U', 'X'), ('Z', 'X'), ('U', 'Y'), ('X', 'Y')]
        super().__init__(incoming_graph_data=data, **attr)

## Grafos con ordenamiento y etiquetas nuevas
# grafo Y
graph_Y = nx.DiGraph([('A', 'X'), ('B', 'X'), ('X', 'Y')])
graph_1 = graph_Y
# grafo instrumental
graph_instrum = nx.DiGraph([('A', 'X'), ('B', 'X'), ('B', 'Y'), ('X', 'Y')])
graph_2 = graph_instrum
# grafo tridente
graph_tridente = nx.DiGraph([('A', 'X'), ('B', 'X'), ('C', 'X'), ('X', 'Y')])
graph_3 = graph_tridente
# tridente instrumental
graph_trid_inst = nx.DiGraph([('A', 'X'), ('B', 'X'), ('C', 'X'), ('B', 'Y'), ('X', 'Y')])
graph_4 = graph_trid_inst
# grafo W
graph_W = nx.DiGraph([('A', 'X'), ('B', 'X'), ('B', 'Y'), ('C', 'Y')])
graph_5 = graph_W
# grafo del riego (creo que no lo voy a usar)
graph_riego = nx.DiGraph([('A', 'X'), ('B', 'X'), ('C', 'A'), ('C', 'B'), ('X', 'Y')])
graph_6 = graph_riego

## Grafos con ordenamiento y etiquetas viejas
# Modelo 3
graph_1_old = nx.DiGraph([('U', 'X'), ('Z', 'X'), ('U', 'Y'), ('X', 'Y')])
# Modelo 4
graph_2_old = nx.DiGraph([('U', 'X'), ('Z', 'X'), ('U', 'Y'), ('X', 'Y'), ('H', 'X')])
# Modelo 5
graph_3_old = nx.DiGraph([('U', 'X'), ('Z', 'X'), ('X', 'Y')])
# Modelo 6
graph_4_old = nx.DiGraph([('U', 'X'), ('Z', 'X'), ('X', 'Y'), ('H', 'X')])
# Modelo 7
graph_5_old = nx.DiGraph([('A', 'C'), ('B', 'C'), ('B', 'D'), ('E', 'D')])

def generate_random_DAG(n, p):
    G = nx.gnp_random_graph(n, p, directed=True)
    DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u < v])
    return DAG

def generate_random_DAG_bounded_indegree(n, p, max_indegree):
    G = nx.gnp_random_graph(n, p, directed=True)
    for (u, v) in list(G.edges()):
        if u >= v:
            G.remove_edge(u, v)
        elif G.in_degree(v) > max_indegree:
            G.remove_edge(u, v)
    return G

if __name__ == "__main__":
    ### Tests for generate_random_DAG_bounded_indegree
    # Visualize one graph
    graph = generate_random_DAG_bounded_indegree(10, 0.5, 3)
    visualize_graph(graph)