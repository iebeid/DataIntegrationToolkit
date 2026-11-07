# importing modules
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import sparse


def hits_main(graph,max_iterations,tolerance):
    #source nodes
    source_nodes = []
    target_nodes = []
    for edge in graph:
        source_nodes.append(edge[0])
        target_nodes.append(edge[1])
    unique_nodes = list(np.unique(np.array(source_nodes + target_nodes)))
    #index unique nodes
    nodes_indexed = {}
    nodes = {}
    for index, node in enumerate(unique_nodes):
        nodes_indexed[node] = int(index)
        nodes[int(index)] = node
    #create indexed graph
    graph_indexed = {}
    weights = []
    for edge in graph:
        graph_indexed[int(nodes_indexed[edge[0]])] = int(nodes_indexed[edge[1]])
        weights.append(float(edge[2]))
    weights = np.array(weights)
    #initialize the the reference score matrix
    number_of_nodes = len(nodes)
    r = np.full(number_of_nodes, float(1 / number_of_nodes), dtype=float)
    e = np.full(number_of_nodes, float(1 / number_of_nodes), dtype=float)
    for _ in range(max_iterations):
        rlast = r
        r = np.zeros(number_of_nodes, dtype=float)
        e = np.zeros(number_of_nodes, dtype=float)

        e = rlast.dot(weights)
        r = weights.dot(e)

        s = 1.0 / np.max(r)
        r = r * s

        s = 1.0 / np.max(e)
        e = e * s

        # check convergence, l1 norm
        err = np.sum([np.abs(r - rlast)])
        if err < tolerance:
            break

        s = 1.0 / np.sum(e)
        e = e * s

        s = 1.0 / np.sum(r)
        r = r * s

    reference_scores = {}
    entity_scores = {}

    for reference_score, i in enumerate(r):
        reference_scores[nodes[i]] = reference_score

    for entity_score, j in enumerate(r):
        entity_scores[nodes[j]] = entity_score

    return reference_scores, entity_scores



def hits_main_scipy(graph,max_iterations,tolerance):
    #source nodes
    source_nodes = []
    target_nodes = []
    for edge in graph:
        source_nodes.append(edge[0])
        target_nodes.append(edge[1])
    unique_nodes = list(np.unique(np.array(source_nodes + target_nodes)))
    #index unique nodes
    nodes_indexed = {}
    nodes = {}
    for index, node in enumerate(unique_nodes):
        nodes_indexed[node] = int(index)
        nodes[int(index)] = node
    #create indexed graph
    graph_indexed = {}
    weights = []
    for edge in graph:
        graph_indexed[int(nodes_indexed[edge[0]])] = int(nodes_indexed[edge[1]])
        weights.append(float(edge[2]))
    weights = np.array(weights)





def main():
    G = nx.DiGraph()

    G.add_edges_from([('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'B'),
                      ('D', 'C'), ('E', 'D'), ('E', 'B'), ('E', 'F'),
                      ('E', 'C'), ('F', 'C'), ('F', 'H'), ('G', 'A'),
                      ('G', 'C'), ('H', 'A')])

    plt.figure(figsize=(10, 10))

    nx.draw_networkx(G, with_labels=True)
    plt.show()


    tic1 = time.clock()
    hubs1, authorities1 = nx.hits(G, max_iter=50, normalized=True)
    toc1 = time.clock()
    time1 = toc1 - tic1
    tic2 = time.clock()
    hubs2, authorities2 = nx.hits_numpy(G, normalized=True)
    toc2 = time.clock()
    time2 = toc2 - tic2
    tic3 = time.clock()
    hubs3, authorities3 = nx.hits_scipy(G, max_iter=50, normalized=True)
    toc3 = time.clock()
    time3 = toc3 - tic3
    # The in-built hits function returns two dictionaries keyed by nodes
    # containing hub scores and authority scores respectively.

    print("NetworkX Hub Scores 1: ", hubs1)
    print("NetworkX Authority Scores 1: ", authorities1)
    print("Time: ", time1)

    print("NetworkX Hub Scores 2: ", hubs2)
    print("NetworkX Authority Scores 2: ", authorities2)
    print("Time: ", time2)

    print("NetworkX Hub Scores 3: ", hubs3)
    print("NetworkX Authority Scores 3: ", authorities3)
    print("Time: ", time3)

    # #---------------------------------------
    #
    # graph = [('A', 'D', 1), ('B', 'C', 1), ('B', 'E', 1), ('C', 'B', 1),
    #                   ('D', 'C', 1), ('E', 'D', 1), ('E', 'B', 1), ('E', 'F', 1),
    #                   ('E', 'C', 1), ('F', 'C', 1), ('F', 'H', 1), ('G', 'A', 1),
    #                   ('G', 'C', 1), ('H', 'A', 1)]
    # references, entities = hits_main(graph, max_iterations=100, tolerance=1.0e-6)
    #
    # print("Akef References Scores: ", references)
    # print("Akef Entities Scores: ", entities)


if __name__ == '__main__':
    main()