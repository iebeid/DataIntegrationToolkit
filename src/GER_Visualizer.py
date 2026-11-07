
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score



import os
import networkx as nx
import numpy as np
import pandas as pd

import stellargraph as sg

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def visualize(edges, nodes):

    def get_node_type(node, nodes):
        type = ""
        for n in nodes:
            if n[1] == node:
                type = n[0]
        return type




    source_nodes = []
    target_nodes = []
    for e in edges:
        source = e[0]
        target = e[1]
        source_nodes.append(source)
        target_nodes.append(target)
    edges_data_frame = pd.DataFrame(
        {"source": source_nodes, "target": target_nodes}
    )
    graph = sg.StellarGraph(edges=edges_data_frame)
    print(graph.info())
    rw = BiasedRandomWalk(graph)

    walks = rw.run(
        nodes=list(graph.nodes()),  # root nodes
        length=100,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    )
    print("Number of random walks: {}".format(len(walks)))
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=5, negative=7, min_count=0, sg=1, hs=0, workers=8, epochs=2, sample=0)

    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [get_node_type(node_id, nodes) for node_id in node_ids]
    # node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

    # Apply t-SNE transformation on node embeddings
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)

    # draw the points
    alpha = 1.0
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="tab20",
        alpha=alpha,
    )


    #figure out edges
    edge_lines_x = []
    edge_lines_y = []
    for e in graph.edges():
        source_node = e[0]
        target_node = e[1]
        edge_lines_x.append(node_embeddings_2d[int(model.wv.key_to_index[source_node]), 0])
        edge_lines_x.append(node_embeddings_2d[int(model.wv.key_to_index[target_node]), 0])
        edge_lines_y.append(node_embeddings_2d[int(model.wv.key_to_index[source_node]), 1])
        edge_lines_y.append(node_embeddings_2d[int(model.wv.key_to_index[target_node]), 1])

    # plt.plot(edge_lines_x, edge_lines_y, alpha=0.3)

    plt.show()
