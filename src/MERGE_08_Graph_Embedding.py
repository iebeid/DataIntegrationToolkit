
import pickle
import stellargraph as sg
import numpy as np
import pandas as pd
from stellargraph.data import UniformRandomMetaPathWalk
import gensim
from matplotlib import *
import sys
from pylab import *
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import random
from tqdm import tqdm
from stellargraph.data import BiasedRandomWalk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

def construct_sg_graph(graph_object_file, graph_metadata, stellargraph_model):
    # open a file, where you stored the pickled data
    file = open(graph_object_file, 'rb')
    # dump information to that file
    graph_triples = pickle.load(file)
    # close the file
    file.close()

    source_nodes = []
    target_nodes = []
    orientation = []
    for triple in graph_triples:
        source_nodes.append(triple[0])
        target_nodes.append(triple[2])
        orientation.append(triple[1])

    all_nodes = source_nodes + target_nodes
    nodes = np.unique(np.array(all_nodes)).tolist()
    #drug, gene, author, article, other types from chem2bio2rdf
    gene_list = []
    drug_list = []
    # sub_structure_list = []
    # gene_ontology_list = []
    # chem_ontology_list = []
    # tissue_list = []
    # pathway_list = []
    disease_list = []
    side_effect_list = []
    article_list = []
    author_list = []

    print("Original size: " + str(len(nodes)))
    for n in nodes:
        node_type = graph_metadata[n]
        if node_type == "gene":
            gene_list.append(n)
        if node_type == "drug":
            drug_list.append(n)
        # if node_type == "sub_structure":
        #     sub_structure_list.append(n)
        # if node_type == "gene_ontology":
        #     gene_ontology_list.append(n)
        # if node_type == "chem_ontology":
        #     chem_ontology_list.append(n)
        # if node_type == "tissue":
        #     tissue_list.append(n)
        # if node_type == "pathway":
        #     pathway_list.append(n)
        if node_type == "disease":
            disease_list.append(n)
        if node_type == "side_effect":
            side_effect_list.append(n)
        if node_type == "article":
            article_list.append(n)
        if node_type == "author":
            author_list.append(n)


    # full_nodes = gene_list + drug_list + sub_structure_list + article_list + author_list + chem_ontology_list + gene_ontology_list + tissue_list + pathway_list + disease_list + side_effect_list

    full_nodes = gene_list + drug_list + article_list + author_list + disease_list + side_effect_list
    print("size: " + str(len(full_nodes)))




    gene_df = pd.DataFrame([], index=gene_list)
    drug_df = pd.DataFrame([], index=drug_list)
    # sub_structure_df = pd.DataFrame([], index=sub_structure_list)
    # gene_ontology_df = pd.DataFrame([], index=gene_ontology_list)
    # chem_ontology_df = pd.DataFrame([], index=chem_ontology_list)
    # tissue_df = pd.DataFrame([], index=tissue_list)
    # pathway_df = pd.DataFrame([], index=pathway_list)
    disease_df = pd.DataFrame([], index=disease_list)
    side_effect_df = pd.DataFrame([], index=side_effect_list)
    article_df = pd.DataFrame([], index=article_list)
    author_df = pd.DataFrame([], index=author_list)

    edges_df = pd.DataFrame({"source": source_nodes, "target": target_nodes, "orientation": orientation})
    sg_graph = sg.StellarGraph(nodes={
                                    "gene": gene_df,
                                    "drug": drug_df,
                                    # "sub_structure": sub_structure_df,
                                    # "gene_ontology": gene_ontology_df,
                                    # "chem_ontology": chem_ontology_df,
                                    # "tissue": tissue_df,
                                    # "pathway": pathway_df,
                                    "disease": disease_df,
                                    "side_effect": side_effect_df,
                                    "article": article_df,
                                    "author": author_df
                            },
                            edges=edges_df,
                            edge_type_column="orientation")
    file = open(stellargraph_model, 'wb')
    # dump information to that file
    pickle.dump(sg_graph, file)
    # close the file
    file.close()
    return sg_graph



def embed_graph_node2vec(sg_model):
    # open a file, where you stored the pickled data
    file = open(sg_model, 'rb')
    # dump information to that file
    sg = pickle.load(file)
    # close the file
    file.close()
    print(sg.info())
    print("Embedding graph")

    rw = BiasedRandomWalk(sg)

    walks = rw.run(
        nodes=list(sg.nodes()),  # root nodes
        length=50,  # maximum length of a random walk
        n=2,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    )
    print("Number of random walks: {}".format(len(walks)))
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=5, negative=7, min_count=0, sg=1, workers=8, epochs=5, sample=0)
    model.save("pubmed2c2b2r.node2vec.npy")





def embed_graph_metapath2vec(sg_model):
    # open a file, where you stored the pickled data
    file = open(sg_model, 'rb')
    # dump information to that file
    sg = pickle.load(file)
    # close the file
    file.close()
    print("Embedding graph")
    walk_length = 50  # maximum length of a random walk to use throughout this notebook
    #   author-article-gene-article-author
	#   author-article-drug-article-author
	#   drug-gene-drug
	#   gene-drug-gene
	#   article-gene-article
	#   article-drug-article

    # specify the metapath schemas as a list of lists of node types.
    metapaths = [
        ["author", "article", "gene", "article", "author"],
        ["author", "article", "drug", "article", "author"],
        ["drug", "article", "author", "article", "drug"],
        ["drug", "author", "article", "author", "drug"],
        ["gene", "article", "author", "article", "gene"],
        ["gene", "author", "article", "author", "gene"],
        ["gene", "drug", "article", "drug", "gene"],
        ["drug", "gene", "article", "gene", "drug"],
        ["drug", "gene", "drug"],
        ["gene", "drug", "gene"],
        ["article", "gene", "article"],
        ["article", "drug", "article"],
        ["drug", "article", "drug"],
        ["gene", "article", "gene"],
    ["drug", "disease", "drug"],
    ["gene", "disease", "gene"],
        ["disease", "drug", "disease"],
        ["disease", "gene", "disease"],
    ["author", "drug", "disease", "drug", "author"],
    ["author", "gene", "disease", "gene", "author"],
    ["disease", "side_effect", "disease"],
    ["side_effect", "disease", "side_effect"],
    ["side_effect", "disease", "drug", "disease", "side_effect"],
    ["side_effect", "disease", "drug", "disease", "side_effect"],["author", "drug", "side_effect", "drug", "author"]]

    # Create the random walker
    rw = UniformRandomMetaPathWalk(sg)

    walks = rw.run(
        nodes=list(sg.nodes()),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=5,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )

    print("Number of random walks: {}".format(len(walks)))


    model = Word2Vec(walks, vector_size=128, alpha=0.05, min_alpha=0.0001, window=5, min_count=0, sg=1, hs=0, workers=8, negative=7, sample=0, compute_loss=True, sorted_vocab=1, epochs=100)

    print(model.wv.vectors.shape)
    metapath2vec_model_file = "metapath2vec.npy"
    model.save(metapath2vec_model_file)

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys

def analyze_model(sg_model, metapath2vec_model_file):
    print("Analyzing model")

    # open a file, where you stored the pickled data
    file = open(sg_model, 'rb')
    # dump information to that file
    sg = pickle.load(file)
    # close the file
    file.close()
    print(sg.info())
    model = gensim.models.Word2Vec.load(metapath2vec_model_file, mmap='r')



    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [sg.node_type(node_id) for node_id in node_ids]

    transform = TSNE  # PCA

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings)



    # draw the points
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    print(label_map)
    node_colours = [label_map[target] for target in node_targets]
    color_list = []
    labels = []
    for n_c in node_colours:
        c = ""
        label = ""
        if n_c == 0:
            c = "red"
            label = "article"
        if n_c == 1:
            c = "green"
            label = "author"
        if n_c == 2:
            c = "cyan"
            label="disease"
        if n_c == 3:
            c = "blue"
            label="drug"
        if n_c == 4:
            c = "black"
            label="gene"
        if n_c == 5:
            c = "yellow"
            label="side_effect"
        labels.append(label)
        color_list.append(c)
    # {'article': 'red', 'author': 'green', 'disease': 'cyan', 'drug': 'blue', 'gene': 'black', 'side_effect': 'yellow'}
    classes = ['article', 'author', 'disease', 'drug', 'gene', 'side_effect']
    # print(node_colours)
    u = np.unique(np.array(node_colours))
    print(u)
    fig = plt.figure(figsize=(50, 50))
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    # ax.set_xticks([0, 2, 4, 6, 8, 10])
    # ax.set_yticks([0, 2, 4, 6, 8, 10])
    # ax.set_zticks([0, 2, 4, 6, 8, 10])
    #
    # ax.axes.set_xlim3d(left=-10, right=10)
    # ax.axes.set_ylim3d(bottom=-10, top=10)
    # ax.axes.set_zlim3d(bottom=-10, top=10)
    # ax.axes().set(aspect="equal")
    for i in range(len(color_list)):
        # ax.scatter(node_embeddings_2d[i, 0], node_embeddings_2d[:, 1], node_embeddings_2d[:, 2], c=color_list, alpha=0.4)
        # ax.scatter(node_embeddings_2d[i, 0], node_embeddings_2d[i, 1], node_embeddings_2d[i, 2],
        #            c=color_list[i], label=labels[i], alpha=0.4, s=2)
        ax.scatter(node_embeddings_2d[i, 0], node_embeddings_2d[i, 1],
                   c=color_list[i], label=labels[i], alpha=0.6, s=500)


    # scatter = ax.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], node_embeddings_2d[:, 2], c=color_list, alpha=0.4, label=labels)


    #figure out edges
    edge_lines_x = []
    edge_lines_y = []
    edge_lines_z = []
    print(model.wv.key_to_index)
    for e in sg.edges():
        source_node = e[0]
        target_node = e[1]
        edge_lines_x.append(node_embeddings_2d[int(model.wv.key_to_index[source_node]), 0])
        edge_lines_x.append(node_embeddings_2d[int(model.wv.key_to_index[target_node]), 0])
        edge_lines_y.append(node_embeddings_2d[int(model.wv.key_to_index[source_node]), 1])
        edge_lines_y.append(node_embeddings_2d[int(model.wv.key_to_index[target_node]), 1])
        # edge_lines_z.append(node_embeddings_2d[int(model.wv.key_to_index[source_node]), 2])
        # edge_lines_z.append(node_embeddings_2d[int(model.wv.key_to_index[target_node]), 2])
        # source_node_coordiantes = (node_embeddings_2d[int(model.wv.key_to_index[source_node]), 0], node_embeddings_2d[int(model.wv.key_to_index[source_node]), 1])
        # target_node_coordinates = (node_embeddings_2d[int(model.wv.key_to_index[target_node]), 0], node_embeddings_2d[int(model.wv.key_to_index[target_node]), 1])
    # ax.plot(edge_lines_x, edge_lines_y, edge_lines_z, alpha=0.3)
    ax.plot(edge_lines_x, edge_lines_y, alpha=0.2)
    selected_nodes = random.choices(node_ids, k=50)
    for txt in selected_nodes:
        # ax.annotate(txt, (node_embeddings_2d[int(model.wv.key_to_index[txt]), 0], node_embeddings_2d[int(model.wv.key_to_index[txt]), 1],
        #                   node_embeddings_2d[int(model.wv.key_to_index[txt]), 2]), fontsize=8)
        # ax.text(node_embeddings_2d[int(model.wv.key_to_index[txt]), 0], node_embeddings_2d[int(model.wv.key_to_index[txt]), 1],
        #                   node_embeddings_2d[int(model.wv.key_to_index[txt]), 2], txt, fontsize=8)
        ax.text(node_embeddings_2d[int(model.wv.key_to_index[txt]), 0], node_embeddings_2d[int(model.wv.key_to_index[txt]), 1], txt, fontsize=16)
    # ax.title("{} visualization of node embeddings".format(transform.__name__))
    # red_patch = mpatches.Patch(color='red', label='The red data')
    #'article': 'red', 'author': 'green', 'disease': 'cyan', 'drug': 'blue', 'gene': 'black', 'side_effect': 'yellow'
    red_patch = mpatches.Patch(color='red', label='Article')
    green_patch = mpatches.Patch(color='green', label='Author')
    cyan_patch = mpatches.Patch(color='cyan', label='Disease')
    blue_patch = mpatches.Patch(color='blue', label='Drug')
    black_patch = mpatches.Patch(color='black', label='Gene')
    yellow_patch = mpatches.Patch(color='yellow', label='Side Effect')
    ax.legend(handles=[red_patch,green_patch,cyan_patch,blue_patch,black_patch,yellow_patch], loc ="lower left")
    # ax.legend()
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Legend")
    # ax.add_artist(legend1)

    plt.show()

    # Visualization 2


    # Analysis and tests
    nodes_types = {}
    for nid in node_ids:
        nodes_types[nid] = sg.node_type(nid)


    # Similarity test 1: finds the most similar author based on common drugs and genes they worked on
    listOfKeys = getKeysByValue(nodes_types, "author")
    author = random.choice(listOfKeys)
    print(author)
    result = model.wv.most_similar(positive=author, topn=10)
    print(result)

    # Similarity test 2: Most similar drugs
    listOfKeys = getKeysByValue(nodes_types, "drug")
    drug = random.choice(listOfKeys)
    print(drug)
    result = model.wv.most_similar(positive=drug, topn=10)
    print(result)



def visualize(sg_model, embedding_model):
    print("Visualize model")
    # Load the stellargraph model
    file = open(sg_model, 'rb')
    sg = pickle.load(file)
    file.close()
    print(sg.info())
    # Load the embedding model
    model = gensim.models.Word2Vec.load(embedding_model, mmap='r')
    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [sg.node_type(node_id) for node_id in node_ids]

    transform = TSNE  # PCA

    trans = transform(n_components=3)
    node_embeddings_2d = trans.fit_transform(node_embeddings)



    sns.set(style="darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = node_embeddings_2d[:, 0]
    y = node_embeddings_2d[:, 1]
    z = node_embeddings_2d[:, 2]

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(x, y, z)

    plt.show()


# def analyze_2():
#     # Retrieve node embeddings and corresponding subjects
#     node_ids = model.wv.index_to_key  # list of node IDs
#     node_embeddings = (
#         model.wv.vectors
#     )  # numpy.ndarray of size number of nodes times embeddings dimensionality
#     node_targets = [sg.node_type(node_id) for node_id in node_ids]
#     # node_targets = node_subjects.loc[[int(node_id) for node_id in node_ids]]
#     transform = TSNE  # PCA
#     trans = transform(n_components=2)
#     node_embeddings_2d = trans.fit_transform(node_embeddings)
#     # draw the embedding points, coloring them by the target label (paper subject)
#     alpha = 0.5
#     label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
#     node_colours = [label_map[target] for target in node_targets]
#     plt.figure(figsize=(7, 7))
#     plt.axes().set(aspect="equal")
#     plt.scatter(
#         node_embeddings_2d[:, 0],
#         node_embeddings_2d[:, 1],
#         c=node_colours,
#         cmap="jet",
#         alpha=alpha,
#     )
#     plt.title("{} visualization of node embeddings".format(transform.__name__))
#     plt.show()






