
import stellargraph as sg
import gensim
from gensim.models import Word2Vec
import pickle
import networkx as nx
import operator

def most_influential(sg_model, embedding_model):
    # open a file, where you stored the pickled data
    file = open(sg_model, 'rb')
    # dump information to that file
    graph = pickle.load(file)
    # close the file
    file.close()
    print(graph.info())
    G = graph.to_networkx()
    pr = nx.degree_centrality(G)
    sorted_d = dict(sorted(pr.items(), key=operator.itemgetter(1), reverse=True))
    print(sorted_d)
    article_list = []
    author_list = []
    disease_list = []
    drug_list = []
    gene_list = []
    side_effect_list = []
    for record in sorted_d.keys():
        node_id = record
        node_type = graph.node_type(node_id)
        if node_type == 'article':
            article_list.append(node_id)
        if node_type == 'author':
            author_list.append(node_id)
        if node_type == 'disease':
            disease_list.append(node_id)
        if node_type == 'drug':
            drug_list.append(node_id)
        if node_type == 'gene':
            gene_list.append(node_id)
        if node_type == 'side_effect':
            side_effect_list.append(node_id)
    print(article_list[0:10])
    print(author_list[0:10])
    print(disease_list[0:10])
    print(drug_list[0:10])
    print(gene_list[0:10])
    print(side_effect_list[0:10])



def similarity_analysis(sg_model, embedding_model):
    # Load the embedding model
    model = gensim.models.Word2Vec.load(embedding_model, mmap='r')
    # # Retrieve node embeddings and corresponding subjects
    # node_ids = model.wv.index_to_key  # list of node IDs
    # node_indices = model.wv.key_to_index
    # node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality
    # node_targets = [sg.node_type(node_id) for node_id in node_ids]

    # open a file, where you stored the pickled data
    file = open(sg_model, 'rb')
    # dump information to that file
    graph = pickle.load(file)
    # close the file
    file.close()
    print(graph.info())

    ['article', 'author', 'disease', 'drug', 'gene', 'side_effect']

    article_list = []
    author_list = []
    disease_list = []
    drug_list = []
    gene_list = []
    side_effect_list = []

    result = model.wv.most_similar(positive=["author/6190153"], topn=1000)
    for record in result:
        node_id = record[0]
        node_type = graph.node_type(node_id)
        if node_type == 'article':
            article_list.append(node_id)
        if node_type == 'author':
            author_list.append(node_id)
        if node_type == 'disease':
            disease_list.append(node_id)
        if node_type == 'drug':
            drug_list.append(node_id)
        if node_type == 'gene':
            gene_list.append(node_id)
        if node_type == 'side_effect':
            side_effect_list.append(node_id)


    print(article_list[0:10])
    print(author_list[0:10])
    print(disease_list[0:10])
    print(drug_list[0:10])
    print(gene_list[0:10])
    print(side_effect_list[0:10])



