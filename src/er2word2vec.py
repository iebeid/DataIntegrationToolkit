# Imports
import pandas as pd
import gensim
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import itertools
from random import sample
import pickle
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import stellargraph as sg
from scipy.stats import entropy


# This function is used to compute the cosine similarity between two vectors. Cosine similarity is useful in this
# case since we are using it to create a graph of linked records.
def cosine_similarity(vector1, vector2):
    return float(float(1.0) - (
        scipy.spatial.distance.cosine(np.array(vector1, dtype=np.float), np.array(vector2, dtype=np.float))))

# This function after we train a word2vec model to learn vectors for documents we use it to retreive the most simlilar
# documents to the input document. Here similarity entails that the two records might refere to the same entity
def top_k_similar_documents(model, embedding_size, doc, docs, k):
    scores_dictionary = {}
    scores = []
    # target_document_vector = model.infer_vector(doc)
    target_document_vector = infer_vector(model, embedding_size, doc)
    for d in docs:
        # current_document_vector = model.infer_vector(d)
        current_document_vector = infer_vector(model, embedding_size, d)
        score = cosine_similarity(target_document_vector, current_document_vector)
        scores_dictionary.update({float(score): d})
        scores.append(score)
    scores.sort(reverse=True)
    similar_documents = []
    for i in range(0, k + 1):
        current_score = scores[i]
        similar_document = scores_dictionary.get(current_score)
        if doc != similar_document:
            similar_documents.append((similar_document, current_score))
    return similar_documents

# This function is used to create a tagged document object for the doc2vec model. However I ended up not using it
# because word2vec showed better results than doc2vec
def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

# This funciton is used to visualize the learned 128 dimnetional vector for each record. It project each vector
# using the tSNE algorthm to a 2 dimentaionl vecotr that can be easily visualized.
def visualize_embedding(model, embedding_size, docs):
    doc_vectors = []
    doc_types = []
    for doc in docs:
        # doc_vector = model.infer_vector(doc)
        doc_vector = infer_vector(model, embedding_size, doc)
        doc_vectors.append(doc_vector)
        doc_type = doc[0]
        doc_types.append(doc_type)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    X = np.array(doc_vectors)
    T = np.array(doc_types)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=T, legend=None, palette='bright')
    plt.show()


# This function is used to evaluate the learned word2vec or doc2vec model using the true id or ground truth
# provided in the data file. The heuristic used simplu counts the number of records that are supposed to be clustered
# with the target record in the top ten similar vector records.
def evaluate_model(model, embedding_size, ground_truth_clusters, docs):
    k = 10
    average_accuracy = 0
    loop_count = 0
    evaluation_list = sample(docs,10)
    for doc in tqdm(evaluation_list):
        doc_type = doc[0]
        current_cluster_count = get_number_of_occurances_in_list(doc_type, ground_truth_clusters)
        test_doc_vec = infer_vector(model, embedding_size, doc)
        top_k_similar_docs = top_k_similar_documents(model, embedding_size, doc, docs, k)
        similar_doc_count = 0
        for s in top_k_similar_docs:
            similar_doc = s[0]
            similar_doc_type = s[0][0]
            if similar_doc_type == doc_type:
                similar_doc_count = similar_doc_count + 1
        if current_cluster_count > 1:
            accuracy = (similar_doc_count / (current_cluster_count)) * 100
            loop_count = loop_count + 1
            average_accuracy = average_accuracy + accuracy
    average_accuracy = average_accuracy / loop_count
    return average_accuracy


# Utility function to count the number of occurances of an item in a list
def get_number_of_occurances_in_list(c, l):
    cluster_occurances = [(g[0], len(list(g[1]))) for g in itertools.groupby(l)]
    for r in cluster_occurances:
        if c == r[0]:
            return int(r[1])


# Infer vector for a document. Simply done by computing the average of the vectors for each word in the record
# learned by word2vec
def infer_vector(model, embedding_size, list_of_words):
    final_vector = np.zeros(embedding_size)
    for w in list_of_words:
        vector = np.array(model.wv[w])
        final_vector = final_vector + vector
    final_vector = final_vector / len(list_of_words)
    return final_vector

# return the unique idetifier for each record in the original data file
def get_document_id(dataframe, doc):
    doc_cleaned = doc[1:]
    separator = ' '
    record = separator.join(doc_cleaned).upper()
    return_id = 0
    for index, row in dataframe.iterrows():
        reference = str(row[3])
        if reference == record:
            return_id = dataframe.loc[index][0]
    return return_id

# return the true id for each record in the data file
def get_document_cluster_id(dataframe, doc_id):
    cluster_id_return = 0
    for index, row in dataframe.iterrows():
        cluster_id = str(row[2])
        if row[0] == doc_id:
            cluster_id_return = cluster_id
    return cluster_id_return

# return each record given its unique identifier
def get_document_by_id(dataframe, id):
    return_referece = ""
    for index, row in dataframe.iterrows():
        target_id = str(row[0])
        if id == target_id:
            return_referece = str(row[3])
    return return_referece

def get_original_cluster_by_id(dataframe, id):
    return_referece = ""
    for index, row in dataframe.iterrows():
        target_id = str(row[0])
        if id == target_id:
            return_referece = str(row[2])
    return return_referece

# utility function for computing entropy
def get_value_int_id(vocab,col):
    for index, item in enumerate(vocab):
        if col == item:
            return index


if __name__ == '__main__':
    data_file = "S9-By-ClusterID-Then-TrueID.txt"
    # Read data file into a pandas dataframe. I used the memory streaming capability in python to show that this
    # also can be done for large data and still be relatively memory friendly
    class MyCorpus(object):
        def __iter__(self):
            corpus_df = pd.read_csv(data_file, sep=",", header=None, encoding='utf-8')
            self.ground_truth_clusters = []
            for index, row in corpus_df.iterrows():
                self.ground_truth_clusters.append(str(row[2]))
                # assume there's one document per line, tokens separated by whitespace
                yield [w.lower() for w in str(row[3]).split()]

    corpus = MyCorpus()

    # Training Word2Vec is mode suitable I think for this task as it uses the skip gram model with negative sampling
    # which trains a neural network to endocde each unique word in the corpus to a unique vector. The trained vectors
    # end up close to each other in the euclidean space if the records are similai or in other words if they should
    # exist in the same cluster.
    embedding_size = 128
    alpha = 0.05
    window = 20
    workers = 1
    iter = 100
    min_alpha = 0.001
    sg = 1
    hs = 0
    negative = 50
    sample = 0
    compute_loss = True
    sorted_vocab = 1
    min_count = 0
    model = gensim.models.word2vec.Word2Vec(sentences=corpus, size=embedding_size, alpha=alpha, window=window,
                                            workers=workers, iter=iter,
                                            min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, sample=sample,
                                            compute_loss=compute_loss, sorted_vocab=sorted_vocab, min_count=min_count)

    # serialize model to file
    model_file = "er2word2vec.npy"
    model.save(model_file)

    # Load trained model from the saved file
    model = gensim.models.Word2Vec.load(model_file, mmap='r')

    # Visualize leanred embeddings for each record
    visualize_embedding(model, embedding_size, corpus)

    # # Simple test
    # test_doc = ['laverne', 'l', 'delinois', '1439', 'sara', 'ct', 'allen', 'tx', '75002', '972', '412', '0632']
    # c = "61"
    # current_cluster_count = get_number_of_occurances_in_list(c, corpus.ground_truth_clusters)
    # test_doc_vec = infer_vector(model, embedding_size, test_doc)
    # top_k_similar_docs = top_k_similar_documents(model, embedding_size, test_doc, corpus, 10)
    # similar_doc_count = 0
    # for s in top_k_similar_docs:
    #     similar_doc = s[0]
    #     similar_doc_type = s[0][0]
    #     if similar_doc_type == c:
    #         similar_doc_count = similar_doc_count + 1
    # accuracy = (similar_doc_count / (current_cluster_count)) * 100
    # print("Test Accuracy: " + str(accuracy))



    # # Evaluate
    # average_accuracy = evaluate_model(model, embedding_size, corpus)
    # print("Evaluation Accuracy: "  + str(average_accuracy))
    #
    # # Form a proposed graph
    # graph_edges = []
    # source_nodes = []
    # target_nodes = []
    # K = 1
    # print(len(corpus)
    # unclean_dataframe = pd.read_csv(data_file, sep=",", header=None, encoding='utf-8')
    # for d in tqdm(corpus):
    #     top_similar_doc_for_graph = top_k_similar_documents(model, embedding_size, d, corpus, K)
    #     if len(top_similar_doc_for_graph) > 0:
    #         for i in range(0,K):
    #             doc_source_id = get_document_id(unclean_dataframe, d)
    #             doc_target_id = get_document_id(unclean_dataframe, top_similar_doc_for_graph[i][0])
    #             source_nodes.append(doc_source_id)
    #             target_nodes.append(doc_target_id)
    #             graph_edges.append([doc_source_id,doc_target_id])
    # edges_df = pd.DataFrame({"source": source_nodes, "target": target_nodes})
    # graph = sg.StellarGraph(edges=edges_df)
    # G = graph.to_networkx()
    # pickle.dump(G, open("records_networkx", "wb"))
    # G = pickle.load(open("records_networkx", "rb"))
    #
    # # compute the best partition using the community detection algorithm Louvain
    # # https://github.com/taynaud/python-louvain
    # clusters = community_louvain.best_partition(G)
    # pickle.dump(clusters, open("records_clusters", "wb"))
    # clusters = pickle.load(open("records_clusters", "rb"))
    #
    # # Sort clusters
    # sorted_clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: item[1])}
    # print(len(sorted_clusters))
    # # # Compute Entropy
    # # # For each group of clusters I create a matrix of the records. Convert the matrix to ints. Then compute counts and entropy of the whole matrix.
    # # unique_clusters = []
    # # records_clustered = []
    # # for key in tqdm(sorted_clusters):
    # #     reference = get_document_by_id(unclean_dataframe, key)
    # #     original_cluster = get_original_cluster_by_id(unclean_dataframe, key)
    # #     print([key,original_cluster,sorted_clusters[key],reference])
    # #     records_clustered.append([key,original_cluster,sorted_clusters[key],reference])
    # #     unique_clusters.append(sorted_clusters[key])
    # # pickle.dump(records_clustered, open("records_clustered", "wb"))
    # # pickle.dump(unique_clusters, open("unique_clusters", "wb"))
    # records_clustered = pickle.load(open("records_clustered", "rb"))
    # unique_clusters = pickle.load(open("unique_clusters", "rb"))
    # unique_clusters = np.unique(np.array(unique_clusters)).tolist()
    # print(len(records_clustered))
    # print(len(unique_clusters))
    #
    # for uc in tqdm(unique_clusters):
    #     size_of_record_in_cluster = []
    #     current_record_cluster = []
    #     for r in records_clustered:
    #         if int(r[2]) == int(uc):
    #             ref = r[3].split()
    #             current_record_cluster.append(ref)
    #             size_of_record_in_cluster.append(len(ref))
    #     size_of_record_in_cluster.sort(reverse=True)
    #     longest_reference = size_of_record_in_cluster[0]
    #     cleaned_matrix = []
    #     for r2 in current_record_cluster:
    #         if len(r2) < longest_reference:
    #             augmentation_length = longest_reference - len(r2)
    #             for i in range(augmentation_length):
    #                 r2.append('0')
    #         cleaned_matrix.append(r2)
    #     cleaned_matrix = np.array(cleaned_matrix)
    #     vocab = np.unique(cleaned_matrix)
    #     for row_index, row in enumerate(cleaned_matrix):
    #         for col_index, col in enumerate(row):
    #             cleaned_matrix[row_index][col_index] = int(get_value_int_id(vocab,col))
    #     cleaned_matrix = np.array(cleaned_matrix)
    #     cleaned_matrix = cleaned_matrix.astype(np.int)
    #     pd_series = pd.DataFrame(cleaned_matrix)
    #     e = entropy(pd_series.value_counts())
    #     # Create final data file
    #     f = open("record-clusters.csv", "a")
    #     for r in records_clustered:
    #         if int(r[1]) == int(uc):
    #             row_to_write = str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(e) + "," + str(r[3]) + "\n"
    #             f.write(row_to_write)


