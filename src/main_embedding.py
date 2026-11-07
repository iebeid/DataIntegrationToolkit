import time
import random
from datasketch import MinHash, MinHashLSH
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
import networkx as nx
import GraphOG_Math
from sklearn.cluster import AgglomerativeClustering
from EvaluatER import EvaluatER
from GraphOG import GraphOG
from SimilER import SimilER
from StagER import StagER
import pprint
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt


def ModER():
    print("Welcome to ModER v0.1")
    # ------------------CONFIGURATION------------------------
    input_file_name = "data/S2G.txt"
    # input_file_name = "data/S22Affiliations.txt"
    # input_file_name = "data/S6GeCo.txt"
    # input_file_name = "data/S18PX.txt"
    truthFileName = "eval/truthABCgoodDQ.txt"
    # truthFileName = "eval/truthAffiliations.txt"
    # truthFileName = "eval/truthGeCo.txt"
    # truthFileName = "eval/truthABCpoorDQ.txt"
    delimiter = ","
    number_of_hashing_permutations = 128
    percentile_of_stop_words = 95.0
    estimated_jaccard_similairty_threshold = 0.5

    modularity_walk_length = 20
    skipgram_epochs = 100
    skipgram_embedding_size = 24
    skipgram_context_window = 3
    skipgram_negative_samples = 9

    global_encoding = "utf-8"
    # ------------------PREPROCESS FILE------------------------
    stager = StagER(delimiter, input_file_name, percentile_of_stop_words)
    stager.generate_unqiue_corpus()

    # ------------------LOCALITY SENSTIVE HASHING------------------------
    lsh = MinHashLSH(threshold=estimated_jaccard_similairty_threshold, num_perm=number_of_hashing_permutations)
    inverted_hash_table = {}
    for rec_id,document in enumerate(stager.token_corpus):
        m = MinHash(num_perm=number_of_hashing_permutations)
        for d in document:
            m.update(d.encode(global_encoding))
        lsh.insert(str(rec_id), m)
        inverted_hash_table[str(rec_id)] = m
    # ------------------MATCHING------------------------
    edges = []
    for docid1, m in tqdm(inverted_hash_table.items()):
        similar_documents = lsh.query(m)
        for docid2 in similar_documents:
            if docid1<docid2:
                jc = m.jaccard(inverted_hash_table[docid2])
                if jc>0:
                    edges.append((docid1,docid2,jc))


    graph = GraphOG(edges, undirected=True, weighted=True, link_single_nodes=True, simple_graph=True)
    graph.info()
    components = graph.algorithms.akef_breadth_first_connected_component_detection()



    walks = graph.algorithms.akef_weighted_biased_random_walker(modularity_walk_length)
    print(len(walks))
    print(len(inverted_hash_table.keys()))

    model = Word2Vec(sentences=walks, vector_size=skipgram_embedding_size, window=skipgram_context_window,
                     negative=skipgram_negative_samples, min_count=1,
                     sg=1, workers=1, epochs=skipgram_epochs, sample=0)
    node_embeddings = np.array(model.wv.vectors)
    node_ids = model.wv.index_to_key
    # model.wv.most_similar()
    new_edges = []
    for n in node_ids:
        s = model.wv.most_similar(n,topn=1)
        for i in s:
            print(n + "-->" + str(i[0]))
            ed = [str(n),str(i[0])]
            ed.sort()
            new_edges.append((ed[0],ed[1],i[1]))
    new_edges = list(set(new_edges))
    graph = GraphOG(new_edges, undirected=True, weighted=True, link_single_nodes=True, simple_graph=True)
    graph.info()
    components = graph.algorithms.akef_breadth_first_connected_component_detection()
    for cc in components:
        if len(cc)>1:
            edges = []
            for n in cc:
                for e in graph.edges:
                    n1 = e[0]
                    n2 = e[1]
                    w = e[2]
                    if (n == n1 and n1 in cc) or (n == n2 and n2 in cc):
                        n1 = stager.record_inverted_index[int(n1)]
                        n2 = stager.record_inverted_index[int(n2)]
                        ed = (n1, n2, w)
                        edges.append(ed)
            print(cc)
            print(edges)
            print("--------------------")



    # initial_k_vectors = []
    # for cc in components:
    #     if len(cc)>1:
    #         vectors = []
    #         for n in cc:
    #             n=graph.node_index[n]
    #             v=node_embeddings[n]
    #             vectors.append(v)
    #         mean_k = list(np.array([np.mean(np.array(vectors),axis=0)])[0])
    #         # max_k = list(np.array([np.max(np.array(vectors), axis=0)])[0])
    #         # min_k = list(np.array([np.min(np.array(vectors), axis=0)])[0])
    #         initial_k_vectors.append(mean_k)
    #         # initial_k_vectors.append(max_k)
    #         # initial_k_vectors.append(min_k)
    # initial_k_vectors = np.array(initial_k_vectors)
    # # centroid, labels = kmeans2(node_embeddings,50, minit='random')
    # # centroid, labels = kmeans2(node_embeddings, 50, minit='points')
    # # centroid, labels = kmeans2(node_embeddings, 50, minit='++')
    # centroid, labels = kmeans2(node_embeddings, k=initial_k_vectors, minit='matrix',seed=1)
    # # centroid, labels = kmeans(whiten(tfidf_matrix.toarray()),number_of_blocks)
    # # print(centroid)
    # # print(labels)
    #
    #
    # record_labels2 = {}
    # for (row, label) in enumerate(labels):
    #     record_labels2[node_ids[row]]=label
    # # print(record_labels2)
    #
    # # suggested_changes = graph.algorithms.akef_modularity_matrix_method_2()
    # # for sc in suggested_changes:
    # #     i = sc[0]
    # #     j = sc[1]
    # #     i_label = graph.node_label[i]
    # #     j_label = graph.node_label[j]
    # #     graph.node_label[i] = j_label
    # graph.update_node_labels(record_labels2)
    # components = graph.node_labels_to_components()





    # ------------------EVALUATION------------------------
    all_components_nodes = set()
    for cc in components:
        for n in cc:
            all_components_nodes.add(n)
    singles = list(set.difference(set(inverted_hash_table.keys()),all_components_nodes))
    for s in singles:
        components.append([s])
    clusters = []
    for cc in components:
        component = []
        for r in cc:
            component.append(stager.record_inverted_index[int(r)])
        clusters.append(component)
    print(clusters)
    evaluator_final = EvaluatER(clusters, truthFileName)
    evaluator_final.run()
    print("Modularity: " + str(graph.tasks.compute_modularity()))










def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


@measure_execution_time
def main():
    ModER()


if __name__ == '__main__':
    main()
