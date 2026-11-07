import time

import Levenshtein
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from ConfigER import ConfigER
from EvaluatER import EvaluatER
from FiltER import FiltER
from GraphOG import GraphOG
from StagER import StagER
from SimilER import SimilER


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper



@measure_execution_time
def main():
    # ------------------PREPROCESS FILE-------------------
    print("------------------PREPROCESS FILE-------------------")
    log_file = open(ConfigER.log_filename, "w")
    stager = StagER(ConfigER.delimiter, ConfigER.input_file_name, ConfigER.percentile_of_stop_words,
                    ConfigER.percentile_of_blocking_words, ConfigER.percentile_of_discriminate_words_lengths)
    stager.generate_word_corpus()
    print("Beta: " + str(stager.beta))
    print("Beta: " + str(stager.beta), file=log_file)
    print("Sigma: " + str(stager.sigma))
    print("Sigma: " + str(stager.sigma), file=log_file)
    print("Delta: " + str(stager.delta))
    print("Delta: " + str(stager.delta), file=log_file)
    # --------------LOCALITY SENSITIVE HASHING------------
    print("--------------LOCALITY SENSITIVE HASHING------------")
    main_lsh = MinHashLSH(threshold=0.9, num_perm=ConfigER.lsh_permutations)
    main_inverted_hash_table = {}
    for id, document in tqdm(stager.filtered_unique_tokenized_corpus.items()):
        m = MinHash(num_perm=ConfigER.lsh_permutations)
        for word in document:
            m.update(word.encode(stager.encoding))
        main_lsh.insert(id, m)
        main_inverted_hash_table[id] = m
    discriminate_lsh = MinHashLSH(threshold=0.0, num_perm=ConfigER.lsh_permutations)
    discriminate_inverted_hash_table = {}
    for id, document in tqdm(stager.discriminative_word_corpus.items()):
        m = MinHash(num_perm=ConfigER.lsh_permutations)
        for word in document:
            m.update(word.encode(stager.encoding))
        discriminate_lsh.insert(id, m)
        discriminate_inverted_hash_table[id] = m
    # ------------Filter--------------------
    print("------------Filter--------------------")
    edges = []
    for doc1, m1 in tqdm(main_inverted_hash_table.items()):
        similar_docs = main_lsh.query(m1)
        for doc2 in similar_docs:
            if doc1>=doc2:
                continue
            m2 = main_inverted_hash_table[doc2]
            s = m1.jaccard(m2)
            if s>0:
                edges.append((doc1,doc2,1.0))
    # -------------TRANSITIVE CLOSURE-----------------------
    print("-------------TRANSITIVE CLOSURE-----------------------")
    nodes = {}
    l = 0
    for doc_id, real_id in stager.document_index.items():
        nodes[doc_id] = {"real_id": real_id, "type": "r",
                         "attribute": [stager.filtered_unique_tokenized_corpus[doc_id],
                                       stager.discriminative_word_corpus[doc_id]],
                         "label": int(l)}
        l = l + 1
    graph = GraphOG(edges, nodes=nodes, undirected=True, weighted=True, link_single_nodes=True, simple_graph=True)
    graph.info(log_file)
    components = graph.algorithms.talburt_transitive_closure_connected_component_detection()
    # ----------------SECOND STAGE ----------------------------
    print("----------------SECOND STAGE ----------------------------")
    second_stage_components = []
    final_components = []
    second_stage_edges = []
    for cc in tqdm(components):
        if len(cc)==1:
            second_stage_components.append(cc[0])
        else:
            final_components.append(cc)
    for doc1 in tqdm(second_stage_components):
        bc1 = stager.blocking_corpus[doc1]
        for doc2 in second_stage_components:
            if doc1 >= doc2:
                continue
            bc2 = stager.blocking_corpus[doc2]
            if set.isdisjoint(set(bc1),set(bc2)):
                continue
            second_stage_edges.append((doc1,doc2,1.0))
    print("-------------TRANSITIVE CLOSURE 2-----------------------")
    final_nodes = {}
    l = 0
    for doc_id in second_stage_components:
        final_nodes[doc_id] = {"real_id": stager.document_index[doc_id], "type": "r",
                         "attribute": [stager.filtered_unique_tokenized_corpus[doc_id],
                                       stager.discriminative_word_corpus[doc_id]],
                         "label": int(l)}
        l = l + 1
    graph = GraphOG(second_stage_edges, nodes=final_nodes, undirected=True, weighted=True, link_single_nodes=True, simple_graph=True)
    graph.info(log_file)
    components = graph.algorithms.talburt_transitive_closure_connected_component_detection()
    # similer = SimilER(stager.document_index, stager.filtered_unique_tokenized_corpus, ConfigER.lsh_permutations, stager.encoding)
    final_edges = []
    for cc in tqdm(components):
        if len(cc)==1:
            final_components.append(cc)
        else:
            nodes = {}
            l = 0
            edges = []
            for doc1 in cc:
                nodes[doc1] = {"real_id": stager.document_index[doc1], "type": "r",
                                 "attribute": [stager.filtered_unique_tokenized_corpus[doc1],
                                               stager.discriminative_word_corpus[doc1]],
                                 "label": int(l)}
                l = l + 1
                # dc1 = stager.discriminative_word_corpus[doc1]
                m1 = discriminate_inverted_hash_table[doc1]
                for doc2 in cc:
                    if doc1 >= doc2:
                        continue
                    # dc2 = stager.discriminative_word_corpus[doc2]
                    # s = similer.scoring_matrix(dc1,dc2)
                    # s_std = similer.scoring_matrix_std(dc1,dc2)
                    # m2 = main_inverted_hash_table[doc2]
                    # s_j = m1.jaccard(m2)
                    m2 = discriminate_inverted_hash_table[doc2]
                    s = m1.jaccard(m2)
                    if s > 0.0:
                        edges.append((doc1,doc2,s))
                        final_edges.append((doc1,doc2,s))
            if len(edges)>0:
                graph = GraphOG(edges, nodes=nodes, undirected=True, weighted=True, link_single_nodes=True,
                                simple_graph=True)
                # further_components = graph.algorithms.networkx_louvain_community_detection()
                further_components = graph.algorithms.akef_modified_brim_bipartite_cluster_detection()
                further_components = graph.algorithms.networkx_louvain_community_detection()
                # further_components = graph.algorithms.akef_modularity_maximizer()
                for ccf in further_components:
                    final_components.append(ccf)
            else:
                final_components.append(cc)



    # ------------------EVALUATION------------------------
    print("------------------EVALUATION------------------------")
    components = []
    for ccf in final_components:
        intermdeiate = []
        for n in ccf:
            node_real_id = stager.document_index[n]
            intermdeiate.append(node_real_id)
        components.append(intermdeiate)
    evaluator_final = EvaluatER(components, ConfigER.truth_filename)
    evaluator_final.run(log_file)
    # Modularity
    # graph = GraphOG(final_edges, nodes=final_nodes, undirected=True, weighted=True, link_single_nodes=True,
    #                 simple_graph=True)
    # graph.update_node_labels_with_components(final_components)
    # graph.info(log_file)
    log_file.close()

if __name__ == '__main__':
    main()