import time

import Levenshtein
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from ConfigER import ConfigER
from EvaluatER import EvaluatER
from FiltER import FiltER
from GraphOG import GraphOG
from StagER import StagER


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
    log_file = open(ConfigER.log_filename, "w")
    stager = StagER(ConfigER.delimiter, ConfigER.input_file_name, ConfigER.percentile_of_stop_words,
                    ConfigER.percentile_of_blocking_words, ConfigER.percentile_of_discriminate_words_lengths)
    stager.generate_word_corpus()
    print("Beta:" + str(stager.beta))
    print("Beta:" + str(stager.beta), file=log_file)
    print("Sigma: " + str(stager.sigma))
    print("Sigma: " + str(stager.sigma), file=log_file)
    print("Delta: " + str(stager.delta))
    print("Delta: " + str(stager.delta), file=log_file)
    # --------------LOCALITY SENSITIVE HASHING------------
    # similer = SimilER(stager.record_inverted_index, stager.token_corpus, lsh_permutations, stager.encoding)
    # similer.run_locality_sensitive_hashing()

    lsh = MinHashLSH(threshold=0.5, num_perm=ConfigER.lsh_permutations)
    inverted_hash_table = {}
    for doc_id, document in tqdm(stager.filtered_unique_tokenized_corpus.items()):
        m = MinHash(num_perm=ConfigER.lsh_permutations)
        for word in document:
            m.update(word.encode(stager.encoding))
        lsh.insert(doc_id, m)
        inverted_hash_table[doc_id] = m
    # -------------------BLOCKING-------------------------
    # filter = FiltER(stager.encoding, stager.beta, stager.word_count_dict, stager.filtered_unique_tokenized_corpus)
    # filter.run_frequency_blocking()
    edges = []
    for doc1, b1 in tqdm(stager.blocking_corpus.items()):
        b1=set(b1)
        for doc2, b2 in stager.blocking_corpus.items():
            if doc1>=doc2:
                continue

            # inter = set.intersection(set(b1),set(b2))
            # if len(inter)>0:
            #
            b2 = set(b2)
            if set.isdisjoint(b1,b2):
                continue

            edges.append((doc1, doc2, 1.0))
    edges.sort()
    # -----------------TRANSITIVE CLOSURE-----------------
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
    final_components = components
    # ------------------ENTITY PROFILING------------------
    final_components = []
    global_edges = []
    for cc in tqdm(components):
        if len(cc) > 1:
            cc_edges = []
            for doc1 in cc:
                d1 = set(stager.discriminative_word_corpus[doc1])
                for doc2 in cc:
                    if doc1 < doc2:
                        d2 = set(stager.discriminative_word_corpus[doc2])
                        s = 0
                        intersect = len(set.intersection(d1, d2))
                        # intersect = set()
                        # for i1 in d1:
                        #     for i2 in d2:
                        #         if (i1 == i2) or (i1 in i2) or (i2 in i1) or Levenshtein.distance(i1,i2)<=2:
                        #             intersect.add(i1)
                        if intersect > 0:
                            s = 1.0
                        else:
                            s = inverted_hash_table[doc1].jaccard(inverted_hash_table[doc2])
                        if s > 0:
                            cc_edges.append((doc1, doc2, s))
                            global_edges.append((doc1, doc2, s))
            if len(cc_edges) > 0:
                cc_nodes = {}
                l = 0
                for doc in cc:
                    cc_nodes[doc] = {"real_id": stager.document_index[doc], "type": "r",
                                     "attribute": [stager.filtered_unique_tokenized_corpus[doc],
                                                   stager.discriminative_word_corpus[doc]],
                                     "label": int(l)}
                    l = l + 1
                cc_graph = GraphOG(cc_edges, nodes=cc_nodes, undirected=True, weighted=True, link_single_nodes=False,
                                   simple_graph=True)
                further_components = cc_graph.algorithms.akef_modified_brim_bipartite_cluster_detection()
                further_components = cc_graph.algorithms.akef_modularity_maximizer()
                for ccf in further_components:
                    final_components.append(ccf)
            else:
                for d in cc:
                    final_components.append([d])
        else:
            for d in cc:
                final_components.append([d])
    # ------------------EVALUATION------------------------
    # Convert IDS and add singles
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
    final_graph = graph.recast_graph(global_edges, link_single_nodes=True, simple_graph=True)
    final_graph.update_node_labels_with_components(final_components)
    final_graph.info(log_file)
    log_file.close()


if __name__ == '__main__':
    main()
