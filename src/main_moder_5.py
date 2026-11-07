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
    print("Beta: " + str(stager.beta))
    print("Beta: " + str(stager.beta), file=log_file)
    print("Sigma: " + str(stager.sigma))
    print("Sigma: " + str(stager.sigma), file=log_file)
    print("Delta: " + str(stager.delta))
    print("Delta: " + str(stager.delta), file=log_file)
    # --------------LOCALITY SENSITIVE HASHING------------
    main_lsh = MinHashLSH(threshold=0.5, num_perm=ConfigER.lsh_permutations)
    main_inverted_hash_table = {}
    for id, document in tqdm(stager.filtered_unique_tokenized_corpus.items()):
        m = MinHash(num_perm=ConfigER.lsh_permutations)
        for word in document:
            m.update(word.encode(stager.encoding))
        main_lsh.insert(id, m)
        main_inverted_hash_table[id] = m
    discriminate_lsh = MinHashLSH(threshold=0.5, num_perm=ConfigER.lsh_permutations)
    discriminate_inverted_hash_table = {}
    for id, document in tqdm(stager.signature_corpus.items()):
        m = MinHash(num_perm=ConfigER.lsh_permutations)
        for word in document:
            m.update(word.encode(stager.encoding))
        discriminate_lsh.insert(id, m)
        discriminate_inverted_hash_table[id] = m
    # -----------BLOCKING---------------

    # --------------SIMILARITY-------------------
    edges = []
    doc_similarity = {}
    for doc1, real1 in tqdm(stager.document_index.items()):
        m1 = main_inverted_hash_table[doc1]
        d1 = discriminate_inverted_hash_table[doc1]
        similar_docs = []
        for doc2, real2 in stager.document_index.items():
            if doc2>=doc1:
                continue
            m2 = main_inverted_hash_table[doc2]
            d2 = discriminate_inverted_hash_table[doc2]
            dj = d1.jaccard(d2)
            mj = m1.jaccard(m2)
            if dj>0.5:
                similar_docs.append(doc2)

            # if dj == 0:
            #     if mj >= 0.9:
            #         edges.append((doc1,doc2,1.0))
            #     elif 0.5 <= mj < 0.9:
            #         edges.append((doc1,doc2,0.5))
            # else:
            #     if dj >= 0.9:
            #         edges.append((doc1, doc2, 1.0))
            #     elif 0 < dj < 0.9:
            #         if mj >= 0.9:
            #             edges.append((doc1, doc2, 1.0))
            #         elif 0.5 <= mj < 0.9:
            #             edges.append((doc1, doc2, 0.5))
        doc_similarity[doc1] = similar_docs
        for d in similar_docs:
            edges.append((doc1,d,1.0))
    # -------------TRANSITIVE CLOSURE-------------
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
    graph.update_node_labels_with_components(final_components)
    graph.info(log_file)
    log_file.close()

if __name__ == '__main__':
    main()