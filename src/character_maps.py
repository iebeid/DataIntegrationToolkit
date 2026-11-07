import time
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datasketch import MinHash, MinHashLSH
from numpy.random import seed
from sklearn.manifold import TSNE
from tqdm import tqdm
from gensim.models import Word2Vec
from graph import DirectedGraph
import networkx as nx
import tensorflow as tf
import sklearn as skl
import matplotlib.pyplot as plt
import scipy.spatial
import graph as gl
from EvaluatER import EvaluatER
from ConfigER import ConfigER

import re

import chardet
import numpy as np
import scipy.spatial

np.random.seed(123)
tf.random.set_seed(123)

tf.config.threading.set_inter_op_parallelism_threads(8)

np.set_printoptions(threshold=np.inf)


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


class CharacterMaps():
    def __init__(self, delimiter, input_file_name):
        print("----STAGING----")
        self.delimiter = delimiter
        self.input_file_name = input_file_name
        self.encoding = None

        with open(self.input_file_name, "rb") as rawdata:
            result = chardet.detect(rawdata.read())
            self.encoding = result["encoding"]
            if self.encoding == "ascii":
                self.encoding = "utf-8"

    def convert_list_of_chars_to_ascii_codes(self, string_sequence):
        ascii_sequence = []
        for c in string_sequence:
            ascii_sequence.append(str(ord(c)))
        return ascii_sequence

    def create_character_map(self):
        with open(self.input_file_name, "r", encoding=self.encoding) as input_file:
            next(input_file)
            self.edges = []
            self.walks = []
            self.documents = {}
            for document in input_file:
                first_delimiter = document.find(self.delimiter)
                doc_id = document[0:first_delimiter]
                body = document[first_delimiter + 1:]
                body = body.strip()
                body = body.replace(self.delimiter, ' ')
                body = re.sub(' +', ' ', body)
                body = list(body)
                body = self.convert_list_of_chars_to_ascii_codes(body)
                self.documents[doc_id] = body
                self.walks.append(body)
                for i in range(len(body)):
                    if i + 1 < len(body):
                        c = body[i]
                        next_c = body[i + 1]
                    self.edges.append(str(c) + "|" + str(next_c))
            print(len(self.edges))
            edge_counter = dict(Counter(self.edges))
            print(edge_counter)
            print(len(edge_counter))
            all_source_nodes = []
            all_target_nodes = []
            all_nodes = []
            self.out_edges = {}
            for e in self.edges:
                nodes = e.split("|")
                s_n = nodes[0]
                t_n = nodes[1]
                self.out_edges[s_n] = t_n
                all_source_nodes.append(s_n)
                all_target_nodes.append(t_n)
                all_nodes.append(s_n)
                all_nodes.append(t_n)
            # initializing empty list as Values
            self.in_edges = {value: key for key, value in self.out_edges.items()}
            print(self.in_edges)
            print(len(self.in_edges))
            all_unique_nodes = np.unique(all_nodes)
            N = len(all_unique_nodes)
            print("All unique nodes: " + str(N))
            print("All unique source nodes: " + str(len(list(np.unique(all_source_nodes)))))
            print("All unique target nodes: " + str(len(list(np.unique(all_target_nodes)))))
            self.node_index = {}
            i = 0
            for n in all_unique_nodes:
                self.node_index[n] = int(i)
                i = i + 1
            node_counter = dict(Counter(all_source_nodes))
            print(node_counter)
            print(edge_counter)
            c = 0
            self.edges = []
            for k, v in edge_counter.items():
                nodes = k.split("|")
                s_n = str(nodes[0])
                t_n = str(nodes[1])
                source_node_total_edge_count = float(node_counter[s_n])
                specific_edge_count = float(v)
                transition_probabilty_per_unique_edge = float(specific_edge_count / source_node_total_edge_count)
                self.edges.append((s_n, t_n, transition_probabilty_per_unique_edge))
                c = c + 1

    def cosine_distance(self, vector1, vector2):
        return scipy.spatial.distance.cosine(vector1, vector2)

    def embed_entities(self, character_graph, character_embedding):
        self.entity_embeddings = {}
        for doc_id, body in self.documents.items():
            converted_body = []
            for c in body:
                character_index = character_graph.nodes[c]
                converted_body.append(character_index)
            found_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(character_embedding), converted_body)
            entity_embedding_tensor = tf.math.reduce_mean(found_embedding, axis=0)
            self.entity_embeddings[doc_id] = entity_embedding_tensor.numpy()

    def embed_ascii_graph(self):
        model = Word2Vec(self.walks, vector_size=64, window=5, negative=10,
                         min_count=0,
                         sg=1, workers=8, epochs=10, sample=0)
        all_embedding = []
        nodes = []
        for node in tqdm(model.wv.index_to_key):
            current_vector = np.array(model.wv[node])
            all_embedding.append(current_vector)
            nodes.append(node)
        all_embedding = np.array(all_embedding)
        np.savetxt("character_embedding.tsv", list(all_embedding), delimiter='\t', fmt='%f')
        np.savetxt("node_to_characters.tsv", nodes, delimiter='\t', fmt='%s')

        all_doc_embedding = []
        doc_nodes = []
        self.entity_embedding = {}
        for doc_id, body in self.documents.items():
            doc_nodes.append(doc_id)
            average_vector = np.zeros((64,))
            for c in body:
                cvector = np.array(model.wv[c])
                average_vector = average_vector + cvector
            average_vector = np.array(average_vector / len(body))
            self.entity_embedding[doc_id] = average_vector
            all_doc_embedding.append(average_vector)
        all_doc_embedding = np.array(all_doc_embedding)

        np.savetxt("entity_embedding.tsv", list(all_doc_embedding), delimiter='\t', fmt='%f')
        np.savetxt("node_to_entities.tsv", doc_nodes, delimiter='\t', fmt='%s')

    def find_matches(self, threshold):
        pairs = []
        for docid1, dvector1 in self.entity_embedding.items():
            for docid2, dvector2 in self.entity_embedding.items():
                if docid1 <= docid2:
                    continue
                cosine_distance = self.cosine_distance(dvector1, dvector2)
                if cosine_distance > 0 and cosine_distance < threshold:
                    pairs.append((docid1, docid2))
        return pairs

    def evaluate_character_maps(self, graph, model, threshold, ground_truth_file, log_file):
        self.embed_entities(graph, model)
        EvaluatER(list(nx.algorithms.components.connected_components(nx.Graph(self.find_matches(threshold)))),
                  ground_truth_file).run(open(log_file, "w"))


@measure_execution_time
def main():
    stager = CharacterMaps(ConfigER.delimiter, ConfigER.input_file_name)
    character_graph = stager.create_character_map()
    print(character_graph)


if __name__ == '__main__':
    main()
