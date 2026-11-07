import time
from collections import defaultdict

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

from ConfigER import ConfigER

import re

import chardet
import numpy as np
import scipy.spatial


seed(1)
tf.random.set_seed(2)

np.set_printoptions(threshold=np.inf)


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper

class StagER():
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

    def convert_list_of_chars_to_ascii_codes(self,string_sequence):
        ascii_sequence=[]
        for c in string_sequence:
            ascii_sequence.append(ord(c))
        return ascii_sequence

    def records_to_graph(self):
        with open(self.input_file_name, "r", encoding=self.encoding) as input_file:
            next(input_file)
            self.edges = []
            self.walks = []
            self.documents={}
            for document in input_file:
                first_delimiter = document.find(self.delimiter)
                doc_id = document[0:first_delimiter]
                body = document[first_delimiter + 1:]
                body = body.strip()
                body = body.lower()
                body = body.replace(self.delimiter, ' ')
                body = re.sub(' +', ' ', body)
                body = list(body)
                body=self.convert_list_of_chars_to_ascii_codes(body)
                self.documents[doc_id]=body
                self.walks.append(body)
                document_edges = []
                for i in range(len(body)):
                    if i+1<len(body):
                        c=body[i]
                        next_c=body[i+1]
                    # self.edges.append(str(c)+"|"+str(next_c))
                    document_edges.append(str(c)+"|"+str(next_c))
                self.documents[doc_id] = document_edges

                # print(body)
            print(len(self.edges))
            self.edges=list(set(self.edges))
            print(len(self.edges))
            print(self.edges)

    def cosine_distance(self, vector1, vector2):
        return scipy.spatial.distance.cosine(vector1, vector2)

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
        all_embedding=np.array(all_embedding)
        # print(node, current_vector)
        np.savetxt("character_embedding.tsv", list(all_embedding), delimiter='\t', fmt='%f')
        np.savetxt("node_to_characters.tsv", nodes, delimiter='\t', fmt='%s')

        all_doc_embedding = []
        doc_nodes = []
        self.entity_embedding = {}
        for doc_id,body in self.documents.items():
            doc_nodes.append(doc_id)
            average_vector = np.zeros((64,))
            for c in body:
                cvector = np.array(model.wv[c])
                average_vector = average_vector + cvector
            average_vector = np.array(average_vector/len(body))
            self.entity_embedding[doc_id]=average_vector
            all_doc_embedding.append(average_vector)
        all_doc_embedding = np.array(all_doc_embedding)

        np.savetxt("entity_embedding.tsv", list(all_doc_embedding), delimiter='\t', fmt='%f')
        np.savetxt("node_to_entities.tsv", doc_nodes, delimiter='\t', fmt='%s')


    def find_matches(self):
        pairs = []
        for docid1,dvector1 in self.entity_embedding.items():
            for docid2, dvector2 in self.entity_embedding.items():
                if docid1<=docid2:
                    continue
                cosine_distance=self.cosine_distance(dvector1,dvector2)
                if cosine_distance==0:
                    pairs.append((docid1,docid2))
        print(pairs)












@measure_execution_time
def main():
    stager = StagER(ConfigER.delimiter, ConfigER.input_file_name)
    stager.records_to_graph()
    stager.embed_ascii_graph()
    stager.find_matches()




if __name__ == '__main__':
    main()