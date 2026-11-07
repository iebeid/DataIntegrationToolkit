from gensim.models import FastText
from gensim.models import TfidfModel
import pandas as pd
from gensim import corpora
from pprint import pprint
import numpy as np
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

def load_data():
    # Read data file into a pandas dataframe
    data_file = "S9-By-ClusterID-Then-TrueID.txt"
    unclean_dataframe = pd.read_csv(data_file, sep=",", header=None, encoding='utf-8')

    # Prepare data
    record_id = []
    predicted_clusters = []
    ground_truth_clusters = []
    docs = []
    untokenized_docs = []
    docs_no_ids = []
    for index, row in unclean_dataframe.iterrows():
        id = str(row[0])
        record_id.append(id)
        cluster_id = str(row[1])
        predicted_clusters.append(cluster_id)
        true_id = str(row[2])
        ground_truth_clusters.append(true_id)
        reference = str(row[3])
        # reference_tokenized = [str(true_id)] + [w.lower() for w in reference.split()]
        reference_tokenized = [w.lower() for w in reference.split()]
        docs.append(reference_tokenized)
        untokenized_reference = reference.lower()
        untokenized_docs.append(untokenized_reference)
        reference_tokenized_no_ids = [w.lower() for w in reference.split()]
        docs_no_ids.append(reference_tokenized_no_ids)
        # print(str(index) + " | " + id + " | " + cluster_id + " | " + true_id + " | " + reference)

    return docs


def train_model(training_corpus):
    mydict = corpora.Dictionary(training_corpus)
    corpus = [mydict.doc2bow(line) for line in training_corpus]
    for doc in corpus:
        print([[mydict[id], freq] for id, freq in doc])
    tfidf = TfidfModel(corpus, smartirs='ntc')
    print(len(tfidf[corpus]))
    for doc in tfidf[corpus]:
        print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

def main():
    training_corpus = load_data()
    train_model(training_corpus)


if __name__ == '__main__':
    main()