from gensim.models import FastText
import pandas as pd

def load_data():
    # Read data file into a pandas dataframe
    data_file = "S9-By-ClusterID-Then-TrueID.txt"
    unclean_dataframe = pd.read_csv(data_file, sep=",", header=None, encoding='utf-8')

    # Prepare data
    record_id = []
    predicted_clusters = []
    ground_truth_clusters = []
    docs = []
    docs_no_ids = []
    for index, row in unclean_dataframe.iterrows():
        id = str(row[0])
        record_id.append(id)
        cluster_id = str(row[1])
        predicted_clusters.append(cluster_id)
        true_id = str(row[2])
        ground_truth_clusters.append(true_id)
        reference = str(row[3])
        reference_tokenized = [str(true_id)] + [w.lower() for w in reference.split()]
        docs.append(reference_tokenized)
        reference_tokenized_no_ids = [w.lower() for w in reference.split()]
        docs_no_ids.append(reference_tokenized_no_ids)
        print(str(index) + " | " + id + " | " + cluster_id + " | " + true_id + " | " + reference)

    return docs


def train_model(training_corpus):
    # sentences = None, corpus_file = None, sg = 0, hs = 0, size = 100, alpha = 0.025, window = 5, min_count = 5,
    # max_vocab_size = None, word_ngrams = 1, sample = 1e-3, seed = 1, workers = 3, min_alpha = 0.0001,
    # negative = 5, ns_exponent = 0.75, cbow_mean = 1, hashfxn = hash, iter = 5, null_word = 0, min_n = 3, max_n = 6,
    # sorted_vocab = 1, bucket = 2000000, trim_rule = None, batch_words = MAX_WORDS_IN_BATCH, callbacks = (),
    # compatible_hash = True
    model = FastText(sentences=training_corpus)

def main():
    training_corpus = load_data()


if __name__ == '__main__':
    main()