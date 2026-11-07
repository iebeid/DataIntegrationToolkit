
# Imports
import pandas as pd
from tqdm import tqdm
import stellargraph as sg
import numpy as np

from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import os


from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt



# Jaccard similarity
def jaccard(s1,s2):
    a = set(s1)
    b = set(s2)
    if len(a) == 0 and len(b) == 0:
        return 0
    return float(len(a&b) / len(a|b))


#Lists all the overlapping ngrams in a string (similar to a sliding window)
def ngrams(seq, n):
    return [seq[i:i+n] for i in range(1+len(seq)-n)]


#Sliding window
def window(fseq, window):
    for i in range(len(fseq) - window + 1):
        yield fseq[i:i+window]


def main():
    data_file = "S1-tokenized.txt"
    records_df = pd.read_csv(data_file, sep="\t", header=None, encoding='utf-8')

    jaccard_coefficient_threshold = 0.25
    bigram_sampling_window_size = 2
    ngram_size = 2

    source_nodes = []
    target_nodes = []
    edge_weights = []
    for index1, row1 in tqdm(records_df.iterrows()):
        record_id = row1[0].split()[0]
        reference_tokenized = [token.lower() for token in row1[0].split()[1:]]
        reference = ' '.join(reference_tokenized)
        n1 = ngrams(reference, ngram_size)
        s1 = [''.join(bigram_seq) for bigram_seq in window(n1, bigram_sampling_window_size)]
        for index2, row2 in records_df.iterrows():
            record_id_2 = row2[0].split()[0]
            reference_tokenized_2 = [token.lower() for token in row2[0].split()[1:]]
            reference_2 = ' '.join(reference_tokenized_2)
            n2 = ngrams(reference_2, ngram_size)
            s2 = [''.join(bigram_seq) for bigram_seq in window(n2, bigram_sampling_window_size)]
            jaccard_coefficient = jaccard(s1,s2)
            if jaccard_coefficient > jaccard_coefficient_threshold:
                if record_id != record_id_2:
                    source_nodes.append(record_id)
                    target_nodes.append(record_id_2)
                    edge_weights.append(jaccard_coefficient)

    unique_nodes = list(np.unique(np.array(source_nodes + target_nodes)))
    high = (1 / (np.sqrt(128)))
    low = -high
    node_features_list = []
    for n in unique_nodes:
        node_features_list.append(
            np.random.uniform(low, high, (1, 128))[0].tolist())

    nodes_df = pd.DataFrame(node_features_list, index=unique_nodes)

    edges_df = pd.DataFrame({"source": source_nodes, "target": target_nodes, "weight": edge_weights})
    graph = sg.StellarGraph(nodes={"One": nodes_df},edges=edges_df)

    print(graph.info())
    node_labels = []
    node_subjects = []
    label = 1
    for c in graph.connected_components():
        print(c)
        for node_id in c:
            node_labels.append((node_id,label))
            node_subjects.append(label)
        label = label + 1

    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=10, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=20, test_size=None, stratify=test_subjects
    )

    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    generator = FullBatchNodeGenerator(graph, method="gcn")

    node_ids = list(train_subjects.index())

    train_gen = generator.flow(node_ids, train_targets)

    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    x_inp, x_out = gcn.in_out_tensors()
    predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    val_gen = generator.flow(val_subjects.index, val_targets)
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=200,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )

    sg.utils.plot_history(history)

    test_gen = generator.flow(test_subjects.index, test_targets)

    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    all_nodes = node_subjects.index
    all_gen = generator.flow(all_nodes)
    all_predictions = model.predict(all_gen)

    node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

    df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
    df.head(20)

    embedding_model = Model(inputs=x_inp, outputs=x_out)


    emb = embedding_model.predict(all_gen)
    emb.shape

    transform = TSNE  # or PCA

    X = emb.squeeze(0)
    X.shape

    trans = transform(n_components=2)
    X_reduced = trans.fit_transform(X)
    X_reduced.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=node_subjects.astype("category").cat.codes,
        cmap="jet",
        alpha=0.7,
    )
    ax.set(
        aspect="equal",
        xlabel="$X_1$",
        ylabel="$X_2$",
        title=f"{transform.__name__} visualization of GCN embeddings for cora dataset",
    )

if __name__ == '__main__':
    main()