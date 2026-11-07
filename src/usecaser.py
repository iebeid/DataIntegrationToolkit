import numpy as np

from importer import *
from grapher import *
from model4 import *
word = 'computer'
model = Word2Vec(sentences=common_texts, vector_size=16, window=5, min_count=1, workers=4, epochs=500)
vector = model.wv[word]
print(common_texts)
print(vector)
edges = []
for s in common_texts:
    for w in s:
        last_char = None
        for i in range(len(w)):
            if i+1 < len(w):
                c_current = w[i]
                c_next = w[i+1]
                last_char = ord(str(c_next))
                edges.append(str(ord(str(c_current)))+"|"+str(ord(str(c_next))))
        edges.append(str(last_char)+"|"+str(ord(" ")))
print(edges)
unique_edges = dict(Counter(edges))
filtered_edges = []
for key, value in unique_edges.items():
    nodes = key.split("|")
    filtered_edges.append((str(nodes[0]), str(nodes[1]), int(value)))
print(filtered_edges)
graph = DirectedGraph(edges=filtered_edges)
print("Loaded number of nodes: ", graph.number_of_nodes)
print("Loaded number of edges: ", graph.number_of_edges)
graph.create_node_labels(True)
graph.create_node_features(None, 64)

hp = Hyperparameters()
hp.add("trials", 1)
hp.add("K", 1)
hp.add("number_of_epochs", 500)
hp.add("split", 70)
hp.add("batch_size", graph.number_of_nodes)
hp.add("regularization_rate", 0.0005)
hp.add("learning_rate", 0.001)
hp.add("dropout_rate", 0.5)
hp.add("layer_1_dimension", 16)
hp.add("layer_2_dimension", graph.number_of_classes)

# Model 4
all_accuracies = []
for _ in range(hp.get("trials")):
    valid_embeddings = []
    valid_losses = []
    valid_accuracies = []
    valid_predictions = []
    valid_parameters = []
    epochs = []
    losses = []
    accuracies = []
    for k in range(hp.get("K")):
        print(str(hp.get("K")) + " fold validation iteration number: " + str(k))
        m1 = Model4(hp, graph, True)
        m1.prepare()
        m1.compile()
        m1.train()
        m1.validate()
        for sublist in m1.predictions:
            if np.isnan(sublist[0]):
                print("The list contains a NaN")
                break
            else:
                valid_predictions.append(m1.predictions)
                valid_losses.append(m1.valid_loss)
                valid_accuracies.append(m1.valid_accuracy)
                valid_embeddings.append(m1.embedding)
                valid_parameters.append(m1.parameters)
        for e in m1.epochs:
            epochs.append(e.numpy())
        for l in m1.losses:
            losses.append(l.numpy())
        for a in m1.accuracies:
            accuracies.append(a.numpy())
    if len(valid_accuracies) != 0:
        max_valid_accuracy_index = np.argmax(np.array(valid_accuracies))
        final_predictions = list(valid_predictions[max_valid_accuracy_index].numpy())
        final_loss = valid_losses[max_valid_accuracy_index].numpy()
        final_accuracy = valid_accuracies[max_valid_accuracy_index].numpy()
        final_embedding = list(valid_embeddings[max_valid_accuracy_index].numpy())
        final_parameters = valid_parameters[max_valid_accuracy_index]
    else:
        final_predictions = [0]
        final_loss = 0
        final_accuracy = 0
        final_embedding = 0
        final_parameters = [0]
    print("Final Loss: " + str(final_loss))
    print("Final Accuracy: " + str(final_accuracy))
    all_accuracies.append(float(final_accuracy))
    if final_embedding != 0:
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(final_embedding))
        x = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        colors = []
        for n, label in graph.nodes.items():
            colors.append(label)
        embedding_df = pd.DataFrame({"x": x, "y": y})
        losses_df = pd.DataFrame({"epochs": epochs, "losses": losses})
        accuracies_df = pd.DataFrame({"epochs": epochs, "accuracies": accuracies})
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle('Model 4 Visualization')
        sns.scatterplot(ax=axes[0], data=embedding_df, x='x', y='y', hue=colors)
        sns.lineplot(ax=axes[1], data=losses_df, x='epochs', y='losses')
        sns.lineplot(ax=axes[2], data=accuracies_df, x='epochs', y='accuracies')
        plt.show()

embedding_lookup = {}
for key, value in graph.node_index.items():
    char = str(chr(int(key)))
    index = int(value)
    vec = final_embedding[index]
    embedding_lookup[char] = vec


print(model.wv.key_to_index)
print(model.wv.index_to_key)
predicted_vectors = {}
for token in model.wv.index_to_key:
    token_vectors = []
    for c in token:
        token_vectors.append(np.array(embedding_lookup[c]))

    predicted_vector = np.mean(np.array(token_vectors), axis=0)
    predicted_vectors[token] = predicted_vector

def similar(token1):
    # for token1 in model.wv.index_to_key:
    all_d = {}
    for token2 in model.wv.index_to_key:
        if token1 == token2:
            continue
        d = sp.spatial.distance.cosine(predicted_vectors[token1], predicted_vectors[token2])
        all_d[token2] = d
    sorted_tokens = dict(sorted(all_d.items(), key=lambda item: item[1], reverse=True))
    return sorted_tokens

# print(common_texts)
# print(vector)
# print(predicted_vector)
print(common_texts)
sims1 = model.wv.most_similar('human', topn=10)
print(sims1)
sims2 = similar('human')
print(sims2)



