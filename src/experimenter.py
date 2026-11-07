from importer import *
from grapher import *
from graph_encoder import *
from mapper import *
from evaluater import *


class Experiment:
    def __init__(self, dummy):

        # Initialize devices
        print(
            "Self-supervised Dense Directed Graph Convolutional Neural Networks for Unsupervised Schema-agnostic Entity Resolution")
        print("Tensorflow version: " + str(tf.__version__))
        print(tf.config.list_physical_devices('GPU'))
        print(tf.config.list_physical_devices('CPU'))

        # Experiment configuration
        self.data_file = "data/S1G.txt"
        self.ground_truth_file = "eval/truthABCgoodDQ.txt"
        self.log_file = "logs/log-file-dgcn-11-22-23.txt"
        self.embedding_file = "embedding.tsv"
        self.metadata_file = "metadata.tsv"
        self.dummy = dummy

        if self.dummy:
            self.raw_data = nx.karate_club_graph().edges
            self.graph = UndirectedGraph(edges=self.raw_data)
            print(self.graph)
        else:
            cm_object = MappER(",", self.data_file)
            print(cm_object.character_edges)
            self.undirected_character_graph = UndirectedGraph(edges=cm_object.character_edges)
            self.directed_character_graph = DirectedGraph(edges=cm_object.character_edges)
            # self.undirected_token_graph = UndirectedGraph(edges=cm_object.token_edges)
            # self.directed_token_graph = DirectedGraph(edges=cm_object.token_edges)
            # self.bipartite_character_token_graph = BipartiteGraph(nodes=cm_object.character_token_nodes, edges=cm_object.character_token_edges)
            # self.bipartite_token_entity_graph = BipartiteGraph(nodes=cm_object.token_entity_nodes, edges=cm_object.token_entity_edges)
            print("Data loaded. Graph created.")
            print(self.undirected_character_graph)
            print(self.directed_character_graph)
            # print(self.undirected_token_graph)
            # print(self.directed_token_graph)
            # print(self.bipartite_character_token_graph)
            # print(self.bipartite_token_entity_graph)

        # # Hyperparameters
        # self.hp = Hyperparameters()
        # self.hp.add("K", 5)
        # self.hp.add("epochs", 20)
        # self.hp.add("patience", 100)
        # self.hp.add("split", 70)
        # self.hp.add("residual", True)
        # self.hp.add("batch_normalization", True)
        # self.hp.add("batch_size", self.undirected_character_graph.number_of_nodes)
        # self.hp.add("batch_normalization_factor", 0.001)
        # self.hp.add("regularization_rate", 0.0005)
        # self.hp.add("learn_rate", 0.01)
        # self.hp.add("dropout_rate", 0.5)
        # self.hp.add("cosine_thershold", 0.001)
        # self.hp.add("activation", "tanh")
        # self.hp.add("input_layer_dimensions",
        #             Shape(self.hp.configuration["batch_size"], 16, self.hp.configuration["batch_size"]))
        # self.hp.add("layer_1_dimensions", Shape(16, 16, self.hp.configuration["batch_size"]))
        # self.hp.add("layer_2_dimensions", Shape(16, 16, self.hp.configuration["batch_size"]))
        # self.hp.add("layer_3_dimensions", Shape(16, 16, self.hp.configuration["batch_size"]))
        # self.hp.add("output_layer_dimensions",
        #             Shape(16, self.undirected_character_graph.number_of_classes, self.hp.configuration["batch_size"]))
        #
        # # Metadata
        # self.valid_embeddings = []
        # self.valid_losses = []
        # self.valid_accuracies = []
        # self.valid_predictions = []
        # self.valid_parameters = []

    def run(self):
        for k in range(self.K):
            print(str(self.K) + " fold validation iteration number: " + str(k))
            m = Model(self.graph, self.hp)
            m.compile()
            m.compute()
            m.train()
            m.validate()

    def finalize(self):
        self.max_valid_accuracy_index = np.argmax(np.array(self.valid_accuracies))
        self.final_predictions = list(self.valid_predictions[self.max_valid_accuracy_index].numpy())
        self.final_loss = self.valid_losses[self.max_valid_accuracy_index].numpy()
        self.final_accuracy = self.valid_accuracies[self.max_valid_accuracy_index].numpy()
        self.final_embeddings = list(self.valid_embeddings[self.max_valid_accuracy_index].numpy())
        self.final_parameters = self.valid_parameters[self.max_valid_accuracy_index]

    def visualize_embeddings(self):
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(self.final_embeddings))
        z = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(z, y)
        for i, txt in enumerate(self.graph.nodes.values()):
            ax.annotate(txt, (z[i], y[i]))
        plt.show()

    def evaluate(self):
        self.sample_precision, self.sample_recall, self.sample_accuracy, self.sample_f1_score = self.evaluate_embeddings()
        self.visualize_embeddings()
        self.embedding_precision, self.embedding_recall, self.embedding_accuracy, self.embedding_f1_score = self.evaluate_samples()
        self.ari = self.evaluate_clusters()
        print("Final Loss: " + str(self.final_loss))
        print("Final Total Accuracy: " + str(self.final_accuracy))
        print("Final Samples F1-Score: " + str(self.sample_f1_score))
        print("Final Embedding F1-Score: " + str(self.embedding_f1_score))
        print("Final Adjusted Rand Index: " + str(self.ari))
        if not self.dummy:
            MappER.embed_entities(self.graph, self.embedding)
            EvaluatER(list(
                nx.algorithms.components.connected_components(
                    nx.Graph(MappER.find_matches(self.cosine_thershold)))),
                self.ground_truth_file).run()


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


@measure_execution_time
def main():
    e = Experiment(False)
    # e.run()
    # e.finalize()
    # e.evaluate()


if __name__ == "__main__":
    main()
