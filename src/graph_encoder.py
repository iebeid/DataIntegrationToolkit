from importer import *
from grapher import *


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node

    def print_list(self):
        current_node = self.head
        while current_node is not None:
            print(current_node.data)
            current_node = current_node.next


class TrainableOperation(Operation):
    def __init__(self):
        super().__init__()
        pass


class LayerOperation(Operation):
    def __init__(self):
        super().__init__()
        pass


class Attribute:
    def __init__(self):
        super().__init__()
        pass


class Shape(Attribute):
    def __init__(self, input, output, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input = input
        self.output = output


class Hyperparameters(Attribute):
    def __init__(self):
        super().__init__()
        self.configuration = {}

    def add(self, key, value):
        self.configuration[key] = value


class Input(Attribute):
    def __init__(self, shape: Shape = None, input=None, initializer: str = "identity"):
        super().__init__()
        self.shape = shape
        self.input = input
        self.initializer = initializer
        self.output = None
        self.parameters = []
        self.compute()

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        if self.input:
            # if isinstance(self.input, Layer):
            #     X = self.input.output
            #     self.shape = self.input.shape
            # elif isinstance(self.input, Operation):
            #     if all(self.input.output):
            #         X = []
            #         for f in self.input.output:
            #             X.append(eval(f))
            #         X = tf.convert_to_tensor(np.array([np.array(xi) for xi in X]))
            #     else:
            self.output = self.input
            self.parameters = self.input.parameters
            self.shape = self.input.shape
        else:
            if self.initializer == "identity":
                self.output = UtilityOperation.identity_initializer(self.shape.batch_size, self.shape.output)
            elif self.initializer == "glorot":
                self.output = UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)


class Parameters:
    def __init__(self):
        self.shared = {}

    def add(self, key, value):
        self.shared[key] = value


class Layer(DoublyLinkedList):
    def __init__(self, input: Input):
        super().__init__()
        self.input = input.output
        self.shape = self.input.shape
        self.output = self.input
        self.embedding = None
        self.regularization_constant = None
        self.parameters = []

    def __str__(self):
        return str(self.output)

    def update(self, values):
        current_node = self.head
        result = self.output
        index = 0
        intermediate_parameters = []
        number_of_operations = int(len(values) / 2)
        if len(values) > 0:
            y = 0
            for _ in range(number_of_operations):
                temp_values = []
                temp_values.append(values[y])
                temp_values.append(values[y + 1])
                intermediate_parameters.append(temp_values)
                y = y + 2
            while current_node is not None:
                if len(current_node.data.parameters) > 0:
                    current_node.data.update(intermediate_parameters[index])
                    current_node = current_node.next
                    index = index + 1
                else:
                    current_node = current_node.next

    def compute(self):
        current_node = self.head
        result = self.output
        while current_node is not None:
            current_node.data.compute()
            if isinstance(current_node.data, Regularizer):
                self.regularization_constant = current_node.data.regularization_constant
            if isinstance(current_node.data, Normalizer):
                self.embedding = current_node.data.embedding
            if len(current_node.data.parameters) > 0:
                self.parameters = self.parameters + current_node.data.parameters
            result = current_node.data.output
            current_node = current_node.next
        self.output = result


class Dropout(LayerOperation):
    def __init__(self, input: Input, rate: float):
        super().__init__()
        self.input = input
        self.shape = self.input.shape
        self.rate = rate
        self.output = None
        self.parameters = []

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.nn.dropout(self.input.output, self.rate)


class Regularizer(LayerOperation):
    def __init__(self, input: Input, rate: float):
        super().__init__()
        self.input = input
        self.rate = rate
        self.output = None
        self.shape = self.input.shape
        self.regularization_constant = None
        self.parameters = []

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        r = tf.keras.regularizers.L2(self.rate)
        self.regularization_constant = r(self.input.output)
        self.output = self.input.output


class Normalizer(LayerOperation):
    def __init__(self, input):
        super().__init__()
        self.input = input
        self.shape = self.input.shape
        self.output = None
        self.embedding = None
        self.parameters = []

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.embedding = tf.nn.l2_normalize(self.input.output, axis=1)
        self.output = self.input.output


class Activation(LayerOperation):
    def __init__(self, input: Attribute, shape: Attribute):
        super().__init__()
        self.input = input
        self.shape = shape


class TanhActivation(Activation):
    def __init__(self, input: Input, shape: Shape):
        super().__init__(input, shape)
        self.output = None
        self.parameters = []

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.nn.tanh(self.input.output)


class SoftmaxActivation(Activation):
    def __init__(self, input: Attribute, shape: Attribute):
        super().__init__(input, shape)
        self.output = None
        self.parameters = self.input.output.parameters

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.nn.softmax(self.input.output)


class ReluActivation(Activation):
    def __init__(self, input: Attribute, shape: Attribute):
        super().__init__(input, shape)
        self.output = None
        self.parameters = self.input.output.parameters

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.nn.relu(self.input.output)


class SoftmaxPrediction(LayerOperation):
    def __init__(self, input: Attribute, shape: Attribute):
        super().__init__()
        self.input = input
        self.shape = shape
        self.output = None
        self.parameters = self.input.output.parameters

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.input.output)


class BatchNormalization(TrainableOperation):
    def __init__(self, input: Input, shape: Shape, factor=0.001):
        super().__init__()
        self.input = input
        self.shape = shape
        self.factor = factor
        self.output = None
        self.parameters = []
        self.parameters.append(tf.Variable(tf.ones([self.shape.output])))
        self.parameters.append(tf.Variable(tf.zeros([self.shape.output])))

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        batch_mean, batch_var = tf.nn.moments(self.input.output, [0])
        self.output = tf.nn.sigmoid(
            self.parameters[0] * ((self.input.output - batch_mean) / tf.sqrt(
                batch_var + self.factor)) + self.parameters[1])


class Residual(TrainableOperation):
    def __init__(self, input1: Input, input2: Input, shape: Shape):
        super().__init__()
        self.input1 = input1
        self.input2 = input2
        self.shape = shape
        self.output = None
        self.parameters = []
        self.parameters.append(tf.Variable(UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)))
        self.parameters.append(
            tf.Variable(UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)))

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.add(self.input2.output,
                             tf.add(tf.matmul(self.input1.output, self.parameters[0]), self.parameters[1]))


class GraphConvolution(TrainableOperation):
    def __init__(self, input: Input, graph: Graph, shape: Shape):
        super().__init__()
        self.input = input
        self.graph = graph
        self.shape = shape
        self.output = None
        self.parameters = []
        self.parameters.append(tf.Variable(UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)))
        self.parameters.append(
            tf.Variable(UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)))

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_adjacency, dtype=tf.float32),
                      self.input.output  # This is a problem
                      ), self.parameters[0]), self.parameters[1])


class DirectedGraphConvolutional(TrainableOperation):
    def __init__(self, output_dim, graph, input, activation, batch_normalization, batch_normalization_factor, residual):
        self.graph = graph
        self.input = input
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.batch_normalization_factor = batch_normalization_factor
        self.residual = residual

        self.W_in = tf.Variable(UtilityOperation.glorot_intializer(int(self.input.shape.dims[1]), output_dim),
                                dtype=tf.float32)
        self.b_in = tf.Variable(UtilityOperation.glorot_intializer(int(self.input.shape.dims[0]), output_dim),
                                dtype=tf.float32)

        self.W_out = tf.Variable(UtilityOperation.glorot_intializer(int(self.input.shape.dims[1]), output_dim),
                                 dtype=tf.float32)
        self.b_out = tf.Variable(UtilityOperation.glorot_intializer(int(self.input.shape.dims[0]), output_dim),
                                 dtype=tf.float32)

    def compute(self):
        a = tf.add(tf.matmul(tf.matmul(self.graph.normalized_out_adjacecny, self.input), self.W_out), self.b_out)
        aT = tf.transpose(a)
        b = tf.add(tf.matmul(tf.matmul(self.graph.normalized_in_adjacecny, self.input), self.W_in), self.b_in)
        bT = tf.transpose(b)
        hidden_layer = tf.add_n([tf.matmul(a, aT), tf.matmul(a, bT), tf.matmul(b, aT), tf.matmul(b, bT)])
        # if self.batch_normalization:
        #     hidden_layer = BatchNormalizationLayer(hidden_layer, self.batch_normalization_factor)
        # output = self.activation(hidden_layer)
        # if self.residual:
        #     output = ResidualLayer(self.input, hidden_layer)
        # return output


class Dense(TrainableOperation):
    def __init__(self, input: Attribute, shape: Attribute):
        super().__init__()
        self.input = input
        self.shape = shape
        self.output = None
        self.parameters = []
        self.parameters.append(tf.Variable(UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)))
        self.parameters.append(
            tf.Variable(UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)))

    def update(self, values):
        for index, value in enumerate(values):
            self.parameters[index] = value

    def compute(self):
        self.output = tf.add(tf.matmul(self.input.output, self.parameters[0]), self.parameters[1])


class Model(DoublyLinkedList):
    def __init__(self, data, hyperparameters: Attribute):
        super().__init__()
        # Data
        self.graph = data

        # Hyperparameters
        self.epochs = hyperparameters.configuration["epochs"]
        self.split = hyperparameters.configuration["split"]
        self.batch_normalization_factor = hyperparameters.configuration["batch_normalization_factor"]
        self.regularization_rate = hyperparameters.configuration["regularization_rate"]
        self.dropout_rate = hyperparameters.configuration["dropout_rate"]
        self.learn_rate = hyperparameters.configuration["learn_rate"]
        self.activation = hyperparameters.configuration["activation"]
        self.input_layer_dimensions = hyperparameters.configuration["input_layer_dimensions"]
        self.layer_1_dimensions = hyperparameters.configuration["layer_1_dimensions"]
        self.layer_2_dimensions = hyperparameters.configuration["layer_2_dimensions"]
        self.layer_3_dimensions = hyperparameters.configuration["layer_3_dimensions"]
        self.output_layer_dimensions = hyperparameters.configuration["output_layer_dimensions"]

        # Extract training, test and validation data
        self.train_mask, self.test_mask, self.valid_mask = self.graph.balanced_node_label_sampler(self.split)

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

    def compile(self):

        # Input Layer
        self.input = Layer(input=Input(shape=self.input_layer_dimensions))
        self.add(self.input)

        # First GCN Layer
        self.H1 = Layer(input=Input(input=self.input, shape=self.layer_1_dimensions))
        operation1 = GraphConvolution(input=self.H1.input, graph=self.graph, shape=self.layer_1_dimensions)
        operation2 = BatchNormalization(input=operation1, factor=self.batch_normalization_factor,
                                        shape=self.layer_1_dimensions)
        operation3 = TanhActivation(input=operation2, shape=self.layer_1_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_1_dimensions)
        operation5 = Regularizer(input=operation4, rate=self.regularization_rate)
        operation6 = Dropout(input=operation5, rate=self.dropout_rate)
        self.H1.add(operation1)
        self.H1.add(operation2)
        self.H1.add(operation3)
        self.H1.add(operation4)
        self.H1.add(operation5)
        self.H1.add(operation6)
        self.add(self.H1)

        # Second GCN Layer
        self.H2 = Layer(input=Input(input=self.H1, shape=self.layer_2_dimensions))
        operation1 = GraphConvolution(input=self.H2.input, graph=self.graph, shape=self.layer_2_dimensions)
        operation2 = BatchNormalization(input=operation1, factor=self.batch_normalization_factor,
                                        shape=self.layer_2_dimensions)
        operation3 = TanhActivation(input=operation2, shape=self.layer_2_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_2_dimensions)
        operation5 = Dropout(input=operation4, rate=self.dropout_rate)
        self.H2.add(operation1)
        self.H2.add(operation2)
        self.H2.add(operation3)
        self.H2.add(operation4)
        self.H2.add(operation5)
        self.add(self.H2)

        # Third GCN Layer
        self.H3 = Layer(input=Input(input=self.H2, shape=self.layer_3_dimensions))
        operation1 = GraphConvolution(input=self.H3.input, graph=self.graph, shape=self.layer_3_dimensions)
        operation2 = BatchNormalization(input=operation1, factor=self.batch_normalization_factor,
                                        shape=self.layer_3_dimensions)
        operation3 = TanhActivation(input=operation2, shape=self.layer_3_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_3_dimensions)
        operation5 = Normalizer(input=operation4)
        operation6 = Dropout(input=operation5, rate=self.dropout_rate)
        self.H3.add(operation1)
        self.H3.add(operation2)
        self.H3.add(operation3)
        self.H3.add(operation4)
        self.H3.add(operation5)
        self.H3.add(operation6)
        self.add(self.H3)

        # Forth dense layer
        self.H4 = Layer(input=Input(input=self.H3, shape=self.output_layer_dimensions))
        operation1 = Dense(input=self.H4.output, shape=self.output_layer_dimensions)
        self.H4.add(operation1)
        self.add(self.H4)

        # Output prediction layer
        self.output = Layer(input=Input(input=self.H4, shape=self.output_layer_dimensions))
        operation1 = SoftmaxPrediction(input=self.output.output, shape=self.output_layer_dimensions)
        self.output.add(operation1)
        self.add(self.output)

    def compute(self):

        self.parameters = []
        self.parameter_lookup = []
        self.input.compute()
        self.parameters = self.parameters + self.input.parameters
        self.parameter_lookup.append(self.input.parameters)
        self.H1.compute()
        self.parameters = self.parameters + self.H1.parameters
        self.parameter_lookup.append(self.H1.parameters)
        self.regularization_constant = self.H1.regularization_constant
        self.H2.compute()
        self.parameters = self.parameters + self.H2.parameters
        self.parameter_lookup.append(self.H2.parameters)
        self.H3.compute()
        self.parameters = self.parameters + self.H3.parameters
        self.parameter_lookup.append(self.H3.parameters)
        self.embeddings = self.H3.embedding
        self.H4.compute()
        self.parameters = self.parameters + self.H4.parameters
        self.parameter_lookup.append(self.H4.parameters)
        self.output.compute()
        self.parameters = self.parameters + self.output.parameters
        self.parameter_lookup.append(self.output.parameters)
        self.predictions = self.output.output
        # Convert parameters into a list of lists
        self.update()

    def update(self):
        self.input.update(self.parameter_lookup[0])
        self.H1.update(self.parameter_lookup[1])
        self.H2.update(self.parameter_lookup[2])
        self.H3.update(self.parameter_lookup[3])
        self.H4.update(self.parameter_lookup[4])
        self.output.update(self.parameter_lookup[5])

    @tf.function
    def train(self):
        for epoch in tf.range(self.epochs):
            start_time = time.perf_counter()
            with tf.GradientTape() as tape:
                self.compute()
                self.train_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                                       self.regularization_constant,
                                                                                       self.train_mask)
                self.train_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                                 self.train_mask)
            self.gradients = tape.gradient(self.train_loss, self.parameters)
            UtilityOperation.optimizer(self.learn_rate).apply_gradients(list(zip(self.gradients, self.parameters)))

            self.compute()
            self.test_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                                  self.regularization_constant,
                                                                                  self.test_mask)
            self.test_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                            self.test_mask)
            end_time = time.perf_counter()
            self.time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            tf.print(" Epoch: " + tf.strings.as_string(epoch)
                     + " Seconds/Epoch: " + tf.strings.as_string(self.time_per_epoch)
                     + " Learning Rate: " + tf.strings.as_string(
                tf.constant(round(self.learn_rate, 3), dtype=tf.float32))
                     + " Train Loss: " + tf.strings.as_string(self.train_loss)
                     + " Train Accuracy: " + tf.strings.as_string(self.train_accuracy)
                     + " Test Loss: " + tf.strings.as_string(self.test_loss)
                     + " Test Accuracy: " + tf.strings.as_string(self.test_accuracy)
                     )

    def validate(self):
        # Validate trained model
        self.compute()
        self.valid_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                               self.regularization_constant,
                                                                               self.valid_mask)
        self.valid_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                         self.valid_mask)

        for sublist in self.predictions:
            if np.isnan(sublist[0]):
                print("The list contains a NaN")
                break
            else:
                self.valid_predictions.append(self.predictions)
                self.valid_losses.append(self.valid_loss)
                self.valid_accuracies.append(self.valid_accuracy)
                self.valid_embeddings.append(self.embeddings)
                self.valid_parameters.append(self.parameters)

    def visualize(self):
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(self.embeddings))
        z = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(z, y)
        for i, txt in enumerate(self.graph.nodes.values()):
            ax.annotate(txt, (z[i], y[i]))
        plt.show()
