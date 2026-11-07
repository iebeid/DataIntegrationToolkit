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

from ConfigER import ConfigER
from StagER import StagER

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


class Utils:
    def __init__(self):
        pass

    def initializer(self, shape):
        r = tf.keras.initializers.RandomUniform(minval=float(-1 * (6 / np.sqrt(shape[1]))),
                                                maxval=float(6 / np.sqrt(shape[1])), seed=1234)
        return r(shape=shape)

    def is_symmetric(self, matrix):
        """Returns True if the matrix is symmetric, False otherwise."""
        for i in range(len(matrix)):
            for j in range(i):
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True

    def collect_dictionary(self, obj):
        inv_obj = defaultdict(list)
        for key, value in obj.items():
            inv_obj[value].append(key)
        return dict(inv_obj)


class VectER:
    def __init__(self):
        # ------------------Initialize Utils------------------
        self.utils = Utils()
        # ------------------Preprocess File-------------------
        self.stager = StagER(ConfigER.delimiter, ConfigER.input_file_name, ConfigER.percentile_of_stop_words,
                             ConfigER.percentile_of_blocking_words, ConfigER.percentile_of_discriminate_words_lengths)
        self.stager.generate_word_corpus()
        print(self.stager.unique_tokenized_corpus)
        # ------------------Hyper Parameters------------------
        self.epochs = 10
        self.P_k = 64
        self.E_k = 32
        self.T_k = 16
        self.number_of_entities = int(len(self.stager.unique_tokenized_corpus.keys()))
        self.number_of_profiles = self.number_of_entities
        self.number_of_tokens = int(len(self.stager.word_count_dict.keys()))
        # ------------------Indexing----------------------------
        self.token_to_index = {}
        for i, w in enumerate(self.stager.word_count_dict.keys()):
            self.token_to_index[w] = i
        self.entity_to_token = {}
        self.entity_to_profile = {}
        self.entity_to_index = {}
        entity_index = 0
        profile_index = 0
        for e, ts in self.stager.unique_tokenized_corpus.items():
            tokens = []
            for t in ts:
                tokens.append(self.token_to_index[t])
            self.entity_to_token[e] = tokens
            self.entity_to_index[e] = entity_index
            self.entity_to_profile[e] = [profile_index]
            entity_index = entity_index + 1
            profile_index = profile_index + 1
        self.token_to_entity = {}
        for t in self.stager.word_count_dict.keys():
            ets = []
            for e, ts in self.entity_to_token.items():
                if self.token_to_index[t] in ts:
                    ets.append(int(e))
            self.token_to_entity[t] = ets
        self.profile_to_index = {}
        self.profile_to_entity = {}
        for i in range(self.number_of_profiles):
            self.profile_to_index[str("P" + str(i))] = i
            self.profile_to_entity[str("P" + str(i))] = [i]

    def label_data(self):
        # -------------------Labeling--------------------
        lsh = MinHashLSH(threshold=ConfigER.lsh_threshold, num_perm=ConfigER.lsh_permutations)
        inverted_hash_table = {}
        for doc_id, document in tqdm(self.stager.unique_tokenized_corpus.items()):
            m = MinHash(num_perm=ConfigER.lsh_permutations)
            for word in document:
                m.update(word.encode(self.stager.encoding))
            lsh.insert(doc_id, m)
            inverted_hash_table[doc_id] = m
        all_distances = []
        for e1, ts1 in self.entity_to_token.items():
            m1 = inverted_hash_table[e1]
            entity_distances = []
            for e2, ts2 in self.entity_to_token.items():
                s = m1.jaccard(inverted_hash_table[e2])
                entity_distances.append(s)
            all_distances.append(entity_distances)
        jaccard_indices_matrix = np.array(all_distances)
        print(jaccard_indices_matrix.shape)
        # For each entity how many scores are greater than 0.5
        print(self.utils.is_symmetric(jaccard_indices_matrix))
        entity_to_count = {}
        for e, i in self.entity_to_index.items():
            entity_to_count[i] = len(np.where(jaccard_indices_matrix[:, i] > 0.5)[0])
        entity_to_count = dict(sorted(entity_to_count.items(), reverse=True, key=lambda item: item[1]))
        count_to_entity = self.utils.collect_dictionary(entity_to_count)
        elements = list(count_to_entity.keys())
        binary_tree_edge_list = []
        for i in range(len(elements)):
            window = elements[i:i + 3:2]
            if len(window) == 1:
                break
            if len(window) == 2:
                binary_tree_edge_list.append((window[0], window[1]))
            if len(window) > 2:
                root = window[0]
                if window[1] < window[2]:
                    left = window[1]
                    right = window[2]
                else:
                    left = window[2]
                    right = window[1]
                binary_tree_edge_list.append((root, left))
                binary_tree_edge_list.append((root, right))
        print(binary_tree_edge_list)

    def init_parameters(self):
        # -------------------Model Parameters------------
        # Profile vector space
        self.P = tf.Variable(self.utils.initializer((self.number_of_profiles, self.P_k)), name="P", trainable=True,
                             dtype=tf.float32)
        self.P_E_Transformation = tf.Variable(self.utils.initializer((self.P_k, self.E_k)), name="P_E_R",
                                              trainable=True,
                                              dtype=tf.float32)
        self.E_P_Transformation = tf.Variable(self.utils.initializer((self.E_k, self.P_k)), name="E_P_R",
                                              trainable=True,
                                              dtype=tf.float32)
        # Entity vector space
        self.E = tf.Variable(self.utils.initializer((self.number_of_entities, self.E_k)), name="E", trainable=True,
                             dtype=tf.float32)
        # Associated transformations
        self.E_T_Transformation = tf.Variable(self.utils.initializer((self.E_k, self.T_k)), name="E_T_R",
                                              trainable=True,
                                              dtype=tf.float32)
        self.T_E_Transformation = tf.Variable(self.utils.initializer((self.T_k, self.E_k)), name="T_E_R",
                                              trainable=True,
                                              dtype=tf.float32)
        # Token vector space
        self.T = tf.Variable(self.utils.initializer((self.number_of_tokens, self.T_k)), name="T", trainable=True,
                             dtype=tf.float32)
        # Descriminative power scores
        self.D = tf.Variable(self.utils.initializer((self.number_of_entities, self.number_of_tokens)), name="D",
                             trainable=True,
                             dtype=tf.float32)
        # Membership scores
        self.M = tf.Variable(self.utils.initializer((self.number_of_profiles, self.number_of_entities)), name="D",
                             trainable=True,
                             dtype=tf.float32)

    def transformational_model(self):
        # First pass
        # Transform token space to entity space to prepare for update
        self.T_E = tf.matmul(self.T, self.T_E_Transformation)
        # Transform entity space to profile space to prepare for update
        self.E_P = tf.matmul(self.E, self.E_P_Transformation)
        # --------------------------------------------------------------
        # Second pass
        # Transform profile space back to entity space
        self.P_E = tf.matmul(self.P, self.P_E_Transformation)
        # Transform entities into token space
        self.E_T = tf.matmul(self.E, self.E_T_Transformation)
        # --------------------------------------------------------------
        # Third pass
        # Compute correction vectors for token space
        entity_correction_vectors_in_token_space = []
        entity_correction_vectors_in_entity_space = []
        profile_correction_vectors_in_entity_space = []
        profile_correction_vectors_in_profile_space = []
        for e, ts in self.entity_to_token.items():
            mean_embedding_of_entity_tokens = tf.expand_dims(tf.reduce_mean(tf.nn.embedding_lookup(self.T, ts), axis=0),
                                                             axis=0)
            projected_entity_embedding = tf.nn.embedding_lookup(self.E_T, [int(self.entity_to_index[e])])
            entity_correction_vector_in_token_space = tf.subtract(mean_embedding_of_entity_tokens,
                                                                  projected_entity_embedding)
            entity_correction_vectors_in_token_space.append(entity_correction_vector_in_token_space)
        # Compute correction vectors for entity space
        for e, ts in self.entity_to_token.items():
            mean_embedding_of_projected_entity_tokens = tf.expand_dims(
                tf.reduce_mean(tf.nn.embedding_lookup(self.T_E, ts), axis=0), axis=0)
            main_entity_embedding = tf.nn.embedding_lookup(self.E, [int(self.entity_to_index[e])])
            entity_correction_vector_in_entity_space = tf.subtract(mean_embedding_of_projected_entity_tokens,
                                                                   main_entity_embedding)
            entity_correction_vectors_in_entity_space.append(entity_correction_vector_in_entity_space)
        for p, es in self.profile_to_entity.items():
            mean_embedding_of_projected_profile_entities = tf.expand_dims(
                tf.reduce_mean(tf.nn.embedding_lookup(self.E, es), axis=0), axis=0)
            main_profile_embedding = tf.nn.embedding_lookup(self.P_E, [int(self.profile_to_index[p])])
            profile_correction_vector_in_entity_space = tf.subtract(mean_embedding_of_projected_profile_entities,
                                                                    main_profile_embedding)
            profile_correction_vectors_in_entity_space.append(profile_correction_vector_in_entity_space)
        # Compute correction vectors for profile space
        for p, es in self.profile_to_entity.items():
            mean_embedding_of_projected_entities_in_profile_space = tf.expand_dims(
                tf.reduce_mean(tf.nn.embedding_lookup(self.E_P, es), axis=0), axis=0)
            main_profile_embedding = tf.nn.embedding_lookup(self.P, [int(self.profile_to_index[p])])
            profile_correction_vector_in_profile_space = tf.subtract(
                mean_embedding_of_projected_entities_in_profile_space, main_profile_embedding)
            profile_correction_vectors_in_profile_space.append(profile_correction_vector_in_profile_space)
        entity_correction_vectors_in_token_space = tf.squeeze(
            tf.convert_to_tensor(entity_correction_vectors_in_token_space))
        entity_correction_vectors_in_entity_space = tf.squeeze(
            tf.convert_to_tensor(entity_correction_vectors_in_entity_space))
        profile_correction_vectors_in_entity_space = tf.squeeze(
            tf.convert_to_tensor(profile_correction_vectors_in_entity_space))
        profile_correction_vectors_in_profile_space = tf.squeeze(
            tf.convert_to_tensor(profile_correction_vectors_in_profile_space))
        # --------------------------------------------------------------
        # Halt and compute cost
        magnitude_of_correction_token_space = tf.sqrt(
            tf.reduce_sum(tf.square(entity_correction_vectors_in_token_space)))
        magnitude_of_correction_entity_space = tf.sqrt(
            tf.reduce_sum(tf.square(entity_correction_vectors_in_entity_space))) + tf.sqrt(
            tf.reduce_sum(tf.square(profile_correction_vectors_in_entity_space)))
        magnitude_of_correction_profile_space = tf.sqrt(
            tf.reduce_sum(tf.square(profile_correction_vectors_in_profile_space)))
        total_correction = magnitude_of_correction_token_space + magnitude_of_correction_entity_space + magnitude_of_correction_profile_space
        print("Total correction distances: " + str(total_correction.numpy()))
        mean_pairwise_profile_distances = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.P, axis=1), tf.expand_dims(self.P, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_entities_distances = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.E, axis=1), tf.expand_dims(self.E, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_tokens_distances = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.T, axis=1), tf.expand_dims(self.T, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_distances_between_entities_in_token_space = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.E_T, axis=1), tf.expand_dims(self.E_T, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_distances_between_tokens_in_entity_space = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.T_E, axis=1), tf.expand_dims(self.T_E, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_distances_between_entities_in_profile_space = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.E_P, axis=1), tf.expand_dims(self.E_P, axis=0)), axis=-1)), 0, 50))
        mean_pairwise_distances_between_profiles_in_entity_space = tf.reduce_mean(
            tf.linalg.band_part(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
                tf.expand_dims(self.P_E, axis=1), tf.expand_dims(self.P_E, axis=0)), axis=-1)), 0, 50))
        total_mean_distances = mean_pairwise_profile_distances + mean_pairwise_entities_distances + \
                               mean_pairwise_tokens_distances + mean_pairwise_distances_between_entities_in_token_space + \
                               mean_pairwise_distances_between_tokens_in_entity_space + \
                               mean_pairwise_distances_between_entities_in_profile_space + \
                               mean_pairwise_distances_between_profiles_in_entity_space
        print("Total mean distances: " + str(total_mean_distances.numpy()))
        # The cost has been computed at this point
        # --------------------------------------------------------------
        # Forth pass
        # Update P
        self.P = tf.add(self.P, profile_correction_vectors_in_profile_space)
        # Transform profile space back to entity space
        self.P_E = tf.matmul(self.P, self.P_E_Transformation)
        # Update E
        self.E = tf.add(self.E, entity_correction_vectors_in_entity_space)
        # Transform entities into token space
        self.E_T = tf.matmul(self.E, self.E_T_Transformation)
        # Update T
        for t, ets in self.token_to_entity.items():
            token_index = int(self.token_to_index[t])
            token_entity_embedding = tf.nn.embedding_lookup(self.E_T, ets)
            score_indices = []
            for ei in ets:
                score_indices.append([ei, token_index])
            discriminative_scores = tf.expand_dims(tf.gather_nd(self.D, score_indices), axis=1)
            scaled_token_entity_embedding = tf.multiply(discriminative_scores, token_entity_embedding)
            new_t_vector = tf.expand_dims(tf.reduce_mean(scaled_token_entity_embedding, axis=0), axis=0)
            tf.tensor_scatter_nd_update(self.T, tf.constant([[token_index]]), new_t_vector)

        # # Update E
        # for e, ps in self.entity_to_profile.items():
        #     entity_index = int(self.entity_to_index[e])
        #     entity_embeddings = tf.nn.embedding_lookup(self.P_E, ps)
        #     new_e_vector = tf.expand_dims(tf.reduce_mean(entity_embeddings, axis=0), axis=0)
        #     tf.tensor_scatter_nd_update(self.E, tf.constant([[entity_index]]), new_e_vector)
        # self.E_P = tf.matmul(self.E, self.E_P_Transformation)
        # # Update P
        # for p, es in self.profile_to_entity.items():
        #     profile_index = int(self.profile_to_index[p])
        #     profile_embeddings = tf.nn.embedding_lookup(self.E_P, es)
        #     new_p_vector = tf.expand_dims(tf.reduce_mean(profile_embeddings, axis=0), axis=0)
        #     tf.tensor_scatter_nd_update(self.P, tf.constant([[profile_index]]), new_p_vector)

    # -------------------Training----------------------
    def train(self):
        threshold = 0.00001
        scaled_projection_distance = 0
        while scaled_projection_distance < threshold:
            # ------------------------------------------------------------------------------
            self.transformational_model()
            # ------------------------------------------------------------------------------

    def visualize_embedding(self):
        # ------------------------Visualization-------------------------
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(self.E.numpy())
        df = pd.DataFrame()
        df["label"] = self.stager.unique_tokenized_corpus.keys()
        df["X"] = z[:, 0]
        df["Y"] = z[:, 1]
        sns.scatterplot(x="X", y="Y", hue=df.label.tolist(),
                        palette=sns.color_palette("hls", len(self.stager.unique_tokenized_corpus.keys())),
                        data=df).set(title="Entity Embedding Projection")
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()


@measure_execution_time
def main():
    transer = VectER()
    transer.label_data()
    transer.init_parameters()
    transer.train()
    transer.visualize_embedding()


if __name__ == '__main__':
    main()
