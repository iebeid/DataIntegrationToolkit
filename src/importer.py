import time
import re
import chardet

from collections import defaultdict
from collections import Counter
from collections import deque

from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import json

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import sklearn as skl
import tensorflow as tf

from sklearn.manifold import TSNE
from gensim.models import Word2Vec

import spacy
nlp = spacy.load("en_core_web_sm")

np.random.seed(123)
tf.random.set_seed(123)
tf.config.threading.set_inter_op_parallelism_threads(8)
np.set_printoptions(threshold=np.inf)

class Operation:
    def __init__(self):
        pass


class UtilityOperation(Operation):
    def __init__(self):
        super().__init__()
        pass

    # Algorithms
    @staticmethod
    def invert_dict(d):
        inverted_dict = {}
        for key, value_list in d.items():
            for value in value_list:
                inverted_dict.setdefault(value, set()).add(key)
        return inverted_dict

    @staticmethod
    def cosine_distance(vector1, vector2):
        return sp.spatial.distance.cosine(vector1, vector2)

    @staticmethod
    def flip_list(list):
        flipped_edges = []
        for es in list:
            flipped_edges.append((es[1], es[0], es[2]))
        return flipped_edges

    @staticmethod
    def determine_file_encoding(input_file_name):
        encoding=None
        with open(input_file_name, "rb") as rawdata:
            result = chardet.detect(rawdata.read())
            encoding = result["encoding"]
            if encoding == "ascii":
                encoding = "utf-8"
        return encoding

    # Initializers
    @staticmethod
    def glorot_intializer(in_d, out_d):
        init = tf.keras.initializers.GlorotUniform()
        return init(shape=(in_d, out_d), dtype=tf.float32)

    @staticmethod
    def identity_initializer(in_d, out_d):
        init = tf.keras.initializers.Identity()
        return init(shape=(in_d, out_d), dtype=tf.float32)

    # Loss function
    @staticmethod
    def masked_cross_entropy_loss_evaluater(prediction, y, r, mask):
        return (-tf.reduce_mean(
            tf.reduce_sum(tf.boolean_mask(y, mask) * tf.math.log(tf.boolean_mask(prediction, mask))))) + r

    # Optimizers
    @staticmethod
    def optimizer(learn_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        return optimizer

    # Evaluation from Kipf et al 2017
    @staticmethod
    def masked_accuracy_evaluater(prediction, y, mask):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy