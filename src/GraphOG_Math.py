import scipy.spatial
import numpy as np
from scipy.stats import norm
import numpy as np


# MATH OPERATIONS
def cosine_distance(vector1, vector2):
    return 1 - scipy.spatial.distance.cosine(vector1, vector2)


def euclidean_distance(vector1, vector2):
    return scipy.spatial.distance.euclidean(vector1, vector2)


def squared_euclidean_distance(vector1, vector2):
    return scipy.spatial.distance.sqeuclidean(vector1, vector2)


def normalize_vector(vector):
    return list((vector - np.min(vector)) / (np.max(vector) - np.min(vector)))


def probability_vector(vector):
    p_v = []
    for v in vector:
        p = norm().cdf(v)
        p_v.append(p)
    return p_v
