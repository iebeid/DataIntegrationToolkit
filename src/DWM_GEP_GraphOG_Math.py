import scipy.spatial


# MATH OPERATIONS
def cosine_distance(vector1, vector2):
    return 1 - scipy.spatial.distance.cosine(vector1, vector2)

def euclidean_distance(vector1, vector2):
    return float(abs(scipy.spatial.distance.euclidean(vector1, vector2) )**2)