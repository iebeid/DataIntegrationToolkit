import tensorflow as tf

# Create a tensor with one dimension
tensor = tf.constant([1, 2, 3, 4, 5])

# Split the tensor into two tensors with two elements each
split_tensors = tf.expand_dims(tensor, axis=1)

# Print the values of the split tensors
print(split_tensors)
