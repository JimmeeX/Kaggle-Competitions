import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import nn_models as qfns
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#network = "none"
#network = "onelayer"
#network = "twolayer"
#network = "conv"
network = "myconvnet"

def read_train_dataset():
	df = pd.read_csv("train.csv")
	y = df.pop("label")
	X = df.values / 255

	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	Y = one_hot_encode(y)
	return (X, Y)

def read_test_dataset():
	df = pd.read_csv("test.csv")
	X = df.values / 255
	return X

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels, n_unique_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode
	
def chunker(seq, size):
   return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def accuracy(sess, dataX, dataY, batch_size, X, Y, accuracy_op):
   # compute number of batches for given batch_size
   m = dataX.shape[0]
   num_test_batches = -(-m // batch_size)

   overall_accuracy = 0.0
   for batchX, batchY in zip(chunker(dataX, batch_size), chunker(dataY, batch_size)):
   	accuracy_batch = sess.run(accuracy_op, feed_dict={X: batchX, Y: batchY})
   	overall_accuracy += accuracy_batch

   return (overall_accuracy/num_test_batches)

def get_accuracy_op(preds_op, Y):
	with tf.name_scope('accuracy_ops'):
		correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(Y, 1))
		accuracy_op = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32))
	return accuracy_op

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope(name+'_summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def plot_image_on_grid(conv_filters, conv1, conv2):
	# W_conv1 = weight_variable([3,3,1,128])
	# #f_x,f_y,depth, number of filters
	# b_conv1 = bias_variable([128])
	# cnn_layer_1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# W_conv2 = weight_variable([3,3,128,128])
	# b_conv2 = bias_variable([128])
	# cnn_layer_2 = tf.nn.relu(conv2d(cnn_layer_1, W_conv2) + b_conv2)

	# W_conv3 = weight_variable([3,3,128,128])
	# b_conv3 = bias_variable([128])
	# cnn_layer_3 = tf.nn.relu(conv2d(cnn_layer_2, W_conv3) + b_conv3)

	#grab only the first element of the batch and 16 filters
	#layer1_image1 = conv1[0, :, :, 0:2]
	# layer2_image1 = conv2[0, :, :, 0:16]
	# layer3_image1 = cnn_layer_3[0:1, :, :, 0:16]

	#layer1_image1 = tf.transpose(layer1_image1)
	# layer2_image1 = tf.transpose(layer2_image1)
	# layer3_image1 = tf.transpose(layer3_image1, perm=[3,1,2,0])
	print("HELLO")
	# layer_combine_1 = tf.concat([layer2_image1, layer1_image1], axis=2)
	#list_lc1 = tf.split(layer1_image1, axis=0, num_or_size_splits=2)
	#print(conv1.get_shape())
	# image = tf.reshape(conv1[:1], [-1, 28, 28, 1])
	# tf.summary.image("filtered_images_1", image)
	# combine this summary with tensorboard (and you get a decent output);

def train(sess, x_train, y_train, x_val, y_val, x_test, n_training_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, train_op, loss_op, accuracy_op):
	m = x_train.shape[0]
	# Compute number of batches
	num_train_batches = -(-m // batch_size)
	# Record Starting time
	train_start = time.time()

	# Run through the entire dataset n_training_epochs times
	for i in range(n_training_epochs):
		# Initialise Statistics
		training_loss = 0
		epoch_start = time.time()

		# Run SGD train op for each minibatch
		for batchX, batchY in zip(chunker(x_train, batch_size), chunker(y_train, batch_size)):
			trainstep_result, batch_loss, summary = qfns.train_step(sess, batchX, batchY, X, Y, train_op, loss_op, summaries_op)
			train_writer.add_summary(summary, i)
			training_loss += batch_loss

		# Timing & Statistics
		epoch_duration = round(time.time() - epoch_start, 2)
		ave_train_loss = training_loss / num_train_batches

		# Get Accuracy
		train_accuracy = accuracy(sess, x_train, y_train, batch_size, X, Y, accuracy_op)
		test_accuracy = accuracy(sess, x_val, y_val, batch_size, X, Y, accuracy_op)

		# Log accuracy at current epoch on training and test sets
		train_acc_summary = sess.run(accuracy_summary_op, feed_dict={accuracy_placeholder: train_accuracy})
		train_writer.add_summary(train_acc_summary, i)
		test_acc_summary = sess.run(accuracy_summary_op, feed_dict={accuracy_placeholder: test_accuracy})
		test_writer.add_summary(test_acc_summary, i)
		[writer.flush() for writer in [train_writer, test_writer]]
		train_duration = round(time.time() - train_start, 2)

		# Output to montior training
		print('Epoch {0}, Training Loss: {1}, Test accuracy: {2}, time: {3}s, total time: {4}s'.format(i, ave_train_loss, test_accuracy, epoch_duration, train_duration))

	print('Total training time: {0}s'.format(train_duration))
	print('Confusion Matrix:')
	true_class=tf.argmax(Y, 1, name="true_class")
	predicted_class=tf.argmax(preds_op, 1, name="predicted_class")
	cm=tf.confusion_matrix(predicted_class,true_class, name="cm")
	print(sess.run(cm, feed_dict={X: x_val, Y: y_val}))

	# UNCOMMENT TO SAVE PREDICTION DATA
	# # Column 1: "ImageId"
	# imageId = [(i+1) for i in range(x_test.shape[0])]

	# # Column 2: "Label"
	# predictions = tf.argmax(preds_op, 1, name="predictions")
	# results = sess.run(predictions, feed_dict={X: x_test})

	# # Save
	# np.savetxt("random.csv", X=np.c_[imageId, results], fmt='%i', delimiter=',', header="ImageId,Label", comments='')

# Read dataset
x, y = read_train_dataset()
x_test = read_test_dataset()

# Shuffle dataset
x, y = shuffle(x, y, random_state=1)

# Split Data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=415)

# Define Hyperparameters
learning_rate = 0.001
batch_size = 128
n_training_epochs = 1
n_dim = x.shape[1]
n_class = y.shape[1]

# myconvnet Architecture
conv_filters = {
	'c1': [5, 5, 1, 1],
	'c2': [5, 5, 1, 1],
}

pool_filters = {
	'p1': [1, 2, 2, 1],
	'p2': [1, 2, 2, 1],
}

x = tf.placeholder(tf.float32, [None, n_dim], name="image_input")
y = tf.placeholder(tf.float32, [None, n_class], name="image_target_onehot")

# Create Model
if network == "onelayer":
	w, b, logits_op, preds_op, xentropy_op, loss_op = qfns.onelayer(x, y)
	[variable_summaries(v, name) for (v, name) in zip((w, b), ("w", "b"))]
	tf.summary.histogram('pre_activations', logits_op)
elif network == "twolayer":
	w1, b1, w2, b2, logits_op, preds_op, xentropy_op, loss_op = qfns.twolayer(x, y)
	[variable_summaries(v, name) for (v, name) in zip((w1, b1, w2, b2), ("w1", "b1", "w2", "b2"))]
	tf.summary.histogram('pre_activations', logits_op)
elif network == "conv":
	conv1out, conv2out, w, b, logits_op, preds_op, xentropy_op, loss_op = qfns.convnet(tf.reshape(x, [-1, 28, 28, 1]), y)
	[variable_summaries(v, name) for (v, name) in ((w,"w"), (b,"b"))]
	tf.summary.histogram('pre_activations', logits_op)
elif network == "myconvnet":
	conv1out, conv2out, pool1out, pool2out, w, b, logits_op, preds_op, xentropy_op, loss_op = qfns.myconvnet(tf.reshape(x, [-1, 28, 28, 1]), y, conv_filters, pool_filters)
	[variable_summaries(v, name) for (v, name) in ((w,"w"), (b,"b"))]
	tf.summary.histogram('pre_activations', logits_op)
	plot_image_on_grid(conv_filters, conv1out, conv2out)
else:
	raise ValueError("Incorrect network string")

with tf.name_scope('train_op'):
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Prediction and accuracy ops
accuracy_op = get_accuracy_op(preds_op, y)

image = tf.reshape(x[:1], [-1, 28, 28, 1])
tf.summary.image("image", image)

# TensorBoard for visualisation
# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
with tf.name_scope('summaries_op'):
	summaries_op = tf.summary.merge_all()

# Separate accuracy summary so we can use train and test sets
accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

# When run, the init_op initialises any tensorflow variables
# hint: weights and biases in our case
init_op = tf.global_variables_initializer()

# Get started
sess = tf.Session()
sess.run(init_op)

# Initialise TensorBoard Summary writers
dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
train_writer = tf.summary.FileWriter('./summaries/'+dtstr+'/train', sess.graph)
test_writer  = tf.summary.FileWriter('./summaries/'+dtstr+'/test')

# Train
print('Starting Training...')
train(sess, x_train, y_train, x_val, y_val, x_test, n_training_epochs, batch_size, summaries_op, accuracy_summary_op, train_writer, test_writer, x, y, train_op, loss_op, accuracy_op)
image = tf.reshape(x[:1], [-1, 28, 28, 1])
tf.summary.image("filtered_images_1", image)
print('Training Complete')

# Clean up
sess.close()

