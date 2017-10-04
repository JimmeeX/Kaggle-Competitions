import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_dataset():
	df = pd.read_csv("train.csv")
	y = df.pop("label")
	X = df / 255

	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	Y = one_hot_encode(y)
	return (X, Y)

def one_hot_encode(labels):
	n_labels = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels, n_unique_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode

# Read dataset
x, y = read_dataset()

# Shuffle dataset
x, y = shuffle(x, y, random_state=1)

# Split Data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=415)

# Define Hyperparameters
learning_rate = 0.3

batch_size = 128


training_epochs = 50
cost_history = np.empty(shape=[1], dtype=float)
n_dim = x.shape[1]
n_class = 10
model_path = "Visualisations"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

# Define the model
def multilayer_perceptron(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.sigmoid(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.sigmoid(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.sigmoid(layer_3)

	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
	layer_4 = tf.nn.relu(layer_4)

	out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
	return out_layer

# Define weights + biases
weights = {
	'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
	'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
	'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
	'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
	'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
}

biases = {
	'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
	'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
	'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
	'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
	'out': tf.Variable(tf.truncated_normal([n_class])),
}

# Initialise Variables
init = tf.global_variables_initializer()

# To save our model
saver = tf.train.Saver()

# Call your model defined
y = multilayer_perceptron(x, weights, biases)

# Define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
	sess.run(training_step, feed_dict={x: x_train, y_: y_train})
	cost = sess.run(cost_function, feed_dict={x: x_train, y_: y_train})
	cost_history = np.append(cost_history, cost)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	pred_y = sess.run(y, feed_dict={x: x_val})
	mse = tf.reduce_mean(tf.square(pred_y - y_val))
	mse_ = sess.run(mse)
	mse_history.append(mse_)
	accuracy = (sess.run(accuracy, feed_dict={x: x_train, y_: y_train}))
	accuracy_history.append(accuracy)

	print('epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

# Plot mse and accuracy graph
plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

# Print the final accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: x_val, y_: y_val})))

# Print the final mean square error
pred_y = sess.run(y, feed_dict={x: x_val})
mse = tf.reduce_mean(tf.square(pred_y - y_val))
print("MSE: %.4f" & sess.run(mse))




