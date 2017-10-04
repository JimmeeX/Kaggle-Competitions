NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = 28*28

import tensorflow as tf

def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """

 
    # m = Batch size
    # X: mx784 (m: no. data, 784: no. pixels/features)
    # Y: mx10 (m: no. data, 10: no. classification outputs)
    # layersize=10 # what does this mean?

    # Initialise weights + bias
    w = tf.Variable(tf.zeros([IMAGE_PIXELS, layersize]), name="w") # 784x10
    # w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, layersize],
    #                        stddev=1.0 / tf.sqrt(float(IMAGE_PIXELS)))) # #Similar accuracy
    b = tf.Variable(tf.zeros([layersize]), name="b") # Array of 10

    # Linear combination + activation function
    logits = tf.add(tf.matmul(X, w), b, name="logits") # m * 10 + b
    preds = tf.nn.softmax(logits, name="preds") # Apply softmax

    # Get loss
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="batch_xentropy")
    batch_loss = tf.reduce_mean(batch_xentropy, name="batch_loss")

    return w, b, logits, preds, batch_xentropy, batch_loss

def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """

    # Initialise weights + bias
    w1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hiddensize],
                            stddev=1.0 / tf.sqrt(float(IMAGE_PIXELS))), name="weight1") # 784x30
    w2 = tf.Variable(tf.truncated_normal([hiddensize, outputsize],
                            stddev=1.0 / tf.sqrt(float(hiddensize))), name="weight2") # 30x10
    # w1 = tf.Variable(tf.random_normal([IMAGE_PIXELS, hiddensize])) #Test Accuracy: 92%
    # w2 = tf.Variable(tf.random_normal([hiddensize, outputsize]))
    b1 = tf.Variable(tf.zeros([hiddensize]), name="bias1") # 30
    b2 = tf.Variable(tf.zeros([outputsize]), name="bias2") # 10

    # Forward Propagation
    z2 = tf.add(tf.matmul(X, w1), b1, name="z2") # m x 30
    a2 = tf.nn.relu(z2, name="a2") # m x 30
    logits = tf.add(tf.matmul(a2, w2), b2, name="logits") # m x 10 + b2
    preds = tf.nn.softmax(logits, name="preds") # m x 10

    # Get loss
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="batch_xentropy")
    batch_loss = tf.reduce_mean(batch_xentropy, name="batch_loss")

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss

def convnet(X, Y, convlayer_sizes=[10, 10], \
        filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """

    # X: [BATCH_SIZE, 28, 28, 1]

    # Convolutional
    conv1 = tf.layers.conv2d(inputs=X, filters=convlayer_sizes[0], kernel_size=filter_shape, activation=tf.nn.relu, padding=padding, name="conv1") # [BATCH_SIZE, 28, 28, 10]
    conv2 = tf.layers.conv2d(inputs=conv1, filters=convlayer_sizes[1], kernel_size=filter_shape, activation=tf.nn.relu, padding=padding, name="conv2") # [BATCH_SIZE, 28, 28, 10]
    conv2_flat = tf.reshape(conv2, [-1, IMAGE_SIZE * IMAGE_SIZE * convlayer_sizes[1]]) # [m x 7840]

    # Weights + bias for fully connected layer
    w = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE * convlayer_sizes[1], outputsize],
                            stddev=1.0 / tf.sqrt(float(IMAGE_SIZE * IMAGE_SIZE * convlayer_sizes[1]))), name="w") # 10 x 10
    b = tf.Variable(tf.constant(0.1), [outputsize], name="b") # 10

    # Forward Propagation
    logits = tf.add(tf.matmul(conv2_flat, w), b, name="logits") # m x 10
    preds = tf.nn.softmax(logits, name="preds")

    # Get loss/cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="batch_xentropy") # 128 x 10
    batch_loss = tf.reduce_mean(batch_xentropy, name="batch_loss")

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss

def myconvnet(X, Y, conv_filters, pool_filters, strides=[1, 1, 1, 1], padding="SAME", outputsize=10):
    # Conv -> Relu -> Pool -> Conv -> Relu -> Pool -> FC -> Softmax

    # Parameters
    K1 = conv_filters['c1'][3] # Number of Filters
    K2 = conv_filters['c2'][3]
    filter1 = tf.Variable(tf.random_normal(conv_filters['c1']), name="filter1")
    filter2 = tf.Variable(tf.random_normal(conv_filters['c2']), name="filter2")

    # Convolutional1
    conv1 = tf.nn.relu(tf.nn.conv2d(input=X, filter=filter1, strides=strides, padding=padding, name="conv1")) # M * 28 * 28 * K1

    # Pooling1
    pool1 = tf.nn.max_pool(value=conv1, ksize=pool_filters['p1'], strides=strides, padding=padding, name="pool1") # M * 28 * 28 * K1
    #pool_flat = tf.reshape(pool, [-1, IMAGE_SIZE * IMAGE_SIZE * K])

    # Convolutional2
    conv2 = tf.nn.relu(tf.nn.conv2d(input=pool1, filter=filter2, strides=strides, padding=padding, name="conv2")) # M * 28 * 28 * K1

    # Pooling2
    pool2 = tf.nn.max_pool(value=conv2, ksize=pool_filters['p2'], strides=strides, padding=padding, name="pool2")
    pool_flat = tf.reshape(pool2, [-1, IMAGE_SIZE * IMAGE_SIZE * K2])

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop = tf.nn.dropout(pool_flat, keep_prob, name="drop")

    # Fully Connected layer
    w = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE * K2, outputsize],
                           stddev=1.0 / tf.sqrt(float(IMAGE_SIZE * IMAGE_SIZE * K2))), name="w")    
    b = tf.Variable(tf.constant(0.1), [outputsize], name="b")

    # Forward Propagation
    logits = tf.add(tf.matmul(pool_flat, w), b, name="logits")
    preds = tf.nn.softmax(logits, name="preds")

    # Get loss/cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="batch_xentropy") # 128 x 10
    batch_loss = tf.reduce_mean(batch_xentropy, name="batch_loss")

    return conv1, conv2, pool1, pool2, w, b, logits, preds, batch_xentropy, batch_loss

def train_step(sess, batchX, batchY, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = sess.run([train_op, loss_op, summaries_op], feed_dict={X: batchX, Y: batchY})
    return train_result, loss, summary
