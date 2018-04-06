    # Conv -> Relu -> Pool -> FC
    # POOLING LAYER
    # FULLY CONNECTED LAYER

    # Convolutional Layer
    # input_ = X.get_shape().as_list()
    # input_W = input_[1] # Input Width
    # input_H = input_[2] # Input Height
    # input_C = input_[3] # Input Channel
    # filter_W = filter_shape[0] # Filter Width
    # filter_H = filter_shape[1] # Filter Height
    K = conv1_filter[3] # Number of Filters
    # stride_W = stride[1] # Stride Width?
    # stride_H = stride[2] # Stride Height?
    # P = 0 # No Padding

    # output_W = int((input_W - filter_W + 2*P) / stride_W + 1) # Output Width
    # output_H = int((input_H - filter_H + 2*P) / stride_H + 1) # Output Height

    # EG: 
    # X: [BATCH_SIZE, 28, 28, 1]
    # Filter: [5, 5, 1, 32]
    # Expected Output: [BATCH_SIZE, 24, 24, 32] if padding= "VALID"

    # Convolutional
    filter_ = tf.Variable(tf.random_normal(conv1_filter), name="filter_")
    conv = tf.nn.conv2d(input=X, filter=filter_, strides=strides, padding=padding, name="conv")
    #conv_flat = tf.reshape(conv, [-1, output_W * output_H * K])
    #conv_flat = tf.reshape(conv, [-1, IMAGE_SIZE * IMAGE_SIZE * K])

    # Relu
    relu = tf.nn.relu(conv, name="relu")

    # Pooling
    pool = tf.nn.max_pool(relu, ksize=kernel_shape, strides=strides, padding=padding, name="pool")
    pool_flat = tf.reshape(pool, [-1, IMAGE_SIZE * IMAGE_SIZE * K])

    # Fully Connected layer
    #w = tf.Variable(tf.truncated_normal([output_W * output_H * K, outputsize],
    #                       stddev=1.0 / tf.sqrt(float(output_W * output_H * K))), name="w")
    w = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE * K, outputsize],
                           stddev=1.0 / tf.sqrt(float(IMAGE_SIZE * IMAGE_SIZE * K))), name="w")    
    b = tf.Variable(tf.constant(0.1), [outputsize], name="b")

    # Forward Propagation
    logits = tf.add(tf.matmul(pool_flat, w), b, name="logits")
    preds = tf.nn.softmax(logits, name="preds")

    # Get loss/cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="batch_xentropy") # 128 x 10
    batch_loss = tf.reduce_mean(batch_xentropy, name="batch_loss")