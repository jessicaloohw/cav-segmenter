import numpy as np
import tensorflow as tf


def segment(x1):

    with tf.variable_scope('encoder1'):
        x2 = tf.layers.conv2d(x1, 32, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x2 = tf.layers.conv2d(x2, 32, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('encoder2'):
        x3 = tf.layers.max_pooling2d(x2, [2, 2], [2, 2], "VALID", name='pool')
        x3 = tf.layers.conv2d(x3, 64, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x3 = tf.layers.conv2d(x3, 64, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('encoder3'):
        x4 = tf.layers.max_pooling2d(x3, [2, 2], [2, 2], "VALID", name='pool')
        x4 = tf.layers.conv2d(x4, 128, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x4 = tf.layers.conv2d(x4, 128, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('encoder4'):
        x5 = tf.layers.max_pooling2d(x4, [2, 2], [2, 2], "VALID", name='pool')
        x5 = tf.layers.conv2d(x5, 256, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x5 = tf.layers.conv2d(x5, 256, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('bottleneck'):
        x6 = tf.layers.max_pooling2d(x5, [2, 2], [2, 2], "VALID", name='pool')
        x6 = tf.layers.conv2d(x6, 512, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x6 = tf.layers.conv2d(x6, 512, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('skip4'):
        x6 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(x6)
        x7 = tf.concat([x5, x6], axis=3, name='concat')

    with tf.variable_scope('decoder4'):
        x7 = tf.layers.conv2d(x7, 256, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x7 = tf.layers.conv2d(x7, 256, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('skip3'):
        x7 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(x7)
        x8 = tf.concat([x4, x7], axis=3, name='concat')

    with tf.variable_scope('decoder3'):
        x8 = tf.layers.conv2d(x8, 128, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x8 = tf.layers.conv2d(x8, 128, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('skip2'):
        x8 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(x8)
        x9 = tf.concat([x3, x8], axis=3, name='concat')

    with tf.variable_scope('decoder2'):
        x9 = tf.layers.conv2d(x9, 64, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x9 = tf.layers.conv2d(x9, 64, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('skip1'):
        x9 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(x9)
        x10 = tf.concat([x2, x9], axis=3, name='concat')

    with tf.variable_scope('decoder1'):
        x10 = tf.layers.conv2d(x10, 32, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv1')
        x10 = tf.layers.conv2d(x10, 32, [3, 3], [1, 1], "SAME", activation=tf.nn.relu, use_bias=True, name='conv2')

    with tf.variable_scope('logits'):
        logits = tf.layers.conv2d(x10, 1, [1, 1], [1, 1], "SAME", activation=tf.nn.sigmoid, use_bias=True, name='conv')

    return logits


def slice_image(image, slice_size, overlap_size):

    # Dimensions:
    nCols = image.shape[1]

    # Calculate:
    gapBetweenSlices = slice_size - overlap_size

    # Initialise:
    slices = []
    slice_idxs = []

    # Slice:
    for sliceStart in range(0, nCols, gapBetweenSlices):

        # Make sure slice does not exceed image:
        endIdx = np.min([sliceStart + slice_size, nCols])
        startIdx = endIdx - slice_size

        # If it already exists, do not repeat:
        if not np.any(np.array(slice_idxs) == startIdx):
            slices.append(image[:, startIdx:endIdx, :])
            slice_idxs.append(startIdx)

    return np.array(slices), np.array(slice_idxs)


def stitch_images(slice_prediction, slice_idxs, num_cols):

    # Initialise:
    image_prediction = np.zeros([slice_prediction.shape[1], num_cols])
    count_prediction = np.zeros([slice_prediction.shape[1], num_cols])

    # Slice parameters:
    sliceSize = slice_prediction.shape[2]
    numSlices = slice_prediction.shape[0]

    # Stitch:
    for i in range(numSlices):

        startIdx = slice_idxs[i]
        endIdx = startIdx + sliceSize

        image_prediction[:, startIdx:endIdx] += slice_prediction[i, :, :]
        count_prediction[:, startIdx:endIdx] += 1

    # Average:
    image_prediction = image_prediction / count_prediction

    # Remove NaN and Inf:
    image_prediction[~np.isfinite(image_prediction)] = 0

    return image_prediction
