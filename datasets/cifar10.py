# -*- coding: utf-8 -*-
import os
import tensorflow as tf

from datasets.preprocess import image_preprocess


def load_data(data_dir, is_training, epoch_num, batch_size):
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.

    if is_training:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    # filename_queue = tf.train.string_input_producer(filenames)

    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    dataset = dataset.prefetch(batch_size)

    # Split each entry to feature and label
    def decode_data(value):
        record = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        label = tf.cast(record[0], tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        # there use label_bytes + image_bytes + 1 in resnet implementation:
        # https://github.com/tensorflow/models/blob/master/official/resnet/cifar10_main.py#L80
        depth_major = tf.reshape(record[label_bytes: label_bytes + image_bytes],
                                 [depth, height, width])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.transpose(depth_major, [1, 2, 0])

        # Convert label to one-hot
        label = tf.one_hot(label, 10)  # For CIFAR-10

        return image, label

    dataset = dataset.map(decode_data).map(lambda image, label: image_preprocess(image, label))

    # Shuffle, repeat, and batch the examples.
    if is_training:
        dataset = dataset.shuffle(1000).repeat(epoch_num).batch(batch_size)
    else:
        dataset = dataset.shuffle(1000).batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
