import tensorflow as tf
import argparse
import datasets.cifar10
from model.NaiveCNN import NaiveCNN
from model.loss import softmax_loss


def parse_arg(argv) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='the path of dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='the path of dataset')
    args = parser.parse_args(argv[1:])

    return args


def main(argv):
    args = parse_arg(argv)

    # define model
    image_batch, label_batch = datasets.cifar10.load_data(args.dataset_path, None, args.batch_size)
    model = NaiveCNN()
    logits = model.inference(image_batch, 10)
    loss = softmax_loss(logits, label_batch)
    tf.summary.scalar("loss", loss)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Set GPU configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        while True:
            _, loss_value = sess.run([train_op, loss])
            print(loss_value)


if __name__ == '__main__':
    tf.app.run()
