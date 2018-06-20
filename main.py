import argparse
import os
import time

import tensorflow as tf

from datasets.cifar10 import load_data
from model.NaiveCNN import NaiveCNN
from model.NerualNetwork import NerualNetwork
from model.loss import softmax_loss


def parse_arg(argv) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='the path of dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='the path of dataset')
    parser.add_argument('--log_dir', type=str, help='log output dir')

    args = parser.parse_args(argv[1:])

    return args


def build_graph(dataset_path: str,
                is_training: bool,
                epoch_num: int,
                batch_size: int,
                cnn_model: NerualNetwork,
                sess: tf.Session):
    with sess.graph.as_default():
        global_step = tf.train.get_or_create_global_step(graph=sess.graph)

        image, label = load_data(dataset_path, is_training, epoch_num, batch_size)

        model = cnn_model()
        logits = model.inference(image, 10, param={'global_steps': global_step})
        loss = softmax_loss(logits, label)
        tf.summary.scalar("loss", loss)

        train_op = tf.train.AdamOptimizer(name='optimizer').minimize(loss, global_step=global_step)

        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=1), tf.argmax(logits, axis=1)), tf.float32))
        tf.summary.scalar("train/acc", train_acc)

        eval_acc, eval_acc_op = tf.metrics.accuracy(tf.argmax(label, axis=1), tf.argmax(logits, axis=1))
        tf.summary.scalar("eval/acc", eval_acc)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        return train_op, eval_acc_op, eval_acc, train_acc, loss, global_step, summary_op, saver


def main(argv):
    args = parse_arg(argv)

    # Set GPU configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # build training graph
        train_op, _, _, train_acc_op, loss, global_step, summary_op, saver = build_graph(dataset_path=args.dataset_path,
                                                                                         is_training=True,
                                                                                         epoch_num=None,
                                                                                         batch_size=args.batch_size,
                                                                                         cnn_model=NaiveCNN,
                                                                                         sess=sess)

        # initialize variables
        latest_ckpt = tf.train.latest_checkpoint(os.path.expanduser(args.log_dir))
        if latest_ckpt is not None:
            # restore model
            saver.restore(sess, latest_ckpt)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sess.graph.finalize()

        log_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        while True:
            start_time = time.time()

            _, loss_value, acc, step, summary = sess.run([train_op, loss, train_acc_op, global_step, summary_op])

            end_time = time.time()
            duration = end_time - start_time
            # print(f'loss: {loss_value}\t acc:{acc}\t time:{duration}')
            print(f'step: %d loss: %.3f acc: %.3f time: %.3f' % (step, loss_value, acc, duration))
            if step % 100 == 0:
                log_writer.add_summary(summary, step)
                saver.save(sess, os.path.join(os.path.expanduser(args.log_dir), 'model.ckpt'), global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
