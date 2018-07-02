import argparse
import math
import os
import time

import tensorflow as tf

from datasets.webface import load_data
from model.NaiveCNN import NaiveCNN
from model.NerualNetwork import NerualNetwork
from model.SphereCNN import SphereCNN
from model.loss import softmax_loss


def parse_arg(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='the path of dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='the path of dataset')

    parser.add_argument('--fail_path', type=str, help='the path of images which failed to detect faces')
    parser.add_argument('--bounding_box_path', type=str, help='the path of face bounding boxes')

    parser.add_argument('--margin', type=int, default=44, help='cropping margin')

    parser.add_argument('--log_dir', type=str, help='log output dir')
    parser.add_argument('--softmax_type', type=str, choices=['vanilla', 'a-softmax'], help='softmax layer type')
    parser.add_argument('--cnn_model', type=str, choices=['naive', 'a-softmax'], help='cnn_structure')

    args = parser.parse_args(argv[1:])

    return args


def build_graph(dataset_path: str,
                log_dir: str,
                is_training: bool,
                epoch_num: int,
                batch_size: int,
                cnn_model: NerualNetwork.__class__,
                cnn_param: dict,
                sess: tf.Session,
                data_param: dict):
    with sess.graph.as_default():
        global_step = tf.train.get_or_create_global_step(graph=sess.graph)

        # weight_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        weight_regularizer = None

        dataset, num_class = load_data(dataset_path, is_training, epoch_num, batch_size, data_param)
        image, label = dataset

        model = cnn_model()
        logits = model.inference(image,
                                 num_class,
                                 param={**cnn_param,
                                        **{'global_steps': global_step, 'image_size': 100},
                                        'weight_regularizer': weight_regularizer})
        base_loss = softmax_loss(logits, label)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name="loss")
        tf.summary.scalar("loss", loss)

        # train_op = tf.train.AdamOptimizer(name='optimizer').minimize(loss, global_step=global_step)
        base_lr = 0.01

        def lr_decay(step):
            """
            calculate learning rate
            same as multistep in caffe
            see https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/sgd_solver.cpp#L54
            :param step: stepvalue
            :return: learning rate for corresponding stepvalue
            """
            gamma = 0.1
            return base_lr * math.pow(gamma, step)

        boundaries = [16000, 24000, 28000]
        values = [base_lr, *list(map(lr_decay, boundaries))]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        # train_op = tf.train.MomentumOptimizer(momentum=0.9,
        #                                       name='optimizer',
        #                                       learning_rate=learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.train.AdamOptimizer(name='optimizer', learning_rate=5e-4).minimize(loss, global_step=global_step)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=5)

        # initialize variables
        latest_ckpt = tf.train.latest_checkpoint(os.path.expanduser(log_dir))
        if latest_ckpt is not None:
            # restore model
            saver.restore(sess, latest_ckpt)
        else:
            sess.run(tf.global_variables_initializer())

        # add accuracy node
        with tf.name_scope("accuracy"):
            if is_training:
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=1), tf.argmax(logits, axis=1)), tf.float32))
                acc_op = None
            else:
                acc, acc_op = tf.metrics.accuracy(tf.argmax(label, axis=1), tf.argmax(logits, axis=1))
            tf.summary.scalar("acc", acc)
            acc_summary = tf.summary.merge_all()

        sess.run(tf.local_variables_initializer())

        # return train_op, acc, acc_op, loss, global_step, summary_op, saver
        return train_op, acc, acc_op, loss, global_step, summary_op, saver, acc_summary


def train_and_evaluate(dataset_path,
                       epoch_num,
                       batch_size,
                       cnn_model,
                       cnn_param,
                       sess_config,
                       logdir,
                       args,
                       eval_every_step=1000):
    is_final_eval = False
    while not is_final_eval:
        training_step = 0
        tf.reset_default_graph()
        with tf.Session(config=sess_config) as sess:
            # build training graph
            train_op, train_acc, _, loss, global_step, summary_op, saver, acc_summary_op = build_graph(
                dataset_path=dataset_path,
                is_training=True,
                epoch_num=epoch_num,
                batch_size=batch_size,
                cnn_model=cnn_model,
                cnn_param=cnn_param,
                sess=sess,
                log_dir=logdir,
                data_param={'fail_path': args.fail_path,
                            'bounding_boxes': args.bounding_box_path,
                            'margin': args.margin})
            train_writer = tf.summary.FileWriter(os.path.join(logdir, 'train'), sess.graph)
            while training_step <= eval_every_step:
                try:
                    training_step += 1

                    start_time = time.time()

                    _, loss_value, acc, step, summary, acc_summary = sess.run(
                        [train_op, loss, train_acc, global_step, summary_op, acc_summary_op])

                    end_time = time.time()
                    duration = end_time - start_time

                    # print(f'loss: {loss_value}\t acc:{acc}\t time:{duration}')
                    tf.logging.info(f'step: %d loss: %.3f acc: %.3f time: %.3f' % (step, loss_value, acc, duration))
                    if step % 100 == 0:
                        train_writer.add_summary(summary, step)
                        train_writer.add_summary(acc_summary, step)
                        saver.save(sess, os.path.join(os.path.expanduser(logdir), 'model.ckpt'),
                                   global_step=step)
                except tf.errors.OutOfRangeError:  # training completed
                    is_final_eval = True
                    break

        acc = evaluate(dataset_path=dataset_path,
                       cnn_model=cnn_model,
                       cnn_param=cnn_param,
                       batch_size=batch_size,
                       sess_config=sess_config,
                       logdir=logdir)
        if is_final_eval:
            tf.logging.info("--------Final Evaluation--------")
            tf.logging.info(f"Accuracy: {acc}")
            tf.logging.info('Training completed.')


def evaluate(dataset_path,
             cnn_model,
             cnn_param,
             batch_size,
             sess_config,
             logdir):
    # tf.reset_default_graph()
    # with tf.Session(config=sess_config) as sess:
    #     tf.logging.info("--------Start Evaluation--------")
    #     tf.logging.info("loading evaluation graph")
    #
    #     train_op, eval_acc, acc_op, loss, global_step, summary_op, saver, acc_summary_op = build_graph(
    #         dataset_path=dataset_path,
    #         is_training=False,
    #         epoch_num=1,
    #         batch_size=batch_size,
    #         cnn_model=cnn_model,
    #         cnn_param=cnn_param,
    #         sess=sess,
    #         log_dir=logdir)
    #
    #     eval_writer = tf.summary.FileWriter(os.path.join(logdir, 'eval'), sess.graph)
    #
    #     while True:
    #         try:
    #             loss_value, acc, _, summary, acc_summary, step = sess.run(
    #                 [loss, eval_acc, acc_op, summary_op, acc_summary_op, global_step])
    #         except tf.errors.OutOfRangeError:
    #             eval_writer.add_summary(summary, global_step=step)
    #             eval_writer.add_summary(acc_summary, global_step=step)
    #
    #             # make sure all summaries are written to disk
    #             eval_writer.flush()
    #             eval_writer.close()
    #
    #             tf.logging.info("--------Evaluation Competed--------")
    #             return acc
    tf.logging.warn("EVALUATION IS NOT IMPLEMENTED.")
    return 0  # TODO implement evaluation code


def save_args(args, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'arguments.txt'), 'w', encoding='utf-8') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")


def main(argv):
    args = parse_arg(argv)
    save_args(args, args.log_dir)

    # Set GPU configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    cnn_model = NaiveCNN if args.cnn_model == 'naive' else SphereCNN

    train_and_evaluate(dataset_path=args.dataset_path,
                       epoch_num=None,
                       batch_size=args.batch_size,
                       cnn_model=cnn_model,
                       cnn_param={'softmax': args.softmax_type},
                       sess_config=config,
                       logdir=args.log_dir,
                       eval_every_step=1000,
                       args=args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
