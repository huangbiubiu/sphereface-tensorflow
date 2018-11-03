import argparse
import math
import os
import time
from model import GraphType
import tensorflow as tf

from datasets.Dataset import Dataset
from datasets.lfw import LFW
from model.NaiveCNN import NaiveCNN
from model.NerualNetwork import NerualNetwork
from model.SphereCNN import SphereCNN
from model.loss import softmax_loss

import numpy as np
import datasets.webface


class Trainer:
    def __init__(self, sess_config,
                 dataset_path,
                 cnn_model,
                 cnn_param,
                 logdir,
                 args,
                 epoch_num,
                 batch_size,
                 data_loader):
        self.session = tf.Session(config=sess_config)
        self.dataset_path = dataset_path
        self.cnn_model = cnn_model
        self.cnn_param = cnn_param
        self.logdir = logdir
        self.args = args
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.data_loader = data_loader
        self.training_step = 0

        self.train_writer, self.train_op, self.train_acc, self.loss, self.global_step, self.summary_op, self.saver, self.acc_summary_op = self.build_graph()

    def build_graph(self):
        sess = self.session
        with sess.graph.as_default():
            # build training graph
            train_op, train_acc, _, loss, global_step, summary_op, saver, acc_summary_op = build_graph(
                dataset_path=self.dataset_path,
                graph_type=GraphType.TRAIN,
                epoch_num=self.epoch_num,
                batch_size=self.batch_size,
                cnn_model=self.cnn_model,
                cnn_param=self.cnn_param,
                sess=sess,
                log_dir=self.logdir,
                data_param={},
                data_loader=self.data_loader,
                base_lr=self.args.learning_rate,
                base_lambda=self.args.base_lambda)
            train_writer = tf.summary.FileWriter(os.path.join(self.logdir, 'train'), sess.graph)
            sess.graph.finalize()

        return train_writer, train_op, train_acc, loss, global_step, summary_op, saver, acc_summary_op

    def train_once(self, max_training_step, max_step_this_run):
        sess = self.session
        is_final_eval = False
        is_training_complete = False

        with sess.as_default():
            with sess.graph.as_default():

                try:
                    self.training_step += 1

                    start_time = time.time()

                    _, loss_value, acc, step, summary, acc_summary = sess.run(
                        [self.train_op, self.loss, self.train_acc, self.global_step, self.summary_op,
                         self.acc_summary_op])

                    end_time = time.time()
                    duration = end_time - start_time

                    tf.logging.info(f'step: %d loss: %.3f acc: %.3f time: %.3f' % (step, loss_value, acc, duration))
                    if step % 100 == 0:  # save variables every 100 steps
                        self.train_writer.add_summary(summary, step)
                        self.train_writer.add_summary(acc_summary, step)

                        if os.path.isfile(self.logdir):
                            model_save_path = os.path.dirname(self.logdir)
                        else:
                            model_save_path = self.logdir
                        self.saver.save(sess, os.path.join(os.path.expanduser(model_save_path), 'model.ckpt'),
                                        global_step=step)

                    if max_training_step is not None and step > max_training_step:
                        is_final_eval = True
                    if max_step_this_run is not None and self.training_step > max_step_this_run:
                        is_final_eval = True
                except tf.errors.OutOfRangeError:  # training completed
                    is_final_eval = True
                    is_training_complete = True

        return is_final_eval, is_training_complete


def parse_arg(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='the path of training dataset')
    parser.add_argument('--eval_path', type=str, help='the path of evaluation dataset')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=None, help='total training epoch number')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--base_lambda', type=float, default=1000, help='base lambda')

    parser.add_argument('--max_step', type=int, default=None, help='max global training steps')
    parser.add_argument('--train_step', type=int, default=None, help='training step at this run')

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
                graph_type: GraphType,
                epoch_num: int,
                batch_size: int,
                cnn_model: NerualNetwork.__class__,
                cnn_param: dict,
                sess: tf.Session,
                data_param: dict,
                data_loader: Dataset,
                base_lr=1e-4,
                base_lambda=1000):
    with sess.graph.as_default():
        global_step = tf.train.get_or_create_global_step(graph=sess.graph)

        weight_regularizer = tf.contrib.layers.l2_regularizer(0.0005)

        is_training = graph_type == GraphType.TRAIN
        dataset, num_class = data_loader.load_data(dataset_path, is_training, epoch_num, batch_size,
                                                   data_param=data_param)

        image, label = dataset

        model = cnn_model()

        if graph_type == GraphType.EVAL:
            logits = model.inference(image,
                                     num_class,
                                     label=None,
                                     param={**cnn_param,
                                            'global_steps': global_step,
                                            'weight_regularizer': weight_regularizer,
                                            'graph_type': graph_type})

            saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

            # initialize variables
            if os.path.isdir(log_dir):
                latest_ckpt = tf.train.latest_checkpoint(os.path.expanduser(log_dir))
            else:
                latest_ckpt = log_dir
            if latest_ckpt is not None:
                # restore model
                saver.restore(sess, latest_ckpt)
            else:
                raise ValueError(f"No model to evaluate in {log_dir}")

            # return normalized features and corresponding labels
            return logits / tf.reshape(tf.norm(logits, axis=1), (-1, 1)), label, global_step

        logits, base_loss = model.inference(image,
                                            num_class,
                                            label=label,
                                            param={**cnn_param,
                                                   'global_steps': global_step,
                                                   'weight_regularizer': weight_regularizer,
                                                   'graph_type': graph_type,
                                                   'base_lambda': base_lambda})

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name="loss")
        tf.summary.scalar("loss", loss)

        # SGD training strategy
        # base_lr = 1e-5
        #
        # def lr_decay(step):
        #     """
        #     calculate learning rate
        #     same as multistep in caffe
        #     see https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/sgd_solver.cpp#L54
        #     :param step: stepvalue
        #     :return: learning rate for corresponding stepvalue
        #     """
        #     gamma = 0.1
        #     return base_lr * math.pow(gamma, step)
        #
        # boundaries = [16000, 24000, 28000]
        # values = [base_lr, *list(map(lr_decay, boundaries))]
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        # tf.summary.scalar("learning_rate", learning_rate)
        # train_op = tf.train.MomentumOptimizer(momentum=0.9,
        #                                       name='optimizer',
        #                                       learning_rate=learning_rate).minimize(loss, global_step=global_step)

        with tf.name_scope("learning_rate"):
            # Adam strategy
            def lr_decay_values(boundaries, gamma=0.1):
                v = [base_lr]
                for i in range(len(boundaries)):
                    if i == 0:
                        last_boundary = 0
                    else:
                        last_boundary = boundaries[i - 1]
                    this_boundary = boundaries[i]
                    interval = this_boundary - last_boundary
                    v.append(base_lr * math.pow(gamma, math.floor(this_boundary / interval)))
                return v

            boundaries = [16000, 24000, 28000]
            values = lr_decay_values(boundaries)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.AdamOptimizer(name='optimizer', learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # gvs = optimizer.compute_gradients(loss)
        # grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in gradients]

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

        # initialize variables
        if os.path.isdir(log_dir):
            latest_ckpt = tf.train.latest_checkpoint(os.path.expanduser(log_dir))
        else:
            latest_ckpt = log_dir
        sess.run(tf.global_variables_initializer())
        if latest_ckpt is not None:
            # restore model
            tf.logging.info(f"loading variables from {latest_ckpt}")
            saver.restore(sess, latest_ckpt)

        # add accuracy node
        with tf.name_scope("accuracy"):
            if graph_type == GraphType.TRAIN:
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=1), tf.argmax(logits, axis=1)), tf.float32))
                acc_op = None
            elif graph_type == GraphType.TEST:
                acc, acc_op = tf.metrics.accuracy(tf.argmax(label, axis=1), tf.argmax(logits, axis=1))
            else:
                raise ValueError(f"Illegal argument graph_type: {graph_type}")
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
                       eval_every_step=1000,
                       max_training_step=None,
                       max_step_this_run=None):
    if eval_every_step is not None and eval_every_step < 100:
        tf.logging.warn(f"Evaluation frequency {eval_every_step} seems too small. Is it intentional?")

    data_loader = datasets.webface.WebFace()
    trainer = Trainer(sess_config=sess_config,
                      dataset_path=dataset_path,
                      cnn_model=cnn_model,
                      cnn_param=cnn_param,
                      logdir=logdir,
                      args=args,
                      epoch_num=epoch_num,
                      batch_size=batch_size,
                      data_loader=data_loader)

    is_final_eval = False

    while not is_final_eval:
        step_in_this_loop = 0
        while eval_every_step is None or step_in_this_loop <= eval_every_step:
            step_in_this_loop += 1
            is_final_eval, is_training_complete = trainer.train_once(max_step_this_run=max_step_this_run,
                                                                     max_training_step=max_training_step)
            if is_training_complete:
                break

        acc = evaluate(cnn_model=cnn_model,
                       cnn_param=cnn_param,
                       batch_size=batch_size,
                       sess_config=sess_config,
                       logdir=logdir,
                       eval_path=args.eval_path)
        if is_final_eval:
            tf.logging.info("--------Final Evaluation--------")
            tf.logging.info(f"Accuracy: {acc}")
            tf.logging.info('Training completed.')

            break
        pass


def evaluate(cnn_model,
             cnn_param,
             batch_size,
             sess_config,
             logdir,
             eval_path):
    """Evaluate on LFW dataset"""
    tf.reset_default_graph()
    with tf.Session(config=sess_config) as sess:
        tf.logging.info("--------START EVALUATION--------")

        dataset = LFW(eval_path)

        tf.logging.info("loading evaluation graph")
        logits_op, file_name_op, step_op = build_graph(
            dataset_path=eval_path,
            graph_type=GraphType.EVAL,
            epoch_num=1,
            batch_size=batch_size,
            cnn_model=cnn_model,
            cnn_param=cnn_param,
            sess=sess,
            log_dir=logdir,
            data_param={},
            data_loader=dataset)
        tf.logging.info("evaluation graph loaded")

        logits_flag = False
        batch_count = 0
        while True:
            try:
                logits_, file_name_, step = sess.run([logits_op, file_name_op, step_op])
                if not logits_flag:
                    logits = logits_
                    file_name = file_name_
                    logits_flag = True
                else:
                    logits = np.vstack((logits, logits_))
                    file_name = np.hstack((file_name, file_name_))
                batch_count += 1
            except tf.errors.OutOfRangeError:
                break

        embeddings = {}
        for i in range(len(logits)):
            embeddings[str(file_name[i], encoding='utf8')] = logits[i]

        eval_acc = dataset.evaluation(embeddings)

        # add to tf summary
        eval_writer = tf.summary.FileWriter(os.path.join(logdir, 'eval'), sess.graph)
        summary = tf.Summary()
        summary.value.add(tag='acc', simple_value=eval_acc)
        eval_writer.add_summary(summary, global_step=step)

        # make sure all summaries are written to disk
        eval_writer.flush()
        eval_writer.close()

        tf.logging.info(f"Evaluation accuracy: {eval_acc}")
        tf.logging.info("--------EVALUATION FINISHED--------")
    return eval_acc


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
                       epoch_num=args.epoch_num,
                       batch_size=args.batch_size,
                       cnn_model=cnn_model,
                       cnn_param={'softmax': args.softmax_type},
                       sess_config=config,
                       logdir=args.log_dir,
                       eval_every_step=1000,
                       max_training_step=args.max_step,
                       max_step_this_run=args.train_step,
                       args=args)


def debug_eval():
    # Set GPU configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    evaluate('/home/hyh/projects/benchmark/data/lfw',
             cnn_model=SphereCNN,
             cnn_param={'softmax': 'a-softmax'},
             batch_size=512,
             sess_config=config,
             logdir='/home/hyh/projects/benchmark/saved_models/webface',
             eval_path='/home/hyh/projects/benchmark/data/lfw'
             )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
