import argparse
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
                graph_type: GraphType,
                epoch_num: int,
                batch_size: int,
                cnn_model: NerualNetwork.__class__,
                cnn_param: dict,
                sess: tf.Session,
                data_param: dict,
                data_loader: Dataset):
    with sess.graph.as_default():
        global_step = tf.train.get_or_create_global_step(graph=sess.graph)

        weight_regularizer = tf.contrib.layers.l2_regularizer(0.001)
        # weight_regularizer = None

        is_training = graph_type == GraphType.TRAIN
        dataset, num_class = data_loader.load_data(dataset_path, is_training, epoch_num, batch_size,
                                                   data_param=data_param)
        image, label = dataset

        model = cnn_model()
        logits = model.inference(image,
                                 num_class,
                                 param={**cnn_param,
                                        **{'global_steps': global_step, 'image_size': 100},
                                        'weight_regularizer': weight_regularizer,
                                        'graph_type': graph_type})

        if graph_type == GraphType.EVAL:
            saver = tf.train.Saver(max_to_keep=5)

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

        base_loss = softmax_loss(logits, label)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name="loss")
        tf.summary.scalar("loss", loss)

        # SGD training strategy
        # train_op = tf.train.AdamOptimizer(name='optimizer').minimize(loss, global_step=global_step)
        # base_lr = 1e-4
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
        # train_op = tf.train.MomentumOptimizer(momentum=0.9,
        #                                       name='optimizer',
        #                                       learning_rate=learning_rate).minimize(loss, global_step=global_step)
        learning_rate = tf.train.polynomial_decay(learning_rate=1e-4, global_step=global_step, decay_steps=500,
                                                  end_learning_rate=1e-5, name="learning_rate")
        learning_rate = 5e-5
        tf.summary.scalar("learning_rate", learning_rate)
        train_op = tf.train.AdamOptimizer(name='optimizer', learning_rate=learning_rate).minimize(loss,
                                                                                                  global_step=global_step)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=5)

        # initialize variables
        if os.path.isdir(log_dir):
            latest_ckpt = tf.train.latest_checkpoint(os.path.expanduser(log_dir))
        else:
            latest_ckpt = log_dir
        if latest_ckpt is not None:
            # restore model
            saver.restore(sess, latest_ckpt)
        else:
            sess.run(tf.global_variables_initializer())

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
                       eval_every_step=1000):
    is_final_eval = False
    while not is_final_eval:
        data_loader = datasets.webface.WebFace()

        training_step = 0
        tf.reset_default_graph()
        with tf.Session(config=sess_config) as sess:
            # build training graph
            train_op, train_acc, _, loss, global_step, summary_op, saver, acc_summary_op = build_graph(
                dataset_path=dataset_path,
                graph_type=GraphType.TRAIN,
                epoch_num=epoch_num,
                batch_size=batch_size,
                cnn_model=cnn_model,
                cnn_param=cnn_param,
                sess=sess,
                log_dir=logdir,
                data_param={'fail_path': args.fail_path,
                            'bounding_boxes': args.bounding_box_path,
                            'margin': args.margin},
                data_loader=data_loader)
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

                        if os.path.isfile(logdir):
                            model_save_path = os.path.dirname(logdir)
                        else:
                            model_save_path = logdir
                        saver.save(sess, os.path.join(os.path.expanduser(model_save_path), 'model.ckpt'),
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
    """Evaluate on LFW dataset"""
    tf.reset_default_graph()
    with tf.Session(config=sess_config) as sess:
        tf.logging.info("--------START EVALUATION--------")
        tf.logging.info("loading evaluation graph")

        dataset = LFW('/home/hyh/datasets/lfw')

        logits_op, file_name_op, step_op = build_graph(
            dataset_path=dataset_path,
            graph_type=GraphType.EVAL,
            epoch_num=1,
            batch_size=batch_size,
            cnn_model=cnn_model,
            cnn_param=cnn_param,
            sess=sess,
            log_dir=logdir,
            data_param={},
            data_loader=dataset)

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
                tf.logging.debug(f'Evaluating batch {batch_count}')
            except tf.errors.OutOfRangeError:
                break

        embeddings = {}
        for i in range(len(logits)):
            embeddings[str(file_name[i], encoding='utf8')] = logits[i]

        # excute 10-fold evaluation
        def get_threshold(val_pairs, similarity, embeddings, thrNum=10000):
            # numpy implementation for
            # https://github.com/wy1iu/sphereface/blob/master/test/code/evaluation.m#L115
            thresholds = np.arange(-thrNum, thrNum, 1) / thrNum
            acc = []
            for t in thresholds:
                acc.append(similarity(embeddings, val_pairs, t))
            return np.mean(thresholds[acc == np.max(acc)])

        def cos_similarity(emb: dict, eval_pairs, threshold):
            # numpy implementation for
            # https://github.com/wy1iu/sphereface/blob/master/test/code/evaluation.m#L69
            pairs = list(map(lambda p: [emb[p[0][0]], emb[p[1][0]]], eval_pairs))
            pairs = np.array(pairs)
            scores = np.sum(pairs[:, 0] * pairs[:, 1], axis=1)  # inner product for each row
            """
            TODO https://github.com/wy1iu/sphereface/blob/master/test/code/evaluation.m#L124
            confused on this implementation:
            
                function accuracy = getAccuracy(scores, flags, threshold)
                    accuracy = (length(find(scores(flags==1)>threshold)) + ...
                    length(find(scores(flags~=1)<threshold))) / length(scores);
                end
            
            since flags == 1 means same face:
            
                if length(strings) == 3
                    i = i + 1;
                    pairs(i).fileL = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{2}), '_%04i.jpg')]);
                    pairs(i).fileR = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{3}), '_%04i.jpg')]);
                    pairs(i).fold  = ceil(i / 600);
                    pairs(i).flag  = 1;
                elseif length(strings) == 4
                    i = i + 1;
                    pairs(i).fileL = fullfile(folder, strings{1}, [strings{1}, num2str(str2num(strings{2}), '_%04i.jpg')]);
                    pairs(i).fileR = fullfile(folder, strings{3}, [strings{3}, num2str(str2num(strings{4}), '_%04i.jpg')]);
                    pairs(i).fold  = ceil(i / 600);
                    pairs(i).flag  = -1;
                end
            
            why score>threshold is correct?
            it is right because consine is a decrease function
            """

            result = scores > threshold
            label = np.array(list(map(lambda p: p[2], eval_pairs)))
            return np.sum(result == label) / len(label)

            pass

        eval_acc = dataset.evaluation(embeddings, get_threshold, cos_similarity)

        # add to tf summary
        eval_writer = tf.summary.FileWriter(os.path.join(logdir, 'eval'), sess.graph)
        summary = tf.Summary()
        summary.value.add(tag='acc', simple_value=eval_acc)
        eval_writer.add_summary(summary, global_step=step)
        # eval_writer.add_summary(eval_acc, global_step=step)

        # make sure all summaries are written to disk
        eval_writer.flush()
        eval_writer.close()

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
                       epoch_num=None,
                       batch_size=args.batch_size,
                       cnn_model=cnn_model,
                       cnn_param={'softmax': args.softmax_type},
                       sess_config=config,
                       logdir=args.log_dir,
                       eval_every_step=1,
                       args=args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
