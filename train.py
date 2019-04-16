import argparse
import json
import logging
import os
import pickle
import time
from functools import partial as partial_func

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from sklearn import metrics
from sklearn.metrics import (adjusted_rand_score,
                             homogeneity_completeness_v_measure)
from sklearn.metrics.cluster import contingency_matrix

from vae import VAE, create_config

np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_path', type=str)
parser.add_argument('--ntopics', type=int, default=4)
parser.add_argument('--layers', type=str, default='20,20')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=str)
parser.add_argument('--temp', type=str)
parser.add_argument('--train_data', type=str)
parser.add_argument('--test_data', type=str, default='')
parser.add_argument('--label_train', type=str, default='')
parser.add_argument('--label_test', type=str, default='')
parser.add_argument('--word2idx', type=str)
parser.add_argument('--word_vec', type=str)
parser.add_argument('--train_wv', action='store_true')
parser.add_argument('--wv_size', type=int, default=-1)
parser.add_argument('--alpha', type=float, default=-1.0)
parser.add_argument('--burn_in', type=int)
parser.add_argument('--accumulate_metrics', type=str, default='runs/results.csv')
parser.add_argument('--experiment_name', type=str, default='some_experiment')
args = parser.parse_args()
alpha = args.alpha
if alpha < 0 or alpha > 1:
    alpha = 1 / args.ntopics


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


logger = get_logger(os.path.join(args.save_path, 'log'))
logger.info(args)
with open(args.word2idx, 'r') as inp:
    word2idx = json.load(inp)
vocab = {i: w for w, i in word2idx.items()}
vocab[0] = '<PAD>'
if args.wv_size <= 0:
    wv_matrix = np.load(args.word_vec)
else:
    logger.info('Random initial word vector.')
    wv_matrix = np.random.uniform(low=-1.0, high=1.0, size=(len(word2idx)+1, args.wv_size))
train = np.load(args.train_data)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # pylint: disable=no-member
session = tf.Session(config=config)
ntopics = args.ntopics
lr = [float(x) for x in args.lr.split(",")]
temp = [float(x) for x in args.temp.split(",")]
layers = [{'units': int(x)} for x in args.layers.split(',')]
lr = {'max': lr[0], 'rate': lr[0]/lr[1]}
temp = {'min': temp[0], 'rate': -np.log(temp[0])/temp[1]}
model_config = create_config(alpha, ntopics, lr, args.beta1, args.beta2, temp, layers, args.train_wv, args.burn_in)
model = VAE(wordvec=wv_matrix, config=model_config)
session.run(tf.global_variables_initializer())
BATCH_SIZE = args.batch_size

tf.summary.scalar('log_q_z', tf.reduce_mean(model.log_q_z))
tf.summary.scalar('log_p_z', tf.reduce_mean(model.log_p_z))
tf.summary.scalar('log_p_w', tf.reduce_mean(model.log_p_w))
tf.summary.scalar('kl_dirichlet', tf.reduce_mean(model.kl_dirichlet))
tf.summary.scalar('avg_elbo', model.tm_loss)
tf.summary.scalar('log_perplexity', model.log_perplexity)
tf.summary.scalar('dirichlet_a', model.dirichlet_a)
tf.summary.scalar('dirichlet_b', model.dirichlet_b)
tf.summary.scalar('temperature', model.temperature)
tf.summary.scalar('learning_rate', model.lr)
tf.summary.histogram('alpha', model.alpha)
tf.summary.histogram('concentration', model.concentration)
merged = tf.summary.merge_all()


def get_topic_words(beta, idx2word):
    res = []
    for idx, topic in enumerate(beta):
        s = {'topic': idx}
        a = np.argsort(-topic)[:30]
        s = {'words': {idx2word[i]: float(topic[i]) for i in a}}
        res.append(s)
    return res


writer = tf.summary.FileWriter(os.path.join(args.save_path, 'summary'), session.graph)
with open(os.path.join(args.save_path, 'config.json'), 'w') as out:
    json.dump(model_config, out)


def batch_iter(data, bs, train=True):
    # data is list of documents, each document is a list of (word_id, word_count)
    if train:
        indices = np.random.permutation(np.arange(data.shape[0]))
        data = data[indices]
    for i in range(0, data.shape[0], bs):
        batch = data[i:i+bs]
        seq_lens = max(len(x) for x in batch)
        indices = np.zeros((len(batch), seq_lens), dtype=np.int32)
        counts = np.zeros((len(batch), seq_lens), dtype=np.int32)
        for ind, cnt, doc in zip(indices, counts, batch):
            ind[:len(doc)] = [x for x, _ in doc]
            cnt[:len(doc)] = [x for _, x in doc]
        yield indices, counts


EPOCH = args.epoch
t0 = time.time()
word_progress = {}
for i in range(EPOCH):
    ite = batch_iter(train, BATCH_SIZE)
    logger.info('Epoch ' + str(i))
    total_ppl = 0
    for ind, cnt in tqdm.tqdm(ite, total=(train.shape[0]-1)//BATCH_SIZE + 1):
        _, merged_sum, itera, loss, ppl, alpha = session.run([model.train_op, merged, model.global_step, model.tm_loss, model.log_perplexity, model.alpha],
                                                             feed_dict={model.indices: ind, model.count: cnt, model.is_training: True})
        total_ppl += ppl
        writer.add_summary(merged_sum, itera)
    logger.info('Perplexity: {:05.5f}'.format(np.exp(total_ppl / train.shape[0])))
    t2w = get_topic_words(session.run(model.topic_word_dist), vocab)
    word_progress[str(session.run(model.global_step))] = t2w
with open(os.path.join(args.save_path, 'topic2words.json'), 'w') as out:
    json.dump(word_progress, out)
runtime = time.time() - t0
logger.info('Training time: {}'.format(runtime))
if args.test_data != '' and args.label_test != '':
    test = np.load(args.test_data)
    result = []
    ppl = 0
    ite = batch_iter(test, BATCH_SIZE, train=False)
    for ind, cnt in tqdm.tqdm(ite, total=(test.shape[0]-1)//BATCH_SIZE + 1):
        theta, p = session.run([model.sample_d2t, model.log_perplexity], feed_dict={model.is_training: False, model.indices: ind, model.count: cnt})
        result.append(theta)
        ppl += p
    logger.info(np.exp(ppl/test.shape[0]))

    result = np.concatenate(result, axis=0)

    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    with open(args.label_test, 'r') as inp:
        label_true = [x.strip() for x in inp.readlines()]
    label_pred = np.argmax(result, axis=1)
    hcv = homogeneity_completeness_v_measure(labels_true=label_true, labels_pred=label_pred)
    ars = adjusted_rand_score(labels_true=label_true, labels_pred=label_pred)
    ps = purity_score(y_true=label_true, y_pred=label_pred)
    logger.info('homogeneity_completeness_v_measure: {}'.format(hcv))
    logger.info('adjusted_rand_score: {}'.format(ars))
    logger.info('purity_score: {}'.format(ps))
    logger.info('Contigency matrix:')
    logger.info(contingency_matrix(labels_true=label_true, labels_pred=label_pred))
    with open(args.accumulate_metrics, 'a') as out:
        out.write('{},{},{},{},{},{},{}\n'.format(args.experiment_name, hcv[0], hcv[1], hcv[2], ars, ps, runtime))

saver = tf.train.Saver()
saver.save(session, os.path.join(args.save_path, 'model', 'model.cpkt'), global_step=itera)
