# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:13:00 2018

@author: Trịnh Quốc Trung
@email: trinhtrung96@gmail.com
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def create_config(alpha, ntopics, lr, beta1, beta2, temperature, layers, train_wv, burn_in):
    return {
        'ntopics': ntopics, 'lr': lr, 'beta1': beta1, 'alpha': alpha, 'temperature': temperature, 'beta2': beta2,
        'layers': layers, 'train_wv': train_wv, 'burn_in': burn_in
    }


class VAE(object):
    def __init__(self, wordvec, config, name="VAE-LDA-W2V"):
        self.ntopics = config['ntopics']
        self.nwords, self.dword = wordvec.shape
        self.build_inputs()
        with tf.variable_scope(name):
            self.build_model(wordvec, config['layers'], config['temperature'], config['train_wv'])
            self.build_loss(config['alpha'])
        self.build_train_op(config['lr'], config['beta1'], config['beta2'], config['burn_in'])

    def build_inputs(self):
        self.indices = tf.placeholder(dtype=tf.int32, shape=(None, None), name="indices")
        self.count = tf.placeholder(dtype=tf.int32, shape=(None, None), name="count")
        self.count_float = tf.to_float(self.count)
        self.expand_count = tf.expand_dims(self.count_float, axis=-1)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

    def build_dense(self, inputs, layers, name):
        outputs = inputs
        with tf.variable_scope(name):
            for layer in layers:
                outputs = tf.layers.dense(outputs, layer['units'], activation=None, kernel_initializer=tf.glorot_uniform_initializer())
                if layer.get('batch_norm', True):
                    outputs = tf.layers.batch_normalization(outputs, trainable=True, training=self.is_training)
                outputs = layer.get('activation', tf.nn.relu)(outputs)
            return outputs

    def build_model(self, np_wvec, layers, temperature, train_wv):
        self.input_shape = tf.shape(self.indices)
        self.wvec = tf.get_variable(trainable=train_wv, shape=np_wvec.shape, initializer=tf.constant_initializer(np.float32(np_wvec)), dtype=tf.float32, name="word_vector")
        self.float_lens = tf.reduce_sum(self.count_float, axis=1)
        with tf.variable_scope('topic_model'):
            topic_embedding = tf.get_variable(name='topic_embedding', shape=(self.dword, self.ntopics), initializer=tf.glorot_uniform_initializer())
            embedding = tf.nn.embedding_lookup(self.wvec, self.indices)
            doc_emb = tf.reduce_sum(embedding * self.expand_count, axis=1) / tf.expand_dims(self.float_lens, axis=-1)
            doc_emb = self.build_dense(doc_emb, layers + [{'units': self.ntopics}], 'doc_emb')
            embedding = tf.reshape(embedding, [-1, self.dword]) @ topic_embedding
            embedding = tf.reshape(embedding, [self.input_shape[0], self.input_shape[1], self.ntopics])
            self.w2t_diff = embedding + tf.expand_dims(doc_emb, axis=1)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.float_step = tf.to_float(self.global_step)
            self.temperature = tf.maximum(temperature['min'], tf.exp(-temperature['rate'] * self.float_step))
            self.temperature = tf.cond(self.is_training, lambda: self.temperature, lambda: 1e-3)
            self.gumbel_w2t = tfp.distributions.RelaxedOneHotCategorical(allow_nan_stats=False, temperature=self.temperature, logits=self.w2t_diff, name="gumbel_w2t")
            self.w2t_dist = self.gumbel_w2t.distribution.probs
            with tf.variable_scope("d2t_dirichlet"):
                self.sample_w2t = self.gumbel_w2t.sample() * self.expand_count  # [batch_size, time_step, ntopics]
                self.sample_sum_w2t = tf.reduce_sum(self.sample_w2t, axis=1, name="sample_sum_w2t")
                self.dirichlet_a = tf.get_variable(name='dirichlet_a', shape=(), trainable=True, initializer=tf.ones_initializer())
                self.dirichlet_b = tf.get_variable(name='dirichlet_b', shape=(), trainable=True, initializer=tf.zeros_initializer())
                self.concentration = tf.nn.softplus(self.sample_sum_w2t * self.dirichlet_a + self.dirichlet_b)
                self.d2t_dirichlet = tfp.distributions.Dirichlet(allow_nan_stats=False, concentration=self.concentration)
            self.sample_d2t = tf.maximum(self.d2t_dirichlet.sample(), np.finfo(np.float32).tiny, name='sample_d2t')
            self.beta = tf.get_variable(name="beta", shape=(self.nwords, self.ntopics), dtype=tf.float32, trainable=True, initializer=tf.ones_initializer())
            beta = tf.layers.batch_normalization(self.beta, trainable=True, training=self.is_training)
            beta_softmax = tf.nn.softmax(beta, axis=0, name='beta_softmax')
            self.beta_lookup = tf.nn.embedding_lookup(beta_softmax, self.indices)
            self.topic_word_dist = tf.transpose(self.beta, [1, 0], name='topic_word_dist')

    def build_loss(self, alpha):
        def _softplus_inverse(x):
            return np.log(np.expm1(x))
        with tf.variable_scope('topic_model_loss'):
            self.alpha = alpha = tf.get_variable(name="alpha", trainable=True, shape=[1, self.ntopics], initializer=tf.constant_initializer(_softplus_inverse(alpha)))
            alpha_dirichlet = tfp.distributions.Dirichlet(concentration=tf.clip_by_value(tf.nn.softplus(alpha), 1e-3, 1e3))
            kl_dirichlet = self.d2t_dirichlet.kl_divergence(alpha_dirichlet)
            with tf.control_dependencies([tf.assert_greater(kl_dirichlet, -1e-3, message="kl_dirichlet")]):
                self.kl_dirichlet = tf.identity(kl_dirichlet, name="kl_dirichlet")
            self.log_q_z = tf.reduce_sum(self.w2t_dist * tf.math.log(self.w2t_dist + 1e-20) * self.expand_count, axis=(1, 2), name="log_q_z")
            self.log_p_z = tf.reduce_sum(self.sample_sum_w2t * tf.math.log(self.sample_d2t), axis=1, name="log_p_z")
            self.log_p_w = tf.reduce_sum(self.sample_w2t * tf.log(self.beta_lookup + 1e-20), axis=(1, 2), name="log_p_w")
            self.bound = self.log_p_w + self.log_p_z - self.log_q_z - self.kl_dirichlet
            self.tm_loss = -tf.reduce_mean(self.bound)
        self.loss = self.tm_loss
        self.log_perplexity = -tf.reduce_sum(self.bound / self.float_lens)

    def build_train_op(self, lr, beta1, beta2, burn_in):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.lr = tf.minimum(lr['max'], (self.float_step + 1.0) * lr['rate'])
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=beta1, beta2=beta2)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads_and_vars_except_prior = [x for x in grads_and_vars if x[1] != self.alpha]

            def train_op_except_prior():
                return optimizer.apply_gradients(grads_and_vars_except_prior, global_step=self.global_step)

            def train_op_all():
                return optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            self.train_op = tf.cond(self.global_step < burn_in, true_fn=train_op_except_prior, false_fn=train_op_all)
