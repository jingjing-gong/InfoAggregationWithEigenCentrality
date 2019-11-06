'''
Created on Sep 21, 2016

@author: jerrik
'''

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import  layers

import utils, nest
from TfUtils import entry_stop_gradients, mkMask, reduce_avg, masked_softmax

NINF = -1e20
EPSILON = 1e-10

class model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, config):
        """options in this function"""
        self.config = config
        self.EX_REG_SCOPE = []

        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_accu = tf.assign_add(self.on_epoch, 1)

        self.build()

    def add_placeholders(self):
        # shape(b_sz, sNum, wNum)
        self.ph_input = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='ph_input')

        # shape(bsz)
        self.ph_labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_labels')

        # [b_sz]
        self.ph_sNum = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_sNum')

        # shape(b_sz, sNum)
        self.ph_wNum = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ph_wNum')

        self.ph_sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='ph_sample_weights')
        self.ph_train = tf.placeholder(dtype=tf.bool, name='ph_train')

    def create_feed_dict(self, data_batch, train):
        '''data_batch:  label_ids, snt1_matrix, snt2_matrix, snt1_len, snt2_len'''

        phs = (self.ph_input, self.ph_labels, self.ph_sNum, self.ph_wNum, self.ph_sample_weights, self.ph_train)
        feed_dict = dict(zip(phs, data_batch+(train,)))
        return feed_dict

    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        inputs: a list of tensors each of which have a size of [batch_size, embed_size]
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        vocab_sz = max(self.config.vocab_dict.values())
        with tf.variable_scope('embedding') as scp:
            self.exclude_reg_scope(scp)
            if self.config.pre_trained:
                embed = utils.readEmbedding(self.config.embed_path)
                embed_matrix, valid_mask = utils.mkEmbedMatrix(embed, dict(self.config.vocab_dict))
                embedding = tf.Variable(embed_matrix, 'Embedding')
                partial_update_embedding = entry_stop_gradients(embedding, tf.expand_dims(valid_mask, 1))
                embedding = tf.cond(self.on_epoch < self.config.partial_update_until_epoch,
                                    lambda: partial_update_embedding, lambda: embedding)
            else:
                embedding = tf.get_variable(
                  'Embedding',
                  [vocab_sz, self.config.embed_size], trainable=True)
        return embedding

    def embed_lookup(self, embedding, batch_x, dropout=None, is_train=False):
        '''

        :param embedding: shape(v_sz, emb_sz)
        :param batch_x: shape(b_sz, sNum, wNum)
        :return: shape(b_sz, sNum, wNum, emb_sz)
        '''
        inputs = tf.nn.embedding_lookup(embedding, batch_x)
        if dropout is not None:
            inputs = tf.layers.dropout(inputs, rate=dropout, training=is_train)
        return inputs

    def hierachical_attention(self, in_x, sNum, wNum, scope=None):
        '''

        :param in_x: shape(b_sz, ststp, wtstp, emb_sz)
        :param sNum: shape(b_sz, )
        :param wNum: shape(b_sz, ststp)
        :param scope:
        :return:
        '''
        b_sz, ststp, wtstp, _ = tf.unstack(tf.shape(in_x))
        emb_sz = int(in_x.get_shape()[-1])
        with tf.variable_scope(scope or 'hierachical_attention'):
            flatten_in_x = tf.reshape(in_x, [b_sz*ststp, wtstp, emb_sz])
            flatten_wNum = tf.reshape(wNum, [b_sz * ststp])

            with tf.variable_scope('sentence_enc'):
                if self.config.seq_encoder == 'bigru':
                    flatten_birnn_x = self.biGRU(flatten_in_x, flatten_wNum,
                                                 self.config.hidden_size, scope='biGRU')
                elif self.config.seq_encoder == 'bilstm':
                    flatten_birnn_x = self.biLSTM(flatten_in_x, flatten_wNum,
                                                 self.config.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encoder %s'%self.config.seq_encoder)

                '''shape(b_sz*sNum, dim)'''
                if self.config.attn_mode == 'avg':
                    flatten_attn_ctx = reduce_avg(flatten_birnn_x, flatten_wNum, dim=1)
                elif self.config.attn_mode == 'max':
                    flatten_attn_ctx = tf.reduce_max(flatten_birnn_x, axis=1)
                elif self.config.attn_mode == 'attn':
                    flatten_attn_ctx = self.task_specific_attention(flatten_birnn_x, flatten_wNum,
                                                            int(flatten_birnn_x.get_shape()[-1]),
                                                            dropout=self.config.dropout,
                                                            is_train=self.ph_train, scope='attention')
                elif self.config.attn_mode == 'eigen':
                    flatten_attn_ctx, time, stop = self.basic_Centrality(flatten_birnn_x, flatten_wNum,
                                                                         config=self.config.vbs_config,
                                                                         is_train=self.ph_train,
                                                                         scope='basic-Centrality')
                else:
                    raise ValueError('no such attn mode %s' % self.config.attn_mode)
            snt_dim = int(flatten_attn_ctx.get_shape()[-1])
            snt_reps = tf.reshape(flatten_attn_ctx, shape=[b_sz, ststp, snt_dim])

            with tf.variable_scope('doc_enc'):
                if self.config.seq_encoder == 'bigru':
                    birnn_snt = self.biGRU(snt_reps, sNum, self.config.hidden_size, scope='biGRU')
                elif self.config.seq_encoder == 'bilstm':
                    birnn_snt = self.biLSTM(snt_reps, sNum, self.config.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encoder %s'%self.config.seq_encoder)

                '''shape(b_sz, dim)'''
                if self.config.attn_mode == 'avg':
                    doc_rep = reduce_avg(birnn_snt, sNum, dim=1)
                elif self.config.attn_mode == 'max':
                    doc_rep = tf.reduce_max(birnn_snt, axis=1)
                elif self.config.attn_mode == 'attn':
                    doc_rep = self.task_specific_attention(birnn_snt, sNum,
                                                           int(birnn_snt.get_shape()[-1]),
                                                           dropout=self.config.dropout,
                                                           is_train=self.ph_train, scope='attention')
                elif self.config.attn_mode == 'eigen':
                    doc_rep, time, stop = self.basic_Centrality(birnn_snt, sNum,
                                                               config=self.config.vbs_config,
                                                               is_train=self.ph_train, scope='basic_Centrality')

                else:
                    raise ValueError('no such attn mode %s' % self.config.attn_mode)
        return doc_rep

    def build(self):
        self.add_placeholders()
        self.embedding = self.add_embedding()
        '''shape(b_sz, ststp, wtstp, emb_sz)'''
        in_x = self.embed_lookup(self.embedding, self.ph_input,
                                 dropout=self.config.dropout, is_train=self.ph_train)
        doc_reps = self.hierachical_attention(in_x, self.ph_sNum, self.ph_wNum, scope='hierachical_attn')

        with tf.variable_scope('classifier'):
            logits = self.Dense(doc_reps, dropout=self.config.dropout,
                                is_train=self.ph_train, activation=tf.nn.tanh)
            opt_loss = self.add_loss_op(logits, self.ph_labels)
            train_op = self.add_train_op(opt_loss)
        self.train_op = train_op
        self.opt_loss = opt_loss
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('opt_loss', self.opt_loss)
        tf.summary.scalar('w_loss', self.w_loss)

    def Dense(self, inputs, dropout=None, is_train=False, activation=None):
        loop_input = inputs
        if self.config.dense_hidden[-1] != self.config.class_num:
            raise ValueError('last hidden layer should be %d, but get %d' %
                             (self.config.class_num,
                              self.config.dense_hidden[-1]))
        for i, hid_num in enumerate(self.config.dense_hidden):
            with tf.variable_scope('dense-layer-%d' % i):
                loop_input = tf.layers.dense(loop_input, units=hid_num)

            if i < len(self.config.dense_hidden) - 1:
                if dropout is not None:
                    loop_input = tf.layers.dropout(loop_input, rate=dropout, training=is_train)
                loop_input = activation(loop_input)

        logits = loop_input
        return logits

    def add_loss_op(self, logits, labels):
        '''

        :param logits: shape(b_sz, c_num) type(float)
        :param labels: shape(b_sz,) type(int)
        :return:
        '''

        self.prediction = tf.argmax(logits, axis=-1, output_type=labels.dtype)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, labels), tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        ce_loss = tf.reduce_mean(loss)

        exclude_vars = nest.flatten([[v for v in tf.trainable_variables(o.name)] for o in self.EX_REG_SCOPE])
        exclude_vars_2 = [v for v in tf.trainable_variables() if '/bias:' in v.name]
        exclude_vars = exclude_vars + exclude_vars_2

        reg_var_list = [v for v in tf.trainable_variables() if v not in exclude_vars]
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
        self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in reg_var_list])

        print('===' * 20)
        print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
        print('excluded variables from regularization')
        print([v.name for v in exclude_vars])
        print('===' * 20)

        print('regularized variables')
        print(['%s:%.3fM' % (v.name, np.prod(v.get_shape().as_list()) / 1000000.) for v in reg_var_list])
        print('===' * 20)
        '''shape(b_sz,)'''
        self.ce_loss = ce_loss
        self.w_loss = tf.reduce_mean(tf.multiply(loss, self.ph_sample_weights))
        reg = self.config.reg

        return self.ce_loss + reg * reg_loss

    def add_train_op(self, loss):

        lr = tf.train.exponential_decay(self.config.lr, self.global_step,
                                        self.config.decay_steps,
                                        self.config.decay_rate, staircase=True)
        self.learning_rate = tf.maximum(lr, 1e-5)
        if self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config.optimizer == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise ValueError('No such Optimizer: %s' % self.config.optimizer)

        gvs = optimizer.compute_gradients(loss=loss)

        capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        return train_op

    def exclude_reg_scope(self, scope):
        if scope not in self.EX_REG_SCOPE:
            self.EX_REG_SCOPE.append(scope)

    @staticmethod
    def biLSTM(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

        with tf.variable_scope(scope or 'biLSTM'):
            cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')

            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    @staticmethod
    def biGRU(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

        with tf.variable_scope(scope or 'biGRU'):
            cell_fwd = tf.nn.rnn_cell.GRUCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.GRUCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')

            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    @staticmethod
    def task_specific_attention(in_x, xLen, out_sz, activation_fn=tf.tanh,
                                dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param activation_fn: activation
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''

        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None

        with tf.variable_scope(scope or 'attention') as scope:
            context_vector = tf.get_variable(name='context_vector', shape=[out_sz],
                                             dtype=tf.float32)
            in_x_mlp = tf.layers.dense(in_x, out_sz, activation=activation_fn, name='mlp')

            attn = tf.tensordot(in_x_mlp, context_vector, axes=[[2], [0]])  # shape(b_sz, tstp)
            attn_normed = masked_softmax(attn, xLen)

            attn_normed = tf.expand_dims(attn_normed, axis=-1)
            attn_ctx = tf.matmul(in_x_mlp, attn_normed, transpose_a=True)  # shape(b_sz, dim, 1)
            attn_ctx = tf.squeeze(attn_ctx, axis=[2])   # shape(b_sz, dim)
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    @staticmethod
    def basic_Centrality(in_x, xLen, config, is_train, scope=None):
        '''

        :param in_x:    shape(b_sz, xlen, h_sz)
        :param xLen:
        :return:
        '''

        def PowerEigen(matrix, max_iter, eta):
            '''

            :param matrix: shape(b_sz, dim, dim)
            :return out: shape(b_sz, dim)
            '''
            b_sz = tf.shape(matrix)[0]
            dim = tf.shape(matrix)[1]

            def body(time, _, y):
                '''

                :param y: shape(b_sz, dim, 1)
                :return out:     shape(b_sz, dim, 1)
                '''
                v = y / (tf.sqrt(tf.reduce_sum(y ** 2, axis=1, keep_dims=True))+EPSILON)
                y = tf.matmul(matrix, v)                                            # shape(b_sz, dim, 1)
                theta = tf.matmul(v, y, transpose_a=True)                 # shape(b_sz, 1, 1)
                stop = tf.less(tf.reduce_sum((y - theta*v)**2, axis=1),
                               eta*tf.squeeze(theta, axis=1)**2)  # shape(b_sz, 1)
                stop = tf.squeeze(stop, axis=1)     # shape(b_sz)
                acc = tf.cast(tf.logical_not(stop), dtype=time.dtype)

                assert_inf = tf.Assert(tf.reduce_all(tf.is_finite(y)),
                                       ['y inf print v', v], summarize=100000)
                assert_nan = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(y))),
                                       ['y Nan print v', v], summarize=100000)
                with tf.control_dependencies([assert_inf, assert_nan]):
                    y = tf.identity(y)
                return time + acc, stop, y,

            def stop_cond(time, stop, _):
                '''

                :param time: shape(b_sz) int32
                :param stop: shape(b_sz,) bool
                :param _:
                :return:
                '''
                return tf.logical_and(tf.reduce_all(tf.less(time, max_iter)), tf.reduce_any(tf.logical_not(stop)))

            zero_step = tf.zeros([b_sz], dtype=tf.int32)
            stop = tf.zeros(shape=[b_sz], dtype=tf.bool)    # shape(b_sz,)
            init_eig = tf.random_normal([b_sz, dim, 1], dtype=matrix.dtype)**2 + EPSILON
            # init_eig = tf.ones([b_sz, dim, 1], dtype=matrix.dtype)

            time, stop, y = tf.while_loop(stop_cond, body=body,
                                          loop_vars=[zero_step, stop, init_eig],
                                          back_prop=False, swap_memory=True)              # shape(b_sz, dim, 1)
            v = y / (tf.sqrt(tf.reduce_sum(y ** 2, axis=1, keep_dims=True))+EPSILON)

            eigenValue = tf.reduce_mean(tf.matmul(matrix, v) / v, axis=1)   # shape(b_sz, 1)

            tf.summary.histogram("converge-time", time)
            return time, stop, eigenValue, v    # shape(b_sz, dim, 1)

        def PowerEigen_with_step(matrix, max_iter, eigenVector):
            '''

            :param matrix: shape(b_sz, dim, dim)
            :return out: shape(b_sz, dim)
            '''
            b_sz = tf.shape(matrix)[0]
            dim = tf.shape(matrix)[1]

            def body(time, y):
                '''

                :param y: shape(b_sz, dim, 1)
                :return out:     shape(b_sz, dim, 1)
                '''
                v = y / (tf.sqrt(tf.reduce_sum(y ** 2, axis=1, keep_dims=True))+EPSILON)
                y = tf.matmul(matrix, v)                                            # shape(b_sz, dim, 1)
                return time + 1, y

            def stop_cond(time, _):
                '''

                :param time: shape(b_sz) int32
                :param stop: shape(b_sz,) bool
                :param _:
                :return:
                '''
                return tf.less(time, max_iter)

            zero_step = tf.constant(0, dtype=tf.int32)

            time, y = tf.while_loop(stop_cond, body=body,
                                          loop_vars=[zero_step, eigenVector],
                                          swap_memory=True)              # shape(b_sz, dim, 1)
            v = y / (tf.sqrt(tf.reduce_sum(y ** 2, axis=1, keep_dims=True))+EPSILON)
            eigenValue = tf.reduce_mean(tf.matmul(matrix, v) / v, axis=1)   # shape(b_sz, 1)

            return time, eigenValue, tf.transpose(v, perm=[0, 2, 1])    # (shape(b_sz, 1), shape(b_sz, 1, dim))

        def connect_fnn(in_x):
            '''

            :param in_x: shape(b_sz, xLen, h_sz)
            :return:
            '''
            h_sz = int(in_x.get_shape()[-1])
            units = config['units']
            left = tf.layers.dense(in_x, units=units, name='left-fnn')   #shape(b_sz, xLen, h_sz)
            right = tf.layers.dense(in_x, units=units, name='right-fnn') # shape(b_sz, xLen, h_sz)
            fnn_concat_mlp = tf.nn.tanh(tf.expand_dims(left, axis=1) + tf.expand_dims(right, axis=2))
            fnn_concat_mlp = tf.layers.dense(fnn_concat_mlp, units=1, name='concat-mlp')
            connectivity = fnn_concat_mlp
            connectivity = tf.squeeze(connectivity, axis=-1)
            return connectivity

        def connect_bilinear(_in_x):
            '''

            :param _in_x: shape(b_sz, xLen, h_sz)
            :return:
            '''
            units = config['units']
            left = tf.layers.dense(in_x, units=units, name='left-fnn')  # shape(b_sz, xLen, h_sz)
            right = tf.layers.dense(in_x, units=units, name='right-fnn')  # shape(b_sz, xLen, h_sz)
            connect = tf.matmul(left, right, transpose_b=True)          # shape(b_sz, xLen, xLen)

            return connect


        h_sz = int(in_x.get_shape()[-1])
        maxLen = tf.shape(in_x)[1]
        b_sz = tf.shape(in_x)[0]
        mask = tf.expand_dims(mkMask(xLen, maxLen), axis=2)     # shape(b_sz, xlen, 1)
        mask = tf.logical_and(mask, tf.transpose(mask, perm=[0, 2, 1]))         # shape(b_sz, xlen, xlen)

        with tf.variable_scope(scope or 'Centrality'):
            if config['center-mode'] == 'center':
                in_x_cent = in_x - tf.reduce_mean(in_x, axis=2, keepdims=True)
            elif config['center-mode'] == 'ln':
                in_x_cent = layers.layer_norm(in_x, begin_norm_axis=-1)
            elif config['center-mode'] == 'none':
                in_x_cent = in_x
            else:
                raise ValueError('center-mode: %s' % config['center-mode'])

            if config['connect'] == 'fnn':
                connectivity = connect_fnn(in_x_cent)
            elif config['connect'] == 'bilinear':
                connectivity = connect_bilinear(in_x_cent)
            else:
                raise ValueError('No such connect: %s'%config['connect'])

            if not config['self-edge']:
                I = tf.eye(maxLen, batch_shape=[b_sz], dtype=tf.bool)
                connectivity = tf.where(I, tf.ones_like(connectivity) * NINF, connectivity)

            masked_connected = tf.where(mask, connectivity, tf.ones_like(connectivity) * NINF)

            if config['soft'] == 'all':
                assert_inf_in = tf.Assert(tf.reduce_all(tf.is_finite(masked_connected)),
                                       ['assert inf in masked_in', masked_connected],
                                        summarize=1000)
                assert_nan_in = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(masked_connected))),
                                       ['assert nan in masked_in', masked_connected],
                                       summarize=1000)
                r_shaped = tf.reshape(masked_connected, shape=[b_sz, -1])
                r_shaped = tf.nn.softmax(r_shaped, axis=1)
                masked_connected = tf.reshape(r_shaped, shape=tf.shape(masked_connected))

                assert_inf_out = tf.Assert(tf.reduce_all(tf.is_finite(masked_connected)),
                                       ['assert inf in masked_out', masked_connected],
                                       summarize=1000)
                assert_nan_out = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(masked_connected))),
                                       ['assert nan in masked_out', masked_connected],
                                       summarize=1000)
                with tf.control_dependencies([assert_inf_in, assert_nan_in, assert_inf_out, assert_nan_out]):
                    masked_connected = tf.identity(masked_connected)
            elif config['soft'] == 'col':
                masked_connected = tf.nn.softmax(masked_connected, axis=1)
            elif config['soft'] == 'none':
                masked_connected = masked_connected
            else:
                raise ValueError('soft: %s' % config['soft'])

            masked_connected = tf.where(mask, masked_connected, tf.zeros_like(connectivity))

            time, stop, _, eigenVec = PowerEigen(masked_connected,
                                                 tf.cond(is_train, lambda: config['max-iter'], lambda: 5000),
                                                 eta=config['power-eta'])

            eigenVec = tf.stop_gradient(eigenVec)

            # eigenVec = tf.transpose(eigenVec, perm=[0, 2, 1])
            _, eigenValue, eigenVec = PowerEigen_with_step(masked_connected,
                                                           max_iter=config['grad-iter'],
                                                           eigenVector=eigenVec)

            attn_weights = tf.abs(eigenVec) / (tf.reduce_sum(tf.abs(eigenVec), axis=-1, keep_dims=True)+EPSILON)
            aggregated = tf.matmul(attn_weights, in_x_cent)                 # shape(b_sz, 1, h_sz)
            aggregated = tf.squeeze(aggregated, axis=1)
            aggregated = layers.layer_norm(aggregated, begin_norm_axis=-1)

            return aggregated, time, stop
