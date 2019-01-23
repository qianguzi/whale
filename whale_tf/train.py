import os
import time
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import model
import data_generator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=np.float32)
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def train_model():
  tf.logging.set_verbosity(tf.logging.INFO)
  with open('../annex/w2ts.pickle', 'rb') as f:
        w2ts = pickle.load(f)
  train_id = np.load('../annex/train_id.npy')
  inputs_a = tf.placeholder(tf.float32, [None, 384, 384, 1], name='input_a')
  inputs_b = tf.placeholder(tf.float32, [None, 384, 384, 1], name='input_b')
  inputs_c = tf.placeholder(tf.float32, [None, 512], name='input_c')
  inputs_d = tf.placeholder(tf.float32, [None, 512], name='input_d')
  labels = tf.placeholder(tf.float32, [None, 1], name='labels')
  with slim.arg_scope(model.train_arg_scope(weight_decay=0.0001)): # pylint: disable=E1129
    outputs_a = model.branch_model(inputs_a, scope='Siamese_branch')
    outputs_b = model.branch_model(inputs_b, reuse=True, scope='Siamese_branch')
  outputs = model.head_model(outputs_a, outputs_b)
  outputs_head = model.head_model(inputs_c, inputs_d, reuse=True)
  cls_loss = slim.losses.sigmoid_cross_entropy(outputs, labels)
  predictions = tf.where(outputs >= 0.5, tf.ones_like(outputs), tf.zeros_like(outputs))
  acc = tf.metrics.accuracy(labels, predictions)
  # Gather update_ops
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # Gather initial summaries.
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
  summaries.add(tf.summary.scalar('losses/accuracy', acc))

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(64e-5, global_step, 10000, 0.98, staircase=True)
  opt = tf.train.AdamOptimizer(learning_rate)
  # opt = tf.train.RMSPropOptimizer(learning_rate, momentum=FLAGS.momentum)
  summaries.add(tf.summary.scalar('learning_rate', learning_rate))

  summaries.add(tf.summary.scalar('losses/cls_loss', cls_loss))
  regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  regularization_loss = tf.add_n(regularization_loss, name='regularization_loss')
  summaries.add(tf.summary.scalar('losses/regularization_loss', regularization_loss))

  total_loss = tf.add(cls_loss, regularization_loss, name='total_loss')
  grads_and_vars = opt.compute_gradients(total_loss)

  total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
  summaries.add(tf.summary.scalar('losses/total_loss', total_loss))

  grad_updates = opt.apply_gradients(grads_and_vars, global_step=global_step)
  update_ops.append(grad_updates)
  update_op = tf.group(*update_ops, name='update_barrier')
  with tf.control_dependencies([update_op]):
    train_tensor = tf.identity(total_loss, name='train_op')

  # Merge all summaries together.
  summary_op = tf.summary.merge(list(summaries))
  saver = tf.train.Saver(max_to_keep=15)
  best_saver = tf.train.Saver(max_to_keep=1)
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True # pylint: disable=E1101
  with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    # summary writer
    writer = tf.summary.FileWriter('../.logs', sess.graph)
    def make_steps(initial_epoch, epochs, ampl):
      random.shuffle(train_id)
      data_eval = data_generator.FeatureGen(train_id, batch_size=64, verbose=1)
      features = []
      for batch_id in range(len(data_eval)):
        feature = sess.run(outputs_a, feed_dict={inputs_a: data_eval[batch_id]})
        features.append(feature)
      features = np.concatenate(features, 0)
      data_eval = data_generator.ScoreGen(features, batch_size=2048, verbose=1)
      scores = []
      for batch_id in range(len(data_eval)):
        score = sess.run(outputs_head, feed_dict={inputs_c: data_eval[0], inputs_d: data_eval[1]})
        scores.append(score)
      scores = np.concatenate(scores, 0)
      scores = score_reshape(scores, features)
      data_train = data_generator.TrainingData(
                       train_id, w2ts,
                       scores + ampl * np.random.random_sample(size=scores.shape), 
                       steps=epochs, batch_size=64)
      min_loss = 10
      for epoch_id in range(initial_epoch, initial_epoch + epochs):
        print('epoch[%02d]'%(epoch_id + 1))
        for batch_id in range(len(data_train)):
          feed_dict = {inputs_a: data_train[batch_id][0][0],
                       inputs_b: data_train[batch_id][0][1]}
          _, _total_loss, _cls_loss, _summ, _global_step, _acc = sess.run(
              [train_tensor, total_loss, cls_loss, summary_op, global_step, acc], feed_dict)
          if min_loss > _cls_loss:
            min_loss = _cls_loss
            best_saver.save(sess, '../.checkpoints/best/model_best', _global_step)
            tf.train.write_graph(sess.graph_def, '../.checkpoints', 'model_bestpb.pb')
          if batch_id % 50 == 0:
            writer.add_summary(_summ, batch_id)
            print(time.strftime("%X"),
                  'batch[{0}] >> global_step:{1}, acc:{2:.6f}, total_loss:{3:.6f}, cls_loss:{4:.6f}'
                      .format(batch_id + 1, _global_step, _acc, _total_loss, _cls_loss))
        save_path = saver.save(sess, '../.checkpoints/model', epoch_id)
        print('Current model saved in ' + save_path)
        data_train.on_epoch_end()
      return initial_epoch + epochs

    current_epoch = make_steps(0, 10, 1000)
    for _ in range(2):
      current_epoch = make_steps(current_epoch, 5, 100)
    for _ in range(18):
      current_epoch = make_steps(current_epoch, 5, 1.0)
    for _ in range(10):
      current_epoch = make_steps(current_epoch, 5, 0.5)
    for _ in range(8):
      current_epoch = make_steps(current_epoch, 5, 0.25)


if __name__ == '__main__':
  train_model()
  
