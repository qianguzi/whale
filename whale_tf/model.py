import tensorflow as tf
from tensorflow.contrib import slim


def _botterneck(inputs, depth, scope=None):
  with tf.variable_scope(scope, 'botterneck', [inputs]):
    net = slim.conv2d(inputs, depth, (1, 1), scope='conv1')
    net = slim.conv2d(net, depth, (3, 3), scope='conv2')
    net = slim.conv2d(net, inputs.shape[-1], (1, 1), scope='conv3')
    net = tf.nn.relu(inputs + net, name='shortcut')
    return net
    

def _sub_block(inputs, repetitions, depth, num_outputs=256, apply_maxpool=True, scope=None):
  with tf.variable_scope(scope, 'block', [inputs]):
    net = slim.repeat(inputs, repetitions, _botterneck, depth)
    if not apply_maxpool:
      return net
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, num_outputs, (1, 1), scope='conv2d_1x1')
    return net


def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size = tf.convert_to_tensor(
        [1, tf.shape(input_tensor)[1],
         tf.shape(input_tensor)[2], 1])
  else:
    kernel_size = [1, shape[1], shape[2], 1]
  output = pool_op(
      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
  # Recover output shape, for unknown shape.
  output.set_shape([None, 1, 1, None])
  return output


def branch_model(inputs, reuse=None, scope=None):
  with tf.variable_scope(scope, 'Siamese_branch', reuse=reuse):
    inputs = tf.identity(inputs, 'input')
    net = slim.conv2d(inputs, 64, (9, 9), stride=2, scope='conv2d_9x9')
    net = slim.max_pool2d(net, (2, 2))
    net = slim.repeat(net, 2, slim.conv2d, 64, (3, 3), scope='conv2d_3x3')
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 128, (1, 1), scope='conv2d_1x1')
    net = _sub_block(net, 4, 64, 256)
    net = _sub_block(net, 4, 64, 384)
    net = _sub_block(net, 4, 96, 512)
    net = _sub_block(net, 4, 128, apply_maxpool=False)
    outputs = global_pool(net, pool_op=tf.nn.max_pool)
    outputs = tf.squeeze(outputs, axis=[1, 2], name='output')
    return outputs


def head_model(inputs_a, inputs_b, reuse=None, scope=None):
  with tf.variable_scope(scope, 'Siamese_head', reuse=reuse):
    inputs_a = tf.identity(inputs_a, 'input_a')
    inputs_b = tf.identity(inputs_b, 'input_b')
    inputs_a = tf.expand_dims(inputs_a, 1)
    inputs_b = tf.expand_dims(inputs_b, 1)
    rela_1 = tf.multiply(inputs_a, inputs_b, name='relation_1')
    rela_2 = tf.add(inputs_a, inputs_b, name='relation_2')
    rela_3 = tf.abs(rela_1 - rela_2, name='relation_3')
    rela_4 = tf.square(rela_3, name='relation_4')
    rela = tf.concat([rela_1, rela_2, rela_3, rela_4], axis=1, name='relation') # (B, 4, 512)
    net = tf.expand_dims(rela, -1) # (B, 4, 512, 1)
    net = slim.conv2d(net, 32, (4, 1), padding='VALID', scope='conv1') # (B, 1, 512, 32)
    net = tf.transpose(net, [0, 2, 3, 1]) # (B, 512, 32, 1)
    net = slim.conv2d(net, 1, (1, 32), activation_fn=None, padding='VALID', scope='conv2') # (B, 512, 1, 1)
    net = tf.transpose(net, [0, 2, 3, 1]) # (B, 1, 1, 512)
    outputs = slim.conv2d(net, 1, (1, 1), activation_fn=None,
                          normalizer_fn=None, padding='VALID', scope='fully_connected')
    outputs = tf.squeeze(outputs, axis=[1, 2], name='logits')
    return outputs

# pylint: disable=E1129
def train_arg_scope(weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True):
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.initializers.variance_scaling_initializer(),
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
