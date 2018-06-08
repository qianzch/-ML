# http://slazebni.cs.illinois.edu/spring17/lec11_gan.pdf gan model ppt
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/ gan blog
# https://github.com/tensorlayer/tensorlayer/issues/17 get layer variables

import tensorflow as tf
import cf_data

MODEL_DIR = './foo'
TRAIN_STEPS = 10000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NOISE_SIZE = cf_data.NOISE_SIZE

def GAN_model(features, labels, mode, params):
	# generate images from noise
	g_net = tf.feature_column.input_layer(features, params['g_columns'])
	g_net = tf.reshape(g_net, [-1] + NOISE_SIZE)
	g_net = tf.layers.conv2d_transpose(g_net, filters = 128, kernel_size = [9, 9])
	g_net = tf.layers.conv2d_transpose(g_net, filters = 64, kernel_size = [17, 17])
	g_net = tf.layers.conv2d_transpose(g_net, filters = 3, kernel_size = [1, 1])

	# REAL
	d_net = tf.feature_column.input_layer(features, params['d_columns'])
	d_net = tf.reshape(d_net, [-1, 32, 32, 3])
	# --------------------------------------------------- layers below will be reused ------------------------------------------------
	d_net = tf.layers.conv2d(d_net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, name = 'd_conv_1')
	d_net = tf.layers.max_pooling2d(d_net, pool_size = [2, 2], strides = [2, 2], name = 'd_pooling_1')
	d_net = tf.layers.conv2d(d_net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, name = 'd_conv_2')
	d_net = tf.layers.max_pooling2d(d_net, pool_size = [2, 2], strides = [2, 2], name = 'd_pooling_2')
	d_net = tf.layers.flatten(d_net, name = 'd_flatten')
	d_net = tf.layers.dense(d_net, units = 384, activation = tf.nn.relu, name = 'd_dense_1')
	d_net = tf.layers.dense(d_net, units = 192, activation = tf.nn.relu, name = 'd_dense_2')
	d_logits = tf.layers.dense(d_net, params['n_classes'], activation = None, name = 'd_dense_3')
	# --------------------------------------------------------- end of reuse ---------------------------------------------------------
	d_real = tf.nn.softmax(d_logits)

	# FAKE
	d_net = g_net # use images generated from noise as input
	d_net = tf.layers.conv2d(d_net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, name = 'd_conv_1', reuse = True)
	d_net = tf.layers.max_pooling2d(d_net, pool_size = [2, 2], strides = [2, 2], name = 'd_pooling_1', reuse = True)
	d_net = tf.layers.conv2d(d_net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu, name = 'd_conv_2', reuse = True)
	d_net = tf.layers.max_pooling2d(d_net, pool_size = [2, 2], strides = [2, 2], name = 'd_pooling_2', reuse = True)
	d_net = tf.layers.flatten(d_net, name = 'd_flatten', reuse = True)
	d_net = tf.layers.dense(d_net, units = 384, activation = tf.nn.relu, name = 'd_dense_1', reuse = True)
	d_net = tf.layers.dense(d_net, units = 192, activation = tf.nn.relu, name = 'd_dense_2', reuse = True)
	d_logits = tf.layers.dense(d_net, params['n_classes'], activation = None, name = 'd_dense_3', reuse = True)
	d_fake = tf.nn.softmax(d_logits)

	d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1.0 - d_fake))
	g_loss = -tf.reduce_mean(tf.log(d_fake))
	




	#predicted_classes = tf.argmax(tf.nn.softmax(logits), 1)

	'''
	# predict
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predicted_classes[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions = predictions)
	'''

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	accuracy = tf.metrics.accuracy(
		labels = labels,
		predictions = predicted_classes,
		name = 'Accuracy'
	)
	metrics = {'Accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	# eval
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
				mode, loss = loss, eval_metric_ops = metrics)

	# train
	optimizer = tf.train.AdagradOptimizer(learning_rate = LEARNING_RATE)
	train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)
	'''

def main(argv):
	#train_data, test_data = load_data()
	train_data = cf_data.load_train_data(5)
	test_data = train_data

	d_columns = [tf.feature_column.numeric_column(key = 'img', shape = [32, 32, 3])]
	g_columns = [tf.feature_column.numeric_column(key = 'noise', shape = NOISE_SIZE)]
	GAN = tf.estimator.Estimator(
		model_fn = GAN_model,
		model_dir = MODEL_DIR,
		params = {
			'd_columns': d_columns,
			'g_columns': g_columns,
			'n_classes': 10
		}
	)

	GAN.train(
		input_fn = lambda:cf_data.train_input_fn(
			train_data[b'data'], train_data[b'labels'], BATCH_SIZE
		),
		steps = TRAIN_STEPS
	)

	'''
	eval_result = classifier.evaluate(
		input_fn = lambda:cf_data.train_input_fn(
			train_data[b'data'], train_data[b'labels'], BATCH_SIZE
		))
	print(type(eval_result), eval_result)
	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	'''
if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)