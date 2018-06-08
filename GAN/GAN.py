import tensorflow as tf
import cf_data

MODEL_DIR = './model'
TRAIN_STEPS = 10000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

def discriminator_model(features, labels, mode, params):
	net = tf.feature_column.input_layer(
		features, params['columns']
	)
	net = tf.reshape(net, [-1, 32, 32, 3])
	net = tf.layers.conv2d(net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu)
	net = tf.layers.max_pooling2d(net, pool_size = [2, 2], strides = [2, 2])
	net = tf.layers.conv2d(net, filters = 64, kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu)
	net = tf.layers.max_pooling2d(net, pool_size = [2, 2], strides = [2, 2])
	net = tf.layers.flatten(net)
	net = tf.layers.dense(net, units = 384, activation = tf.nn.relu)
	net = tf.layers.dense(net, units = 192, activation = tf.nn.relu)
	logits = tf.layers.dense(net, params['n_classes'], activation = None)
	predicted_classes = tf.argmax(tf.nn.softmax(logits), 1)

	# predict
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predicted_classes[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions = predictions)

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

def main(argv):
	#train_data, test_data = load_data()
	train_data = cf_data.load_train_data(5)
	test_data = train_data

	discriminator_classifier = tf.estimator.Estimator(
		model_fn = discriminator_model,
		model_dir = MODEL_DIR,
		params = {
			'columns': [tf.feature_column.numeric_column(
				key = 'img', shape = [32, 32, 3]
			)],
			'n_classes': 10
		}
	)

	discriminator_classifier.train(
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