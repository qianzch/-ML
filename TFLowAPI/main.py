# https://stackoverflow.com/questions/40822234/how-to-read-cifar-10-dataset-in-tensorflow

import numpy as np
import tensorflow as tf
import cf_data

def t(x):
	print(type(x))
def s(x):
	print(x.shape)

def get_train_data_iterator(batch_size  = 128, need_label = False):
	train_data = cf_data.load_train_data()
	train_imgs = train_data[b'data'] / 255.0 # to [0, 1]
	train_labels = train_data[b'labels']
	dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))\
		if need_label else tf.data.Dataset.from_tensor_slices(train_imgs)
	return dataset.batch(128).make_one_shot_iterator()

def CNN(imgs, labels = None):
	conv2d1 = tf.layers.Conv2D(filters = 64, kernel_size = [5, 5])
	conv2d2 = tf.layers.Conv2D(filters = 32, kernel_size = [5, 5])
	dense1 = tf.layers.Dense(units = 128, activation = tf.nn.relu)
	dense2 = tf.layers.Dense(units = 64, activation = tf.nn.relu)
	dense3 = tf.layers.Dense(units = 10, activation = None)

	net = tf.transpose(tf.reshape(imgs, [-1, 3, 32, 32]), perm = [0, 2, 3, 1])

	net = conv2d1(net)
	net = conv2d2(net)
	net = tf.layers.flatten(net)
	net = dense1(net)
	net = dense2(net)
	net = dense3(net)
	
	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = net)

	net = tf.nn.softmax(net)
	pred = tf.argmax(net, 1)
	
	tf.summary.scalar('Loss', loss)

	return pred, loss

if __name__ == '__main__':

	train_data_iterator = get_train_data_iterator(need_label = True)
	train_imgs, train_labels = train_data_iterator.get_next()

	pred = CNN(train_imgs, train_labels)
	init = tf.global_variables_initializer()
	summary = tf.summary.merge_all()

	sess = tf.Session()
	sess.run(init)

	writer = tf.summary.FileWriter('./tensorboard', sess.graph)

	for i in range(10):
		pred_res, summary_str = sess.run([pred, summary])
		writer.add_summary(summary_str, i)