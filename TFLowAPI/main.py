# https://stackoverflow.com/questions/40822234/how-to-read-cifar-10-dataset-in-tensorflow

import numpy as np
import tensorflow as tf
import cf_data

def t(x):
	print(type(x))

def get_train_data_iterator(need_label = False):
	train_data = cf_data.load_train_data()
	train_imgs = train_data[b'data']
	train_labels = train_data[b'labels']
	dataset = tf.data.Dataset.from_tensor_slices({'img': train_imgs, 'label': train_labels})\
		if need_label else tf.data.Dataset.from_tensor_slices(train_imgs)
	return dataset.make_one_shot_iterator()

def CNN(raw_input):
	conv2d1 = tf.layers.Conv2D(filters = 64, kernel_size = [5, 5])
	conv2d2 = tf.layers.Conv2D(filters = 32, kernel_size = [5, 5])

	net = tf.transpose(tf.reshape(raw_input, [-1, 3, 32, 32]), perm = [0, 2, 3, 1])
	#net = conv2d1(net)
	#net = conv2d2(net)

	return raw_input, net

if __name__ == '__main__':

	#writer = tf.summary.FileWriter('./tensorboard')
	#writer.add_graph(tf.get_default_graph())

	train_data_iterator = get_train_data_iterator()
	train_data = train_data_iterator.get_next()

	sess = tf.Session()

	#for i in range(10):
	#print(train_data)
	x, y = sess.run(CNN(train_data))
	y = y[0][0:32][0:32][0]
	with open('debug.txt', 'w') as f:
		f.write(np.array2string(x, threshold = 1e10))
		f.write('\n\n\n')
		f.write(np.array2string(y, threshold = 1e10))