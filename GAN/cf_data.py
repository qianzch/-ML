# https://www.cs.toronto.edu/~kriz/cifar.html

import pickle, png
import numpy as np
import tensorflow as tf

DATA_DIR 		= 'dataset/'
META_FILE 		= DATA_DIR + 'batches.meta'
DATA_BATCHES 	= [(DATA_DIR + 'data_batch_' + str(i)) for i in range(1, 6)]
TEST_BATCH 		= DATA_DIR + 'test_batch'
NOISE_SIZE 		= [8, 8, 1]

# ------------------------------------------ Interface -----------------------------------------------

def load_data():
	return load_train_data(), load_test_data()

def save_image(r, g, b, filename = 'image.png'):
	img_data = []
	for i in range(32):
		row_data = []
		for j in range(32):
			row_data += [r[i * 32 + j], g[i * 32 + j], b[i * 32 + j]]
		img_data.append(row_data)
	png_write = png.Writer(32, 32)
	with open(filename, 'wb') as f:
		png_write.write(f, img_data)

def train_input_fn(imgs, labels, batch_size):
	data = {'img': imgs, 'noise': np.random.random(([len(imgs)] + NOISE_SIZE))}
	dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	dataset = dataset.shuffle(int(5e5)).repeat().batch(batch_size)
	return dataset.make_one_shot_iterator().get_next()

# ---------------------------------------- End Interface ---------------------------------------------

def load_train_data(size = 5):
	# smaller data size for dev
	if size > 5 or size < 1:
		raise ValueError('size must be in range [1, 5]')

	train_data = {}
	for i in range(size):
		with open(DATA_BATCHES[i], 'rb') as f:
			train_data.update(pickle.load(f, encoding = 'bytes'))

	return train_data

def load_test_data():
	with open(TEST_BATCH, 'rb') as f:
		test_data = pickle.load(f, encoding = 'bytes')
	return test_data

if __name__ == '__main__':
	data = load_train_data(2)
	#r = data[b'data'][0][0: 1024]
	#g = data[b'data'][0][1024: 2048]
	#b = data[b'data'][0][2048: 3072]
	#save_image(r, g, b)