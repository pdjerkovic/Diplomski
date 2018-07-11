import tensorflow as tf
from dodaci import modcrop, ucitaj_ckpt, sacuvaj_ckpt
import numpy as np
import os
import time
import cv2 as cv
import glob
import scipy.misc

# Inicijalizacija vrijednosti kao sto je sugerisano u radu
# label - ground truth images

def preprocesiranje(dirpath, config):
	
	data = []
	label = []
	counter = 0
	datai = glob.glob(os.path.join(dirpath, "*.*"))
	padding = abs(config.i_size- config.l_size) / 2
	# print(datai)
	for i in datai:

		img = cv.imread(i)
		# cv.imshow('image',img)
		# cv.waitKey(0)
		# cv.destroyAllWindows()
		img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
		img = img / 255.

		im_label = modcrop(img)
		size = im_label.shape
		h = size[0]
		w = size[1]
		im_temp = scipy.misc.imresize(im_label, 1./config.scale, interp='bicubic') #deprecated, nova: interpolation.zoom, rade isto
		im_input = scipy.misc.imresize(im_temp, size, interp='bicubic') 
	
		# korak preprocesiranja - nalazenje subslika
		for x in range(0, h - config.i_size, config.stride):
			for y in range(0, w - config.i_size, config.stride):
				subim_input = im_input[x : x + config.i_size, y : y + config.i_size]
				subim_label = im_label[int(x + padding) : int(x + padding + config.l_size), int(y + padding) : int(y + padding + config.l_size)]
				
				subim_input = subim_input.reshape([config.i_size, config.i_size, 1])
				subim_label = subim_label.reshape([config.l_size, config.l_size, 1])
				
				data.append(subim_input)
				label.append(subim_label)
				counter += 1


	# print(counter) #broj subslika
	# randomiziranja radi
	order = np.random.choice(counter, counter, replace=False)
	data = np.array([data[i] for i in order])
	label = np.array([label[i] for i in order])

	# print(data.shape)
	# print(label.shape)

	return data, label 

# razlikovacemo ovu funkciju u fazi treniranja i fazi testiranja, kako bismo mogli da dobijemo krajnji rezultat (zasto SAME a ne VALID?)
def conv2d(x, W, config):
	
	if config.trening:
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
	else:
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #kako bismo mogli da primijenimo mrezu u fazi testiranja na bilo koju velicinu
		# bez igranja sa dimenzija kanala; ovo i daje bolji rezultat
		# mana: granice. jos jedno rijesenje problema: parcanje slike i -stride=l_size(+oprez kod dimenzija konacnog rezultata) ali to je nepotrebna komplikacija
def SRCNN(x, config):
	# Kao u radu: osim cinjenice da kod rekonstrukcije imamo padding, i da zadnji sloj nema manju stopu ucenja
	# elem, to je neoptimalno
	# f1 = 9, f3 = 5, n1 = 64, n2 = 32;  -----------H  W  C  N---------------
	weights = {'w1' : tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3)),
			   'w2' : tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3)),
			   'w3' : tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3))}

	biases = {'b1' : tf.Variable(tf.zeros([64])),
			  'b2' : tf.Variable(tf.zeros([32])),
			  'b3' : tf.Variable(tf.zeros([1]))}

	conv1 = tf.nn.relu(conv2d(x, weights['w1'], config) + biases['b1'])
	conv2 = tf.nn.relu(conv2d(conv1, weights['w2'], config) + biases['b2'])
	conv3 = conv2d(conv2, weights['w3'], config) + biases['b3']
	return conv3

# Treniranje
def poziv(img_dir, config):
	
	train_data, train_label = preprocesiranje(img_dir, config)
	print(train_data.shape, ' ', train_label.shape)
	images = tf.placeholder(tf.float32, [None, config.i_size, config.i_size, 1], name='images')
	labels = tf.placeholder(tf.float32, [None, config.l_size, config.l_size, 1], name='labels')
	

	model = SRCNN(images, config)
	loss = tf.reduce_mean(tf.square(labels - model))
	optimizer = tf.train.GradientDescentOptimizer(config.koef_ucenja).minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Treniranje pocelo!')
		start_time = time.time()
		counter = 0
		saver = tf.train.Saver()

		if ucitaj_ckpt(sess, saver, config):
			print('Sacuvane vrijednosti ucitane...')
		else:
			print('Greska sa checkpointom (provjeri ima li ga...)')


		batch = config.batch
		for epoch in range(config.br_epoha):
			epoch_loss = 0
			idx_batch = len(train_data) // batch
			for i in range(idx_batch):
				epoch_images = train_data[i * batch : (i + 1) * batch]
				epoch_labels = train_label[i * batch : (i + 1) * batch]

				_, c = sess.run([optimizer, loss], feed_dict = {images: epoch_images, labels: epoch_labels})
				epoch_loss += c
				counter += 1

				if counter % 10 == 0:
					print('Epoha:[', epoch + 1, '] - duration:', time.time() - start_time, '; korak:', counter, ' ;gubitak:', c)
				
				# cuvaj i manji br epoha, zbog racunara i brzine. <200, 500
				if counter % 50 == 0:
					sacuvaj_ckpt(sess, counter, saver)



