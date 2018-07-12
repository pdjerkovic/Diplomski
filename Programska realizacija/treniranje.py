import tensorflow as tf
from dodaci import modcrop, ucitaj_ckpt, sacuvaj_ckpt, SRCNN
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
	padding = abs(config.i_size- config.l_size) / 2 # i_size > l_size 
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
		# print(h, w, im_input.shape)
	
		# korak preprocesiranja - nalazenje subslika
		for images in range(0, h - config.i_size, config.stride):
			for y in range(0, w - config.i_size, config.stride):
				subim_input = im_input[images : images + config.i_size, y : y + config.i_size]
				subim_label = im_label[int(images + padding) : int(images + padding + config.l_size), int(y + padding) : int(y + padding + config.l_size)]
				# print(int(images + padding), int(images + padding + config.l_size), int(y + padding), int(y + padding + config.l_size) )
				subim_input = subim_input.reshape([config.i_size, config.i_size, 1])
				subim_label = subim_label.reshape([config.l_size, config.l_size, 1])
				
				data.append(subim_input)
				label.append(subim_label)
				counter += 1


	print("Napravili smo: ", counter, " sub-slika")
	# randomiziranja radi
	order = np.random.choice(counter, counter, replace=False)
	data = np.array([data[i] for i in order])
	label = np.array([label[i] for i in order])

	# print(data.shape)
	# print(label.shape)

	return data, label 


# Treniranje
def trening(img_dir, config):
	
	train_data, train_label = preprocesiranje(img_dir, config)
	# print(train_data.shape, ' ', train_label.shape)
	images = tf.placeholder(tf.float32, [None, config.i_size, config.i_size, 1], name='images')
	labels = tf.placeholder(tf.float32, [None, config.l_size, config.l_size, 1], name='labels')
	

	model = SRCNN(images, config)
	loss = tf.reduce_mean(tf.square(labels - model)) #na cjelokupni segment
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
			print('Greska sa checkpointom (provjerite ima li ga...)')


		batch = config.batch
		for epoha in range(config.br_epoha):
			epoha_loss = 0
			idx_batch = len(train_data) // batch
			for i in range(idx_batch):
				epoha_images = train_data[i * batch : (i + 1) * batch]
				epoha_labels = train_label[i * batch : (i + 1) * batch]

				_, c = sess.run([optimizer, loss], feed_dict = {images: epoha_images, labels: epoha_labels})
				epoha_loss += c
				counter += 1

				if counter % 10 == 0:
					print("Epoha: [%2d], korak: [%2d], vrijeme: [%4.4f], gubitak: [%.8f]" \
						% ((epoha+1), counter, time.time()-start_time, c))
				
				# cuvaj 
				if counter % 500 == 0:
					sacuvaj_ckpt(sess, counter, saver, config)



