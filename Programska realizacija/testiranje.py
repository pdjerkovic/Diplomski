import tensorflow as tf
from dodaci import revert, modcrop_color, ucitaj_ckpt
from treniranje import SRCNN
import numpy as np
import os
import time
import cv2 as cv
import scipy.misc
import scipy.ndimage
import math
import glob


def podaci(path, config):

	data = []
	label = []

	img_input = cv.imread(path)
	im = cv.cvtColor(img_input, cv.COLOR_BGR2YCR_CB)
	img = im / 255. # normalizacija

	im_label = modcrop_color(img, scale=config.scale) # ycrcb sema
	color_base = modcrop_color(im, scale=config.scale)
	# im_label = im_label / 255.

	#im_input = scipy.ndimage.interpolation.zoom(im_label, (1./config.scale), prefilter=False)
	#im_input = scipy.ndimage.interpolation.zoom(im_input, (config.scale/1.), prefilter=False) #revert...
	size = im_label.shape
	h = size[0]
	w = size[1]
	# print(h, " ", w)
	im_blur = scipy.misc.imresize(im_label, 1. / config.scale, interp='bicubic')
	im_input = scipy.misc.imresize(im_blur, config.scale * 1.0, interp='bicubic')

	data = np.array(im_input[:,:,0]).reshape([1, h, w, 1])
	color = np.array(color_base[:,:,1:3])      #im_input

	label = np.array(modcrop_color(img_input, config.scale))

	return data, label, color

# za direktno uvecanje slike

def direktni_podaci(path, config):

	data = []
	color = []
	img = cv.imread(path) 	 
	im = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
	img = im / 255. 
	size = img.shape
	img_temp = scipy.misc.imresize(img, [size[0] * config.scale, size[1] * config.scale], interp='bicubic')  #ovako nalazimo HR sliku scale*size,
	# mozemo testirat i za originalnu sliku bez ovog skaliranja - njen kvalitetniji pandan
	color_temp = scipy.misc.imresize(im, [size[0] * config.scale, size[1] * config.scale], interp='bicubic') 
	im_label = img_temp[:, :, 0]   
	im_color = color_temp[:, :, 1:3]

	data = np.array(im_label).reshape([1, img.shape[0] * config.scale, img.shape[1] * config.scale, 1])
	color = np.array(im_color)
	

	return data, color


def provjera(path, save_dir, config):

	images = tf.placeholder(tf.float32, [None, None, None, 1], name='images') 
	mreza = SRCNN(images, config)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())


		saver = tf.train.Saver()
		if ucitaj_ckpt(sess, saver, config):
			print('Uspjesno ucitavanje sacuvanih...')
		else:
			print('GRESKA pri ucitavanju sacuvanih! Provjeriti da li postoji checkpoint. Ako ne, sprovedite fazu treniranja prije testiranja.')
		
		if os.path.isfile(path):
			data=[path];
		else:
			data = glob.glob(os.path.join(path, "*.*"))
			save_dir=os.path.join(save_dir,config.test_dir)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
		print("Testiranje:");
		for i in data:
			print('Testiranje na...', os.path.basename(i))
			test_data, test_label, color = podaci(i, config)

			# print(color.shape)

			izlaz = mreza.eval({images: test_data}) #labels: test_label
			izlaz = izlaz.squeeze() #postprocesiranje
			result_bw = revert(izlaz) #revert obavezno kako bismo se vratili na prave vrijednosti boje - vidjeti stare rezultate
			# color = revert(color)
			result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
			result[:, :, 0] = result_bw
			# result_color = np.zeros([result_bw.shape[0], result_bw.shape[1], 2], dtype=np.uint8)
			# result_color = color[(color.shape[0]-result_bw.shape[0]):(color.shape[0]), (color.shape[1]-result_bw.shape[1]):(color.shape[1]),0:2]			
			result[:, :, 1:3] = color # result_color
			result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB) #aha

			bicubic = scipy.misc.imresize(test_label, 1. / config.scale, interp='bicubic')
			bicubic = scipy.misc.imresize(bicubic, config.scale * 1.0, interp='bicubic')
			bicubic = cv.cvtColor(bicubic, cv.COLOR_BGR2RGB)


			# image_path1 = os.path.join(os.getcwd(), config.sample_dir)
			image_path1 = os.path.join(save_dir, os.path.splitext(os.path.basename(i))[0])
			if not os.path.exists(image_path1):
				os.makedirs(image_path1)
			save_path = os.path.join(image_path1, os.path.basename(i))
			scipy.misc.imsave(save_path, result)
			bicubic_path = os.path.join(image_path1, 'bicubic_' + os.path.basename(i))
			scipy.misc.imsave(bicubic_path, bicubic)
			print('Zavrseno sa... ', os.path.basename(i))
		print('Zavrseno testiranje svih slika.')
		print('Savuvani rezultati u ', save_dir)

def uvecaj(path, save_dir, config):

	# uvecaj sliku za faktor scale i predaj je mrezi 
	images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
	mreza = SRCNN(images, config)
	with tf.Session() as sess:
		# pazi na inicijaliziranje vrijednosti sesije
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		if ucitaj_ckpt(sess, saver, config):
			print('Uspjesno ucitavanje sacuvanih...')
		else:
			print('GRESKA pri ucitavanju sacuvanih! Provjeriti da li postoji checkpoint. Ako ne, sprovedite fazu treniranja prije testiranja.')

		if os.path.isfile(path):
			data=[path];
		else:
			data = glob.glob(os.path.join(path, "*.*"))
			save_dir=os.path.join(save_dir,(str(config.scale)+"x"+config.test_dir))
			if not os.path.exists(save_dir):			
				os.makedirs(save_dir)
		print("Uvecavam slike...");
		for i in data:
			print('Uvecavanje slike...', os.path.basename(i))
			test_data, color = direktni_podaci(i, config)


			izlaz = mreza.eval({images: test_data}) #labels: test_label
			izlaz = izlaz.squeeze() #postprocesiranje
			result_bw = revert(izlaz)
			# result_color = np.zeros([result_bw.shape[0], result_bw.shape[1], 2], dtype=np.uint8)
			# result_color = color[0:result_bw.shape[0], 0:result_bw.shape[1],0:2]
			result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
			result[:, :, 0] = result_bw
			result[:, :, 1:3] = color #result_color
			result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
			# image_path1 = os.path.join(os.getcwd(), config.sample_dir)
			image_path1 = os.path.join(save_dir, (str(config.scale) + "x" + os.path.splitext(os.path.basename(i))[0]))
			if not os.path.exists(image_path1):
				os.makedirs(image_path1)
			save_path = os.path.join(image_path1, os.path.basename(i))
			scipy.misc.imsave(save_path, result)
			print('Zavrseno sa... ', os.path.basename(i))
		print('Zavrseno uvecavanje svih slika.')
		print("Sacuvano u folderu ", save_dir)
