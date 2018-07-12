import tensorflow as tf
from dodaci import revert, modcrop_color, ucitaj_ckpt, SRCNN
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
	# im = cv.cvtColor(img_input, cv.COLOR_BGR2YCR_CB)
	img = img_input / 255. # im; normalizacija (da li je potrebna normalizacija?)

	im_label = modcrop_color(img, scale=config.scale) # ycrcb sema
	color_base = modcrop_color(img, scale=config.scale)
	# im_label = im_label / 255.

	#im_input = scipy.ndimage.interpolation.zoom(im_label, (1./config.scale), prefilter=False)
	#im_input = scipy.ndimage.interpolation.zoom(im_input, (config.scale/1.), prefilter=False) #revert...
	size = im_label.shape
	h = size[0]
	w = size[1]
	# print(h, " ", w)
	im_blur = scipy.misc.imresize(im_label, 1. / config.scale, interp='bicubic')
	im_input = scipy.misc.imresize(im_blur, config.scale * 1.0, interp='bicubic')

	data = np.array(im_input[:,:,0:3]).reshape([1, h, w, 3])
	color = np.array(color_base[:,:,1:3])      #im_input

	label = np.array(modcrop_color(img_input, config.scale))

	return data, label, color #data=im_input

# za direktno uvecanje slike

def direktni_podaci(path, config):

	data = []
	color = []
	img = cv.imread(path) 	 
	# im = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
	img = img / 255. 
	size = img.shape
	img_temp = scipy.misc.imresize(img, [size[0] * config.scale, size[1] * config.scale], interp='bicubic')  #ovako nalazimo HR sliku scale*size,
	# mozemo testirat i za originalnu sliku bez ovog skaliranja - njen kvalitetniji pandan
	# color_temp = scipy.misc.imresize(im, [size[0] * config.scale, size[1] * config.scale], interp='bicubic') 
	# im_label = img_temp[:, :, 0]   
	# im_color = color_temp[:, :, 1:3]

	data = np.array(im_temp).reshape([1, img.shape[0] * config.scale, img.shape[1] * config.scale, 3])
	color = np.array(im_color)
	

	return data, color


def test(path, save_dir, config):

	images = tf.placeholder(tf.float32, [None, None, None, 3], name='images') 
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
			pom = True
			if config.uvecanje:
				poruka="UVECAVANJE SLIKE"
				poruka2="Uvecavam sliku..."
			else:
				poruka="TESTIRANJE SLIKE"
				poruka2="Testiram sliku..."
		else:
			pom = False
			data = glob.glob(os.path.join(path, "*.*"))
			if config.uvecanje:
				save_dir=os.path.join(save_dir,(str(config.scale)+"x"+config.test_dir))
				poruka = "UVECAVANJE SLIKA"
				poruka2 = "Uvecavam sliku... "
			else:
				save_dir=os.path.join(save_dir,config.test_dir)
				poruka = "TESTIRANJE SLIKA"
				poruka2 = "Testiranje na... "
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
		
		print(poruka, "...");
		for i in data:
			print(poruka2, os.path.basename(i))
			if config.uvecanje:
				test_data, color = direktni_podaci(i, config)
			else:
				test_data, test_label, color = podaci(i, config)

			# print(color.shape)
			print(test_data.shape)
			izlaz = mreza.eval({images: test_data}) #labels: test_label
			izlaz = izlaz.squeeze() #postprocesiranje
			result = revert(izlaz) #result_bw; revert obavezno kako bismo se vratili na prave vrijednosti boje - vidjeti stare rezultate
			print("resultat ima ", izlaz.shape)
			# color = revert(color)
			# result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
			# result[:, :, 0] = result_bw
			# result_color = np.zeros([result_bw.shape[0], result_bw.shape[1], 2])
			p = (int)((color.shape[0]-result.shape[0])/2) # p = 6
			# result_color = color[p:(color.shape[0])-p, p:(color.shape[1])-p,0:2]			
			# result[:, :, 1:3] = result_color # color
			# result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB) #aha
			result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
			if config.uvecanje:
				if pom:
					save_dir = os.path.join(save_dir, (str(config.scale) + "x" + os.path.splitext(os.path.basename(i))[0]))
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
				save_path = os.path.join(save_dir, (str(config.scale)) + "x" + os.path.basename(i))
				scipy.misc.imsave(save_path, result)
			else:
				bicubic = scipy.misc.imresize(test_label, 1. / config.scale, interp='bicubic')
				bicubic = scipy.misc.imresize(bicubic, config.scale * 1.0, interp='bicubic')
				bicubic = cv.cvtColor(bicubic, cv.COLOR_BGR2RGB)
				# print(color.shape,bicubic.shape)
				bicubic = bicubic[p:(bicubic.shape[0])-p, p:bicubic.shape[1]-p, :] #kako bismo vidjeli iste dimenzije
				print("Bicubic ima shape ", bicubic.shape)
				# image_path1 = os.path.join(os.getcwd(), config.sample_dir)
				image_path1 = os.path.join(save_dir, os.path.splitext(os.path.basename(i))[0])
				if not os.path.exists(image_path1):
					os.makedirs(image_path1)
				save_path = os.path.join(image_path1, os.path.basename(i))
				scipy.misc.imsave(save_path, result)
				bicubic_path = os.path.join(image_path1, 'bicubic_' + os.path.basename(i))
				scipy.misc.imsave(bicubic_path, bicubic)
			print('Zavrseno sa... ', os.path.basename(i))
		print('ZAVRSENO ', poruka)
		print('Rezultati sacuvani u ', save_dir)

		
#function psnr=compute_psnr(im1,im2)
#if size(im1, 3) == 3,
#    im1 = rgb2ycbcr(im1);
#    im1 = im1(:, :, 1);
#end

#if size(im2, 3) == 3,
#    im2 = rgb2ycbcr(im2);
#    im2 = im2(:, :, 1);
#end

#imdff = double(im1) - double(im2);
#imdff = imdff(:);

#rmse = sqrt(mean(imdff.^2));
#psnr = 20*log10(255/rmse);

