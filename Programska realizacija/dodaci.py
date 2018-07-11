import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import scipy.misc


# Referenca: https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py
def ucitaj_ckpt(sess, saver, config):
	print(" ... Ucitavanje sacuvanih ... ... ...")
	model_dir = "%s_%s" % ("srcnn", config.l_size) #config.scale
	checkpoint_dir = os.path.join(config.check_dir, model_dir)

	# Nadji barem jedno sacuvano stanje mreze
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
	    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	    print('NADJENO - - -', os.path.join(checkpoint_dir, ckpt_name))
	    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	    return True
	else:
	    return False

def sacuvaj_ckpt(sess, step, saver, config):
	model_name = 'SRCNN.model'
	model_dir = "%s_%s" % ("srcnn", config.l_size) #config.scale
	checkpoint_dir = os.path.join(config.check_dir, model_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def revert(im):
	im = im * 255
	im[im > 255] = 255
	im[im < 0] = 0
	return im.astype(np.uint8)

# kako bismo sto ''cistije'' prosli kod skaliranja sredi dimenzije da nema ostataka
# takodje, prevedi 3-kanalni na 1-kanalni, radi treninga na Y-kanalu
def modcrop(image, scale=3):
	if image.shape[2] == 1:
		size = image.shape
		size -= np.mod(size, scale)
		image = image[0:size[0], 0:size[1]]
	else:
		size = image.shape[0:2]
		size -= np.mod(size, scale)
		image = image[0:size[0], 0:size[1], 0]
	return image
	
#samo sirinu i visinu mijenjaj, trebaju mi boje
	
def modcrop_color(image, scale=3):
	size = image.shape[0:2]
	size -= np.mod(size, scale)
	image = image[0:size[0], 0:size[1], :]
	return image

# scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float) <--- flatten=True za grayscale
