import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import scipy.misc


def conv2d(images, W, config):
	
	#if config.trening:
		return tf.nn.conv2d(images, W, strides=[1, 1, 1, 1], padding='VALID') #HAJMO!
	#else:
	#	return tf.nn.conv2d(images, W, strides=[1, 1, 1, 1], padding='SAME') 
	#mogli smo da stavimo SAME kako bismo u testiranju prosli bez igranja sa dimenzijama i imali iste dimenzije kao original; mana - ivice
	#takodje mozemo opet da parcamo sliku i spajamo segmente kod testiranja
	#ali za -stride=l_size(drugaciji slajd odvodi piksalizaciji) ali to je nepotrebna komplikacija
def SRCNN(images, config):
	# Kao u radu: osim cinjenice da da zadnji sloj nema manju stopu ucenja
	# elem, to je neoptimalno
	# f1 = 9, f3 = 5, n1 = 64, n2 = 32;  -----------H  W  C  N---------------
	tezine = {'w1' : tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3)),
			   'w2' : tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3)),
			   'w3' : tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=1e-3))}

	
	biases = {'b1' : tf.Variable(tf.zeros([64])),
			  'b2' : tf.Variable(tf.zeros([32])),
			  'b3' : tf.Variable(tf.zeros([3]))}

	conv1 = tf.nn.relu(conv2d(images, tezine['w1'], config) + biases['b1'])
	conv2 = tf.nn.relu(conv2d(conv1, tezine['w2'], config) + biases['b2'])
	conv3 = conv2d(conv2, tezine['w3'], config) + biases['b3']
	return conv3
	
def ucitaj_ckpt(sess, saver, config):
	print(" ... Ucitavanje sacuvanih ... ... ...")
	model_dir = "srcnn" # "%s_%s" % ("srcnn", config.scale) # oprezno ovdje, za novi faktor skaliranja se po pravilu treba obuciti nova mreza
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
	model_dir = "srcnn" # "%s_%s" % ("srcnn", config.scale)
	checkpoint_dir = os.path.join(config.check_dir, model_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def revert(im):
	im = im * 255
	im[im > 255] = 255
	im[im < 0] = 0
	return im.astype(np.uint8)

# kako bismo sto cisto prosli kod skaliranja sredi dimenzije da nema ostataka
# takodje, prevedi 3-kanalni na 1-kanalni, radi treninga na Y-kanalu

	
#samo sirinu i visinu mijenjaj, trebaju mi boje
	
def modcrop_color(image, scale=3):
	size = image.shape[0:3]
	size -= np.mod(size, scale)
	image = image[0:size[0], 0:size[1], :]
	return image

# scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float) <--- flatten=True za grayscale
