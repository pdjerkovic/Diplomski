import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import scipy.misc
from skimage.measure import compare_psnr as psnr, compare_ssim as ssim

def ocjena(gt, srcnn, bicubic, config, save_dir, ime, pom, first):

	if pom:
		save_dir = os.path.join(save_dir, os.path.splitext(ime)[0])
		if not os.path.exists(save_dir):	
			os.makedirs(save_dir)
	
	save_dir=os.path.join(save_dir, "Ocjena.txt")
	if first:
		f = open(save_dir,"w+")
	else:
		f = open(save_dir, "a+")
	
	if config.rgb:
		f.write("****" + ime + "*****\n")
		f.write("RGB prostor: PSNR ocjena SRCNN: " + str(psnr(gt,srcnn)) + "\n")
		f.write("RGB prostor: PSNR ocjena BI: " + str(psnr(gt, bicubic)) + "\n")
		f.write("***\n")
		f.write("RGB prostor: SSIM ocjena SRCNN:" + str(ssim(gt,srcnn,multichannel=True)) + "\n")
		f.write("RGB prostor: SSIM ocjena BI: " + str(ssim(gt,bicubic,multichannel=True)) + "\n")
		f.write("\n")
	else:
		gt_y = gt[:,:,0]
		gt_cr = gt[:,:,1]
		gt_cb = gt[:,:,2]
		srcnn_y = srcnn[:,:,0]
		srcnn_cr = srcnn[:,:,1]
		srcnn_cb = srcnn[:,:,2]
		bi_y = bicubic[:,:,0]
		bi_cr = bicubic[:,:,1]
		bi_cb = bicubic[:,:,2]		
		f.write("*****" + ime + "******\n")
		f.write("PSNR ocjena Y kanala SRCNN: " + str(psnr(gt_y, srcnn_y)) + "\n")
		f.write("PSNR ocjena Y kanala BI: " + str(psnr(gt_y, bi_y)) + "\n")
		f.write("*********************\n")
		f.write("SSIM ocjena Y kanala SRCNN: " + str(ssim(gt_y, srcnn_y)) + "\n")
		f.write("SSIM ocjena Y kanala BI: " + str(ssim(gt_y, bi_y)) + "\n")
		f.write("***************************************************\n")
		f.write("PSNR ocjena Cr kanala SRCNN: " + str(psnr(gt_cr, srcnn_cr))+"\n")
		f.write("PSNR ocjena Cr kanala BI: " + str(psnr(gt_cr, bi_cr))+"\n")
		f.write("*********************\n")
		f.write("SSIM ocjena Cr kanala SRCNN: " + str(ssim(gt_cr, srcnn_cr))+"\n")
		f.write("SSIM ocjena Cr kanala BI: " + str(ssim(gt_cr, bi_cr))+"\n")
		f.write("***************************************************\n")
		f.write("PSNR ocjena Cb kanala SRCNN: " + str(psnr(gt_cr, srcnn_cr)) + "\n")
		f.write("PSNR ocjena Cb kanala BI: " + str(psnr(gt_cr, bi_cr))+"\n")
		f.write("*********************\n")
		f.write("SSIM ocjena Cb kanala SRCNN: " + str(ssim(gt_cb, srcnn_cb))+"\n")
		f.write("SSIM ocjena Cb kanala BI: " + str(ssim(gt_cb, bi_cb))+"\n")
		f.write("***************************************************\n")
		
	
	f.close()	
	
def conv2d(images, W, config):
	
	#if config.trening:
		return tf.nn.conv2d(images, W, strides=[1, 1, 1, 1], padding='VALID')
	#else:
	#	return tf.nn.conv2d(images, W, strides=[1, 1, 1, 1], padding='SAME') 
	#mogli smo da stavimo SAME kako bismo u testiranju prosli bez igranja sa dimenzijama i imali iste dimenzije kao original; mana - ivice
	#takodje mozemo opet da parcamo sliku i spajamo segmente kod testiranja
	#ali za -korak=l_size(drugaciji slajd odvodi piksalizaciji) ali to je nepotrebna komplikacija
def SRCNN(images, config):
	
	if config.rgb:
		c=3
	else:
		c=1
	# f1 = 9, f2=1, f3 = 5, n1 = 64, n2 = 32;  -----------H  W  C  N---------------
	tezine = {'w1' : tf.Variable(tf.random_normal([9, 9, c, 64], stddev=1e-3)),
			   'w2' : tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3)),
			   'w3' : tf.Variable(tf.random_normal([5, 5, 32, c], stddev=1e-3))}

	
	biases = {'b1' : tf.Variable(tf.zeros([64])),
			  'b2' : tf.Variable(tf.zeros([32])),
			  'b3' : tf.Variable(tf.zeros([c]))}

	conv1 = tf.nn.relu(conv2d(images, tezine['w1'], config) + biases['b1'])
	conv2 = tf.nn.relu(conv2d(conv1, tezine['w2'], config) + biases['b2'])
	conv3 = conv2d(conv2, tezine['w3'], config) + biases['b3']
	return conv3
	
def ucitaj_ckpt(sess, saver, config):
	print(" ... Ucitavanje sacuvanih ... ... ...")
	if config.rgb:
		model_dir="srcnn_color" #"%s_%s_%s" %("srcnn", "color", config.scale)
	else:
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
	if config.rgb:
		model_dir= "srcnn_color"  # "%s_%s_%s" % ("srcnn", "color", config.scale)
	else:
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
def modcrop(image, scale=3):
	#if image.shape[2] == 1:
	#	size = image.shape
	#	size -= np.mod(size, scale)
	#	image = image[0:size[0], 0:size[1]]
	#else:
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
# 