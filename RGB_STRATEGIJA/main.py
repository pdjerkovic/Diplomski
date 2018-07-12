
# https://github.com/pdjerkovic/Diplomski
# Reference u README fajlu

from treniranje import trening
from testiranje import test
import argparse
import os

parser = argparse.ArgumentParser(description='Postavljanje hiperparametara mreže. Smanjivati batch, i_size i l_size sa ubrzanje. Imati na umu da mora da vazi: i_size = l_size + 12')

parser.add_argument('--br_epoha',  dest='br_epoha', 
					help='Broj epoha. Default: [2000]', type=int, default=2000)
parser.add_argument('-checkpoint', default='checkpoint', 
					dest='check_dir', help='Checkpoint dir. za cuvanje predjenih epoha [checkpoint]')
parser.add_argument('-batch',  dest='batch', 
					help='Br. subslika koji se predaje SRCNN-u u jednom prolazu [128]', type=int, default=128)
parser.add_argument('-i_size', dest='i_size', 
					help='Velicina LR segmenta (33x33) [33]', type=int, default=33)
parser.add_argument('-l_size', dest='l_size',
					help='Velicina HR segmenta (21x21) [21]', type=int, default=21)
parser.add_argument('-stride', dest='stride',
					help='Korak kod rezanja na subslike [14]', default=14)
parser.add_argument('-eta', dest='koef_ucenja',
					type=float, help="Koef. ucenja [1e-4]", default=1e-4) #koef.ucenja zadnjeg sloja 10 puta manji
parser.add_argument('-s',  dest='scale', 
					help='Faktor skaliranja [3]', type=int, default=3)
parser.add_argument('-sample_dir', dest='sample_dir', 
					help='Mjesto gdje se cuvaju rezultati [rezultati]', default='rezultati')
parser.add_argument('-trening', dest='trening', 
					help="True za fazu treniranje, false za test (False)", default=False)
parser.add_argument('-train_dir', dest='train_dir', 
					help="Trening-baza(mora biti smjestena u folderu Treniranje) [Basic]", default='Basic')
parser.add_argument('-test_dir', dest='test_dir', 
					help="Folder/fajl za testiranje(mora biti smjesten u Testiranje) [MojTest]", default='MojTest')
parser.add_argument('-e', dest='uvecanje', 
					help="Direktno uvecanje slike [False]", default=False)					

# parser.add_argument('-rgb', dest='rgb', 
#					help="RGB-strategija", default=False) <---unapredjenje

#analiza - psnr

# FSRCNN 



config = parser.parse_args()

if config.i_size!=config.l_size+12:
	print("Moram usaglasiti parametre...")
	config.i_size = config.l_size+12


if not os.path.exists(config.check_dir):
    os.makedirs(config.check_dir)
if not os.path.exists(config.sample_dir):
    os.makedirs(config.sample_dir)
# oznake foldera za razlicite skalare
if config.trening:
	img_dir = os.path.join(os.getcwd(), "Treniranje")
	img_dir = os.path.join(img_dir, config.train_dir)
	trening(img_dir, config)
else:
	save_dir = os.path.join(os.getcwd(), config.sample_dir)
	# save_dir = os.path.join(save_dir, config.sample_dir)
	img_dir = os.path.join(os.getcwd(), "Testiranje")
	img_dir = os.path.join(img_dir, config.test_dir)
	if config.uvecanje:
		save_dir = os.path.join(save_dir, "Uvecane slike")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
	#	uvecaj(img_dir, save_dir, config)
	#else:
	test(img_dir, save_dir, config)