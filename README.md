# Diplomski
**Specijalisticki rad - Konvolucione mreze super rezolucije**
 
 
 *Glavni rad* - **Konvolucione mreze super rezolucije SPEC.pdf**
 RAD - tex fajlovi
 
 **Programska realizacija:**
 
 Radjeno u pythonu3 i za programsku realizaciju SRCNN mreže odrabran je u * *tensorflow*, biblioteka koja omogućava brza matematička izračunavanja. Jedan od glavnih razloga zašto baš taj alat jeste činjenica da pruža veliku podršku mašinskom učenju.
 
  Što se same mreže tiče, implementirana je osnovna struktura 9-5-1 i Y-samo strategija, sa bazičnim parametrima, po glavnom uzoru na MATLAB kod samih autora. Trenirano je na skupu Basic, za faktor skaliranja s=3. Ispitivano je na originalnim skupovima Setu 5 i Setu 14, kao i na jednom novo-formiranom MojTest i ManjiTest, koji se nalaze u folderu Testiranje/.  Mreža je prošla kroz 15000+ epoha, sa periodičnim čuvanjem stanja na svakih 500 koraka i zaista pokazala sjajne rezultate.
  
  U folderu Eksperimenti i rezultati - rezultati testiranja, kao i uvid u moguce greske. Kod je tesko iskomentarisan, za bolje pracenje funkcija.
    
 Prikaz modula je sljedeći:
 
 **main.py** - pokretački modul, koji parsira argumente. Korisnik postavlja hiperparametre prilikom pokretanja
 ```
 $ python main.py -help
 
 usage: main.py [-h] [--br_epoha BR_EPOHA] [-checkpoint CHECK_DIR]
               [-batch BATCH] [-i_size I_SIZE] [-l_size L_SIZE]
               [-stride STRIDE] [-eta KOEF_UCENJA] [-s SCALE]
               [-sample_dir SAMPLE_DIR] [-trening TRENING]
               [-train_dir TRAIN_DIR] [-test_dir TEST_DIR] [-e UVECANJE]

Postavljanje hiperparametara mreže. Smanjivati batch, i_size i l_size sa
ubrzanje. Imati na umu da mora da vazi: i_size = l_size + 12 (ako ne postavite tako, kod ce sadm rijesiti da
zadovoljavaju te uslove. (size>0))

optional arguments:
  -h, --help            show this help message and exit
  --br_epoha BR_EPOHA   Broj epoha. Default: [2000]
  -checkpoint CHECK_DIR
                        Checkpoint dir. za cuvanje predjenih epoha
                        [checkpoint]
  -batch BATCH          Br. subslika koji se predaje SRCNN-u u jednom prolazu
                        [128]
  -i_size I_SIZE        Velicina LR segmenta (33x33) [33]
  -l_size L_SIZE        Velicina HR segmenta (21x21) [21]
  -stride STRIDE        Korak kod rezanja na subslike [14]
  -eta KOEF_UCENJA      Koef. ucenja [1e-4]
  -s SCALE              Faktor skaliranja [3]
  -sample_dir SAMPLE_DIR
                        Mjesto gdje se cuvaju rezultati [rezultati]
  -trening TRENING      True za fazu treniranje, false za test (False)
  -train_dir TRAIN_DIR  Trening-baza(mora biti smjestena u folderu Treniranje)
                        [Basic]
  -test_dir TEST_DIR    Folder/fajl za testiranje(mora biti smjesten u
                        Testiranje) [MojTest]
  -e UVECANJE           Direktno uvecanje slike [False]
```
  npr.
  ```
  $ python3 main.py -trening=True -eta=1e-5
  
  ```
  
  **trening.py** -   Za trening postaviti -trening=True, i train_dir koji se mora nalaziti u Treniranje/. Ucitavaju se, ako postoje, sacuvana stanja mreze. Prikaz prolaza kroz epohe na svakih 10 koraka. Za brzi proces smanjivati batch, i_size i l_size. Prije treninga, vrši se procesiranje podataka - slika se dijeli na sub-slike, koje imaju veličinu : i_size za LR sub-slike, l_size za ground-truth sub-slike. Ako se promijeni i_size, mora se takođe promijeniti i l_size - zbog konvolucije:  ```l_size=i_size-12 ```. batch, i_size i l_size su bitne samo kod procesa treniranja.
  
  **testiranje.py** - ukoliko nije drugačije naglašeno, mreža se testira. Njen folder za testiranje definiše se preko test\_dir, što može biti i fajl; i svi ti fajlovi treba da se nalaze u folderu Testiranje/.  Tensor-sesija se inicijalizira, čita se sačuvano stanje mreže i vrši se evaluacija ulaza. Izlaz mreže, sa malo postprocesiranja, zajedno se čuva sa slikom koja je dobijena samo bikubičnom interpolacijom. Zaista vidimo da SRCNN pobjeđuje BI metod. Sa dužim vremenom treniranja, pokazao bi još bolje performanse.  Takođe, ovaj modul omogućava da, ako je korisnik specificirao pri pozivu parametar -e=True, da skripta direktno uveća samu testnu sliku za faktor skaliranja s. Svi izlazi mreže su sačuvani u folderu rezultati/.
 
  **dodaci.py** - pomoćni modul, koji sadrži mnoge pomoćne funkcije kao što su citanje i cuvanje checkpointa, pretvaranje 3-kanalnog sistema u jedan (izvlačenje kanala Y kako bi se nad njemu vršila obrada), kao i sama gradnja SRCNN().
  
  Za razlicite faktore *s* treba se trenirati nova mreza, ali ostavljeno radi ispitivanja.
  
  **REFERENCE:**
  
  - http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
  - https://github.com/Edwardlzy/SRCNN - checkpoint fajl, pratila se i unaprijedila njegova struktura
  - https://github.com/tegg89/SRCNN-Tensorflow/
  - https://www.tensorflow.org/guide/#putting-it-all-together-example-trainer-program
  
  
 
  
  
  
