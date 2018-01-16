import os, sys
import PIL.Image as im

import wget
import zipfile

size = 128, 128
baseurl = "http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/zipped images/"

for i in range(10):
    if i <= 3:
        continue
    #if i==9:
    #    myurl = baseurl + "10_images.zip"
    #else:
    #    myurl = baseurl + "0" + str(i+1) + "_images.zip"
    myurl = baseurl + "part" + str(i+1) + ".zip"
    filename = wget.download(myurl)
    archive = zipfile.ZipFile(filename)
    noms_fichiers = archive.namelist()

    for nom in noms_fichiers:
        print(nom)
        archive.extract(nom)
        imag = im.open(nom)
        imag.thumbnail(size, im.ANTIALIAS)
        imag.save(nom, "JPEG")
    os.remove(filename)
