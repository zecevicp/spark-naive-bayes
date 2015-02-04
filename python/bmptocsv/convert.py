from PIL import Image
import glob
import os
import sys

def convert_image(dirname, name):
    '''Converts BMP image dirname/name.bmp into a line of comma-separated
    zeros and ones: zeros for white pixels and ones for others and writes
    the line to stdout.
    BMP image will first be scaled to the size of 20 by 20 pixels and converted to
    black and white.
    '''
    print "\nconverting "+name
    image = Image.open(dirname+"/"+name+".bmp")
    #resize image to reduce dimensions
    image = image.resize((20, 20), Image.NEAREST)
    #convert to black and white
    image = image.convert('1')
    #get the pixels
    pix = image.load()
    first = True
    for x in list(image.getdata()):
        if not first:
            sys.stdout.write(",")
        first = False
        if x == 255:
            sys.stdout.write("0")
        else:
            sys.stdout.write("1")
    sys.stdout.write("\n")

def convert_all(dirname, namematch, classvalue):
    '''Converts all BMP images in directory dirname, whose names start with namematch,
    to a CSV file dirname/total.txt (it will be appended).
    Output file will contain one line for each image. Each line will contain classvalue
    as the first element and 0s for white pixels and 1s for others.
    BMP images will be scaled to the size of 20 by 20 pixels.
    '''
    files = glob.glob(dirname+"/"+namematch+"*.bmp")
    total = open(dirname+"/total.txt", "a")
    for imgfile in files:
        name = os.path.basename(imgfile)
        print "\nconverting "+name
        image = Image.open(imgfile)
        #resize image to reduce dimensions
        image = image.resize((20, 20), Image.NEAREST)
        #convert to black and white
        image = image.convert('1')
        #get the pixels
        pix = image.load()

        total.write(str(classvalue))
        for x in list(image.getdata()):
            if x == 255:
                total.write(",0")
            else:
                total.write(",1")
        total.write("\n")
    total.close()
