from __future__ import print_function
import cv2

from PIL import Image
import numpy as np
from dill import dill
from scipy.ndimage.filters import median_filter
import os

from scipy import ndimage

with open('nn.dill', 'rb') as f: # load the trained Neural Network
    nn = dill.load(f)

char_number_map = {0:'0', 1:'1', 2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
                   16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',
                   31:'V',32:'W',33:'X',34:'Y',35:'Z'}


files = os.listdir("./testcaptcha")
for file in files:
    img_orginal = Image.open("./testcaptcha/" + file)

    #cropping image and converting to RGB
    img_crop = img_orginal.crop((25, 0, 180, 45))

    img_crop.save("./testcapt-cropped/" + file.split(".")[0] + ".png")


files == os.listdir("./testcapt-cropped")
for file in files:
    img_crop_read = cv2.imread("./testcapt-cropped/" + file)
    #print img_crop.shape

    img_gray = cv2.cvtColor( img_crop_read, cv2.COLOR_RGB2GRAY )
    #cv2.imwrite( "gray.png", img_gray )
    #cv2.imshow( "gray.jpg", img_gray )


    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #print thresh
    # thresh = 205
    # img_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('bnw',img_bw)
    #cv2.imwrite("bnw.png", img_bw)


    # #median filter

    img_median = median_filter(img_bw, 5)
    #cv2.imwrite("median.png", img_median)
    #cv2.imshow('median.png',img_median)


    img_dilation = ndimage.grey_dilation(img_median, size=(3,3))

    # #imsave("dilation1.gif", img_dilation)
    #cv2.imshow("dilation1.jpg", img_dilation)
    cv2.imwrite("./testcapt-dilated/" + file.split(".")[0] + ".png", img_dilation)

kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
ekernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)

files == os.listdir("./testcapt-dilated")
for file in files:
    im = cv2.imread("./testcapt-dilated/" + file)
    im = 255-im
    img2gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./testcapt-dil-grayed/" + file.split(".")[0] + ".png", img2gray)

files == os.listdir("./testcapt-dil-grayed")
for file in files:
    img_bnw = cv2.imread("./testcapt-dil-grayed/" + file)
    # cv2.imshow("bnw",img_bnw)

    dilate = cv2.dilate(img_bnw, kernel)
    erosion = cv2.erode(dilate, ekernel)

    erosion_gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./testcapt-erosed/" + file.split(".")[0] + ".png", erosion_gray)


def captch_ex(original_file,total_file):

    my_original = original_file
    img2gray = cv2.cvtColor(original_file, cv2.COLOR_BGR2GRAY)

    # img2gray = cv2.cvtColor(file_name, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(img2gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


    index = 0
    our_contours= []


    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        our_contours.append([x,y,w,h])

    #print (our_contours)
    our_contours.sort()
    #print(our_contours)


    for contour in our_contours:
        # get rectangle bounding contour
        [x, y, w, h] = contour

        # Don't plot small false positives that aren't text
        if w < 20 and h < 20:
            continue

        # cv2.imshow("Original",original_file)
        # cv2.waitKey(100000)


        #you can crop image and send to OCR,false detected will return no text
        cropped = my_original[y:y +  h , x : x + w]
        res = cv2.resize(cropped,(28,28), interpolation=cv2.INTER_AREA)
        cropped_to_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Each Cropped",cropped_to_gray)
        # cv2.waitKey(100000)


        s = './captcha-test/' + str(total_file)+'__' + str(index)+ '.png'
        #s = "./test-/" + file.split(".")[0] + ".png"
        #s = ("./test-/%s.png" % (str(index)))
        cv2.imwrite(s, cropped_to_gray)
        index = index + 1

        ###################


total_file = 200
files == os.listdir("./testcapt-dil-grayed")
for file in files:
    original_file = cv2.imread("./testcapt-dil-grayed/" + file)

    # cv2.imshow('image',original_file)
    # cv2.waitKey(1000000)
    # exit(0)
    captch_ex(original_file,total_file)
    total_file = total_file + 1
print (total_file)



