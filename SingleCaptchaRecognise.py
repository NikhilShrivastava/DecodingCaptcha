from __future__ import print_function
import cv2

from PIL import Image, ImageStat
import numpy as np
from scipy.ndimage.filters import median_filter
import os, glob

from scipy.misc import imsave
from scipy import ndimage
from selenium import webdriver

from singleCSV import gray_to_csv
from recognise import recognize_single

captcha_list = []
driver = webdriver.Chrome()

def download_captch():

    def get_captcha(driver, element, path):
        location = element.location
        size = element.size
        driver.save_screenshot(path)
        image = Image.open(path)


        left = location['x']
        top = location['y'] - 18
        right = location['x'] + size['width']
        bottom = location['y'] + size['height'] -19

        image = image.crop((left, top, right, bottom))  # defines crop points
        image.save(path, 'png')  # saves new cropped image

    driver.maximize_window()
    driver.get("http://www.snaphost.com/captcha/Demo.htm")
    driver.implicitly_wait(30)

    # driver.get('http://www.freelibros.org/wp-login.php')
    # img = driver.find_element_by_xpath('html/body/div[1]/form/p[3]/label/img')

    #img = driver.find_element_by_xpath('html/body/form/table[2]/tr[2]/td[2]/img')
    img = driver.find_element_by_id('CaptchaImage')

    get_captcha(driver, img, "captcha.png")
    #driver.close()


    img_orginal = Image.open("captcha.png")

    img_crop = img_orginal.crop((25, 20, 180, 65))

    img_crop.save("crop.png")

    img_crop_read = cv2.imread("crop.png")

    img_gray = cv2.cvtColor( img_crop_read, cv2.COLOR_RGB2GRAY )

    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_median = median_filter(img_bw, 5)

    img_dilation = ndimage.grey_dilation(img_median, size=(3, 3))

    cv2.imwrite("dilation.png", img_dilation)

    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    ekernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    im = cv2.imread("dilation.png")
    im = 255-im
    img2gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("dil-grayed.png", img2gray)

    img_bnw = cv2.imread("dil-grayed.png")
    # cv2.imshow("bnw",img_bnw)

    dilate = cv2.dilate(img_bnw, kernel)
    erosion = cv2.erode(dilate, ekernel)

    erosion_gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("erosed.png", erosion_gray)


def captch_ex(original_file):

    my_original =cv2.imread(original_file)
    # cv2.imshow("Current Image is ",my_original)
    # cv2.waitKey(10000)
    img2gray = cv2.cvtColor(my_original, cv2.COLOR_BGR2GRAY)

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
        # cv2.imshow("Here it",cropped_to_gray)
        # cv2.waitKey(1000)



        # cv2.imshow("Each Cropped",cropped_to_gray)
        # cv2.waitKey(100000)


        s = 'image' + str(index)+ '.png'

        #s = "./test-/" + file.split(".")[0] + ".png"
        #s = ("./test-/%s.png" % (str(index)))
        cv2.imwrite(s, cropped_to_gray)
        gray_to_csv(s)
        single_digit = recognize_single('single.csv')
        captcha_list.append(single_digit)
        index = index + 1

        ###################





download_captch()
original_file = "dil-grayed.png"
captch_ex(original_file)
print (captcha_list)
str_cap = ''.join(captcha_list)
print (str_cap)

fill_field = driver.find_element_by_id("skip_CaptchaCode")
fill_field.send_keys(str_cap)

submit_field = driver.find_element_by_id("skip_DemoSubmit")
submit_field.click()

confirm_element = driver.find_element_by_xpath("//*[@id='form1']/div[3]")


