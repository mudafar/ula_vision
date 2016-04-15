# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetect2 import ShapeDetector
import argparse
import pyimagesearch.convenience as utils
import cv2
import numpy as np
import sys


def changeCannyMin(x):
    global thresh, cannyMin, cannyMax
    cannyMin = x
    cannyMax = x + 50
    #thresh = cv2.Canny(gray, cannyMin, cannyMax)
    cv2.setTrackbarMin('Canny Thresh Max', 'ULA VISION 2016', cannyMin + 1)
    #cv2.setTrackbarMax('Canny Thresh Max', 'ULA VISION 2016', cannyMax)
    cv2.setTrackbarPos('Canny Thresh Max', 'ULA VISION 2016', cannyMax)
    #thresh = cv2.threshold(blurred, cannyMin, cannyMax, cv2.THRESH_BINARY)[1]
    #cv2.imshow('ULA VISION 2016', thresh)
    process_image()


def changeCannyMax(x):
    global thresh, cannyMax
    cannyMax = x
    process_image()
    #cv2.setTrackbarPos('Canny Thresh Max', 'ULA VISION 2016', cannyMax)
    #thresh = cv2.Canny(gray, cannyMin, cannyMax)
    #thresh = cv2.threshold(blurred, cannyMin, cannyMax, cv2.THRESH_BINARY)[1]
    #cv2.imshow('ULA VISION 2016', thresh)


def changeBlur(x):
    global blurred
    blurred = cv2.GaussianBlur(gray, (blurSize, blurSize), x)
    cv2.imshow('ULA VISION 2016', blurred)


def changeBlurSize(x):
    global blurSize
    blurSize = x
    global blurred
    blurred = cv2.GaussianBlur(gray, (blurSize, blurSize), 0)
    cv2.imshow('ULA VISION 2016', blurred)


def startApp(event, x, y, flag, data):

    #if flag == cv2.EVENT_FLAG_LBUTTON:
    if event == cv2.EVENT_LBUTTONDOWN:
        #global image, thresh
        image2 = image.copy()

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]  # if imutils.is_cv2() else cnts[1]
        sd = ShapeDetector()

        # loop over the contours
        print 'Loading: ',
        sys.stdout.flush()

        for c in cnts:

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                break
            # detect the name of the
            # shape using only the contour
            shape = sd.detect(c)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            #cc = np.array(c, dtype='f')
            #cc *= ratio
            #c = np.array(cc, dtype=np.int64)
            #if shape == "?":
                #print 'unknown'
                #cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)
                #utils.setLabel(image2, '?', c)

            #elif shape == "-":
                #print 'ignored'
                #cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)

            if shape != '-':
                cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)
                utils.setLabel(image2, shape, c)
            # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # show the output image
            cv2.imshow("ULA VISION 2016", image2)
            # exit when e it is pressed, others to continue detecting
            print '*',
            sys.stdout.flush()
        print 'analysis finished'
        cv2.waitKey(0)


def process_image():
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better

    global thresh

    # resized = utils.resize(image, width=300)
    # ratio = image.shape[0] / float(resized.shape[0])
    # ratio = 1

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.GaussianBlur(gray, (blurSize, blurSize), 0)
    thresh = cv2.Canny(gray, cannyMin, cannyMax)

    cv2.imshow("ULA VISION 2016", thresh)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Ula Vision')
ap.add_argument("-i", "--image", required=False, help="path to the input image")
#args = vars(ap.parse_args())
args, leftover = ap.parse_known_args()

cannyMin = 0
cannyMax = 50
image = blank_image = np.zeros((400, 600, 3), np.uint8)
cv2.namedWindow('ULA VISION 2016')
cv2.createTrackbar('Canny Thresh Min', 'ULA VISION 2016', 0, 400, changeCannyMin)
cv2.createTrackbar('Canny Thresh Max', 'ULA VISION 2016', 0, 450, changeCannyMax)
cv2.setMouseCallback('ULA VISION 2016', startApp)

cv2.setTrackbarPos('Canny Thresh Max', 'ULA VISION 2016', cannyMax)

#cv2.createTrackbar('Blur Thresh', 'ULA VISION 2016', 0, 120, changeBlur)
#cv2.createTrackbar('Blur Size', 'ULA VISION 2016', 0, 20, changeBlurSize)

# image proveded via command line, use it
if args.image is not None:
    image = cv2.imread(args.image)
    cannyMin, cannyMax, edge = utils.auto_canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    process_image()
    cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    cannyMin, cannyMax, edge = utils.auto_canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    while True:
        ret, image = cap.read()
        process_image()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()
