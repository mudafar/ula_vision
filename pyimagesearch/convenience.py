# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com

# import the necessary packages
import numpy as np
import cv2
import sys
import json
import random

def load_json_dic():
    return json.load(open("db.json", "r"))

def dump_json_dic(dic):
    json.dump(dic, open("db.json", "wb"))


def init_dic():
    dic = {"cannyMin": 0, "cannyMax": 50,
           "contourArea": 100, "isContourConvex": 1, "4_minarcos": -0.1, "4_maxarcos": 0.3,
           "5_minarcos": -0.50, "5_maxarcos": -0.1,
           "6_minarcos": -0.62, "6_maxarcos": -0.4,
           "rect_p": 50, "pent_p": 50, "hex_p": 50, "cir_p": 50,
           "w_h_relation": 0.2, "area_relation": 0.2
           }
    json.dump(dic, open("db.json", "wb"))

    #das = json.load(open("db.json", "r"))
    #print das['contourArea']

def load_from_json(key):
    das = json.load(open("db.json", "r"))
    return das[key]

def update_into_json(key, new_value):
    dic = load_json_dic()
    dic[key] = new_value
    dump_json_dic(dic)

def has_percentage(str):
    if '%' in str:
        return True
    else:
        return False


def parase_figure(str, good):
    dic = load_json_dic()

    per = has_percentage(str)
    if "RECTANGULO" in str:
        if good:
            dic["4_minarcos"] -= random.uniform(0.01, 0.03)
            dic["4_maxarcos"] += random.uniform(0.01, 0.03)
            if per:
                dic["rect_p"] += 5
                if dic["rect_p"] > 100:
                    dic["rect_p"] = 100

        else:
            dic["4_minarcos"] += 0.02
            dic["4_maxarcos"] -= 0.02
            if per:
                dic["rect_p"] -= 5
                if dic["rect_p"] < 10:
                    dic["rect_p"] = 10


    elif "PENTAGONO" in str:
        if good:
            dic["5_minarcos"] -= 0.02
            dic["5_maxarcos"] += 0.02
            if per:
                dic["pent_p"] += 5
                if dic["pent_p"] > 100:
                    dic["pent_p"] = 100

        else:
            dic["5_minarcos"] += 0.02
            dic["5_maxarcos"] -= 0.02
            if per:
                dic["pent_p"] -= 5
                if dic["pent_p"] < 10:
                    dic["pent_p"] = 10


    elif "HEXAGONO" in str:
        if good:
            dic["6_minarcos"] -= 0.02
            dic["6_maxarcos"] += 0.02
            if per:
                dic["hex_p"] += 5
                if dic["hex_p"] > 100:
                    dic["hex_p"] = 100

        else:
            dic["6_minarcos"] += 0.02
            dic["6_maxarcos"] -= 0.02

            if per:
                dic["pent_p"] -= 5
                if dic["hex_p"] < 10:
                    dic["hex_p"] = 10

    elif "CIRCULO" in str:
        if good:
            dic["area_relation"] += 0.02
            dic["w_h_relation"] += 0.02
            if per:
                dic["cir_p"] += 5
                if dic["cir_p"] > 100:
                    dic["cir_p"] = 100

        else:
            dic["area_relation"] -= 0.02
            dic["w_h_relation"] -= 0.02
            if per:
                dic["cir_p"] -= 5
                if dic["cir_p"] < 10:
                    dic["cir_p"] = 10


    dump_json_dic(dic)



def setLabel(im, label, contour):

    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    # baseline = 0

    sz, baseline = cv2.getTextSize(label, fontface, scale, thickness)
    x, y, w, h = cv2.boundingRect(contour)

    pt = (x + ((w - sz[0]) / 2), y + ((h + sz[1]) / 2))
    im = cv2.rectangle(im, (pt[0], pt[1] + baseline), (pt[0] + sz[0], pt[1] - sz[1]), (255, 255, 255), cv2.FILLED)
    cv2.putText(im, label, pt, fontface, scale, (0, 0, 0), thickness, 8)


def setLabelUnder(im, label):

    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    # baseline = 0

    sz, baseline = cv2.getTextSize(label, fontface, scale, thickness)

    height, width, channels = im.shape

    pt = ((width - sz[0]) / 2, (height - sz[1]) )
    im = cv2.rectangle(im, (pt[0], pt[1] + baseline), (pt[0] + sz[0], pt[1] - sz[1]), (255, 255, 255), cv2.FILLED)
    cv2.putText(im, label, pt, fontface, scale, (0, 0, 0), thickness, 8)



def angle(pt1, pt2, pt0):
    dx1 = pt1[0, 0] - pt0[0, 0]
    dy1 = pt1[0, 1] - pt0[0, 1]
    dx2 = pt2[0, 0] - pt0[0, 0]
    dy2 = pt2[0, 1] - pt0[0, 1]
    # print dx1, dx2, dy1, dy2
    aux = (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
    return aux


def cleanVertex(approx, tol=3):
    todel = []

    for index1, it in enumerate(approx):
        it = it[0]

        for index2, it2 in enumerate(approx):
            it2 = it2[0]
            if index1 != index2:
                if (np.abs(it[0] - it2[0]) <= tol) and (np.abs(it[1] - it2[1]) <= tol):
                    it2[0] = 0
                    it2[1] = 0
                    todel.append(index2)

    aux = len(approx)
    # print 'Of a: ', aux
    cleaned_approx = np.delete(approx, todel, axis=0)
    # print 'Cleaned :', aux - len(cleaned_approx)
    return cleaned_approx


# import any special Python 2.7 packages
if sys.version_info.major == 2:
    from urllib import urlopen

# import any special Python 3 packages
elif sys.version_info.major == 3:
    from urllib.request import urlopen

def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def skeletonize(image, size, structuring=cv2.MORPH_RECT):
    # determine the area (i.e. total number of pixels in the image),
    # initialize the output skeletonized image, and construct the
    # morphological structuring element
    area = image.shape[0] * image.shape[1]
    skeleton = np.zeros(image.shape, dtype="uint8")
    elem = cv2.getStructuringElement(structuring, size)

    # keep looping until the erosions remove all pixels from the
    # image
    while True:
        # erode and dilate the image using the structuring element
        eroded = cv2.erode(image, elem)
        temp = cv2.dilate(eroded, elem)

        # subtract the temporary image from the original, eroded
        # image, then take the bitwise 'or' between the skeleton
        # and the temporary image
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        # if there are no more 'white' pixels in the image, then
        # break from the loop
        if area == area - cv2.countNonZero(image):
            break

    # return the skeletonized image
    return skeleton

def opencv2matplotlib(image):
    # OpenCV represents images in BGR order; however, Matplotlib
    # expects the image in RGB order, so simply convert from BGR
    # to RGB and return
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return lower, upper, edged

def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
        
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)


