# import the necessary packages
import cv2
import numpy as np
from pyimagesearch.convenience import angle, cleanVertex


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = '?'

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * 0.04, True)


        # cleanVertex(approx)

        if (np.fabs(cv2.contourArea(c)) < 100) or (not cv2.isContourConvex(approx)):
            return '-'

        #print len(approx)

        if len(approx) == 3:
            # cv2.drawContours(dst, [approx], -1, (0, 255, 0), 3)
            shape = "TRIANGULO"

        elif (len(approx) >= 4) and (len(approx) <= 6):
            vtc = len(approx)
            cos = []
            for j in range(2, vtc + 1, 1):
                # print j % vtc, 'module '
                cos.append(angle(approx[j % vtc], approx[j - 2], approx[j - 1]))

            cos.sort()
            mincos = cos[0]
            maxcos = cos[-1]

            if vtc == 4:
                if mincos >= -0.1 and maxcos <= 0.3:
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = w / float(h)

                    shape = "CUADRADO" if (ar >= 0.95) and (ar <= 1.05) else "RECTANGULO"
                elif mincos >= -0.1 or maxcos <= 0.3:
                    shape = "RECTANGULO 40%"
                #else:
                #    shape = "rectangle 10%"


            elif vtc == 5:
                if mincos >= -0.50 and maxcos <= -0.15:
                    shape = "PENTAGONO"
                elif mincos >= -0.50 or maxcos <= -0.15:
                    shape = "PENTAGONO 40 %"
                #else:
                #    shape = "PENTA 10%"
            elif vtc == 6:
                if mincos >= -0.55 and maxcos <= -0.45:
                    shape = "HEXAGONO"
                if mincos >= -0.55 or maxcos <= -0.45:
                    shape = "HEXAGONO 40 %"
                #else:
                #    shape = "HEXA 10%"

        if (len(approx) >= 4) and (shape == '?'):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            radius = int
            radius = w / 2

            if abs(1 - (float(w) / h)) <= 0.2 and abs(1 - (area / (np.pi * pow(radius, 2)))) <= 0.2:
                shape = "CIRCULO"
            elif abs(1 - (float(w) / h)) <= 0.2 or abs(1 - (area / (np.pi * pow(radius, 2)))) <= 0.2:
                shape = "CIRCULO 40%"
            else:
                shape = "CIRCULO 20%"
                # cv2.drawContours(dst, approx, -1, (0, 255, 0), 3)
        # print 'length approx: ', len(approx)
        return shape



