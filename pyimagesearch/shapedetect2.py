# import the necessary packages
import cv2
import numpy as np
import pyimagesearch.convenience as utils


class ShapeDetector:
    def __init__(self):
        # read data from json file and load it
        self.dic = utils.load_json_dic()
        #pass

    def update_data(self, data):
        self.dic = data

    def detect(self, c):
        shape = '?'

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * 0.04, True)
        edges = len(approx)

        # cleanVertex(approx)

        if np.fabs(cv2.contourArea(c)) < 100:
            return '-'
        elif not cv2.isContourConvex(approx):
            return ''

        #print len(approx)

        if edges == 3:
            # cv2.drawContours(dst, [approx], -1, (0, 255, 0), 3)
            shape = "TRIANGULO"

        elif (edges >= 4) and (edges <= 6):
            vtc = edges
            arcos = []

            if edges <= 6:
                for j in range(2, vtc + 1, 1):
                    # print j % vtc, 'module '
                    arcos.append(utils.angle(approx[j % vtc], approx[j - 2], approx[j - 1]))

            arcos.sort()
            minarcos = arcos[0]
            maxarcos = arcos[-1]

            if vtc == 4:
                #           <= 95.7              >= 72.5
                #if minarcos >= -0.1 and maxarcos <= 0.3:
                if minarcos >= self.dic["4_minarcos"] and maxarcos <= self.dic["4_maxarcos"]:
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = w / float(h)

                    shape = "CUADRADO" if (ar >= 0.95) and (ar <= 1.05) else "RECTANGULO"
                elif minarcos >= self.dic["4_minarcos"] or maxarcos <= self.dic["4_maxarcos"]:
                    shape = "RECTANGULO " + str(self.dic["rect_p"]) + "%"
                #else:
                #    shape = "rectangle 10%"


            elif vtc == 5:
                #           <= 120                >= 95.7
                #if minarcos >= -0.50 and maxarcos <= -0.1:
                if minarcos >= self.dic["5_minarcos"] and maxarcos <= self.dic["5_maxarcos"]:

                    shape = "PENTAGONO"
                elif minarcos >= self.dic["5_minarcos"] or maxarcos <= self.dic["5_maxarcos"]:
                    shape = "PENTAGONO " + str(self.dic["pent_p"]) + "%"
                #else:
                #    shape = "PENTA 10%"
            elif vtc == 6:
                #print ('mincos: ', minarcos ,'maxcos: ', maxarcos)
                #           <= 128.3              >= 113.5
                #if minarcos >= -0.62 and maxarcos <= -0.4:
                if minarcos >= self.dic["6_minarcos"] and maxarcos <= self.dic["6_maxarcos"]:

                    shape = "HEXAGONO"
                elif minarcos >= self.dic["6_minarcos"] or maxarcos <= self.dic["6_maxarcos"]:
                    shape = "HEXAGONO " + str(self.dic["hex_p"]) + "%"
                #else:
                #    shape = "HEXA 10%"

        if (edges >= 4) and (shape == '?'):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            radius = int
            radius = w / 2

            w_h_relation = abs(1 - (float(w) / h))
            area_relation = abs(1 - (area / (np.pi * pow(radius, 2))))


            if w_h_relation <= self.dic["w_h_relation"] and area_relation <= self.dic["area_relation"]:
                shape = "CIRCULO"
            elif w_h_relation <= self.dic["w_h_relation"] or area_relation <= self.dic["area_relation"]:
                    shape = "CIRCULO " + str(self.dic["cir_p"]) + "%"
            else:
                shape = "CIRCULO 20 %"
                # cv2.drawContours(dst, approx, -1, (0, 255, 0), 3)
        # print 'length approx: ', edges
        return shape



