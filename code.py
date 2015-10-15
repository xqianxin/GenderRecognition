import cv2 as cv
from numpy import *
import csv
import os

ABS_PATH = "./image"

def readCSVFile(filePath):
    images = []
    labels = []
    myfile = open(filePath).readlines()
    try:
        for row in myfile:
            val = row.strip().split(";")
            images.append(val[0])
            labels.append(int(val[1]))
    finally:
        pass
    print 'in read csv file, (imgs,labels): ',(images,labels)    
    return (images, labels)

class GenderRecognizer:

    def __init__(self):
        print("Init")

    def readImage(self, imagePath):
        return cv.imread(imagePath, cv.CV_LOAD_IMAGE_GRAYSCALE)

    def getData(self):
        print("getData")
        data = readCSVFile(ABS_PATH + "/" + "gender.csv")
        images = []
        for image in data[0]:
            matImg = self.readImage(image)
            print 'reading: ', (matImg)
            images.append(matImg)
        labels = data[1]
        print 'in det date, (imgs, labels):',(images, labels)
        return(images, labels)

    def trainingModel(self):
        print("training")
        data = self.getData()
        print(data)
        self.model = cv.createFisherFaceRecognizer()
        self.model.train(array(data[0]), array(data[1]))
        print("Training over")

    def getGender(self, testSample):
        print("getGender")
        predictedLabel = self.model.predict(testSample)
        print(predictedLabel)
        return predictedLabel
