import shutil

import cv2
import os
import numpy as np

dataFolder = 'data/'
healthyFolder = 'Resized/healthy/'
resultsFolder = 'Resized/results/'
FoVFolder = 'Resized/FoV/'
standardsFolder = 'Resized/standard/'
gaborsNumber = 16

claheFilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gaborfilters = []

for i in range(gaborsNumber):
    gaborfilters.append(cv2.getGaborKernel((6, 6), 20, i*(180/16), 60, 0))

def normalizeMatrix(matrix):
    data = matrix.astype(float)/255
    return data


def calculateGradient(matrix):
    kernely = np.array(
        [[1, 1, 1],
         [0, 0, 0],
         [-1, -1, -1]
        ])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    dx = cv2.filter2D(matrix, cv2.CV_32F, kernelx)
    dy = cv2.filter2D(matrix, cv2.CV_32F, kernely)
    return dx, dy


def calculateEigenValuesOfHessian(hessian):
    print('Calculating EigenValues:')
    z2, z1, y, x = hessian.shape
    values = []
    for i in range(y):
        if i % 50 == 0:
            print(str(i)+'/'+str(y))
        valuesRow = []
        for j in range(x):
            valuesRow.append(np.linalg.eigvals(hessian[:, :, i, j]))
        values.append(valuesRow)
    return np.asarray(values)


def createImgData(file):
    print('Loading: ', file)
    data = []
    image = cv2.imread(healthyFolder + file + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    # standards = cv2.imread(standardsFolder + file + '.tif', cv2.IMREAD_GRAYSCALE)
    FoV = cv2.imread(FoVFolder + file + '_mask.tif', cv2.IMREAD_GRAYSCALE)
    clahe = claheFilter.apply(g)
    gabors = []
    enhancedClahe = None

    for i, GaborFilter in enumerate(gaborfilters):
        print('Applying filter: ', i)
        gabor = cv2.filter2D(clahe, cv2.CV_32F, GaborFilter)
        if i == 0:
            enhancedClahe = gabor
        else:
            enhancedClahe = cv2.addWeighted(enhancedClahe, 1, gabor, 1, 0)
        gabors.append(normalizeMatrix(gabor))

    enhancedClahe = cv2.addWeighted(enhancedClahe, 1, clahe.astype(np.float32), 1, 0)

    grayDx, grayDy = calculateGradient(gray)
    grayDxDx, grayDxDy = calculateGradient(grayDx)
    grayDyDx, grayDyDy = calculateGradient(grayDy)
    grayHessian = np.asarray([[grayDxDx, grayDxDy], [grayDyDx, grayDyDy]])
    eigenValues = calculateEigenValuesOfHessian(grayHessian)


    print('Concatenating data for ' + file)
    data.append(normalizeMatrix(b))
    data.append(normalizeMatrix(g))
    data.append(normalizeMatrix(r))
    data.append(normalizeMatrix(gray))
    data.append(normalizeMatrix(clahe))
    data.append(normalizeMatrix(FoV))
    data.append(normalizeMatrix(enhancedClahe))
    data.append(normalizeMatrix(np.sqrt((grayDx * grayDx + grayDy * grayDy))))
    data.append(eigenValues[:, :, 0])
    data.append(eigenValues[:, :, 1])
    data = np.array(data)
    data = np.concatenate((data, gabors), 0)

    print(data.shape)

    return data


def tranformImgDataToInput(data):
    print('Transforming data!')
    newData = []
    z, y, x = data.shape

    for i in range(y):
        for j in range(x):
            newData.append(data[:, i, j])
    newData = np.asarray(newData)
    return newData


def __init__():
    if os.path.isdir(dataFolder):
        shutil.rmtree(dataFolder)
    os.mkdir(dataFolder)

    for file in os.listdir(healthyFolder):
        fileName = file.split('.')[0]
        data = createImgData(fileName)
        data = tranformImgDataToInput(data)
        np.save(dataFolder + fileName, data)

__init__()

