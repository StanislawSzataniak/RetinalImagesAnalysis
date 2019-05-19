import cv2
import numpy as np
import os
from numdifftools import core
from keras import models

dataFolder = 'data/'
standardsFolder = 'Resized/standard/'
weightsFolder = 'NN-weights/'
resultsFolder = 'NN-results/'
resultShape = cv2.imread(standardsFolder + 'Resized01_h.tif', cv2.IMREAD_GRAYSCALE).shape
print(resultShape)

if not(os.path.isdir(resultsFolder)):
    os.mkdir(resultsFolder)

def loadData():
    data = []
    names = []
    for file in os.listdir(dataFolder):
        print('Loading: ' + file)
        names.append(file.split('.')[0])
        matrixFromFile = np.load(dataFolder + file)
        data.append(matrixFromFile)

    return np.asarray(data), names

def saveResults(directory, file, prediction):
    resultArray = np.empty(resultShape)
    resultArrayNoised = np.empty(resultShape)

    print(prediction.shape)
    print('Generating Array for ' + file)
    for i in range(resultShape[0]):
        for j in range(resultShape[1]):
            resultArrayNoised[i, j] = prediction[i * resultShape[1] + j] * 255
            resultArray[i, j] = 0 if prediction[i * resultShape[1] + j] < .65 else 255

    if not(os.path.isdir(resultsFolder + directory)):
        os.mkdir(resultsFolder + directory)

    cv2.imwrite(resultsFolder + directory + '/' + file + '.jpg', resultArray)
    cv2.imwrite(resultsFolder + directory + '/' + file + 'Noised.jpg', resultArrayNoised)


dataMatrix, names = loadData()

print('Loading model')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = models.model_from_json(loaded_model_json)

for directory in os.listdir(weightsFolder):
    model.load_weights(weightsFolder + directory + "/weights.hdf5")

    print("Loaded model from disk - " + directory)

    for insert, name in zip(dataMatrix, names):
        results = model.predict(insert, verbose=0)
        saveResults(directory, name, results)