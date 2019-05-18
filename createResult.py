import cv2
import numpy as np
import os
from numdifftools import core
from keras import models

dataFolder = 'data/'
standardsFolder = 'Resized/standard/'

file = 'Resized12_h'

insert = np.load(dataFolder + file + '.npy')
standard = cv2.imread(standardsFolder + file + '.tif', cv2.IMREAD_GRAYSCALE)

resultShape = standard.shape

print(resultShape)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


print("Predicting")
results = loaded_model.predict(insert, verbose=1)
resultArray = np.empty(resultShape)
print(resultArray.shape)

print(results.shape)
print('Generating Array')
for i in range(resultShape[0]):
    for j in range(resultShape[1]):
        resultArray[i, j] = results[i * resultShape[1] + j]*255

cv2.imwrite('Result.jpg', resultArray)

