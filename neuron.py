import cv2
import numpy as np
import os
from numdifftools import core
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense

dataFolder = 'data/'
standardsFolder = 'Resized/standard/'

def loadData():
    data = []
    for file in os.listdir(dataFolder):
        print('Loading: ' + file)
        matrixFromFile = np.load(dataFolder + file)
        data.append(matrixFromFile)

    return np.asarray(data)

def loadStandards():
    standards = []
    for file in os.listdir(standardsFolder):
        if file.split('.')[1] != 'tif':
            break
        print('Loading: ' + file)
        img = cv2.imread(standardsFolder + file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        standards.append(np.ndarray.flatten(img)/255)

    return np.asarray(standards)

print('Load Data')
dataMatrix = loadData()
print('Load Standards')
standards = loadStandards()
print('Loading Finished')
print('Data: ' + str(dataMatrix.shape))
print('Standarts: ' + str(standards.shape))

print('Dividing data: ')
training_x = np.concatenate((dataMatrix[:3, :, :], dataMatrix[6:, :, :]), axis=0)
training_x = np.concatenate(training_x, axis=0)
testing_x = np.concatenate(dataMatrix[3:6, :, :], axis=0)
training_y = np.concatenate((standards[:3], standards[6:]), axis=0)
training_y = np.concatenate(training_y, axis=0)
testing_y = np.concatenate(standards[3:6], axis=0)

print('Training:')
print(training_x.shape)
print(training_y.shape)
print('Testing:')
print(testing_x.shape)
print(testing_y.shape)

filepath = "weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = Sequential()
model.add(Dense(256, input_dim=training_x.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # budujemy model! ustawiamy funkcję kosztu - mamy klasyfikację z dwiema etykietami, więc stosujemy 'binary_crossentropy'
              optimizer='Adam',  # wybieramy w jaki sposób sieć ma się uczyć
              metrics=['accuracy']) # i wybieramy jaka miara oceny nas interesuje

with open("model.json", "w") as json_file:
    json_file.write(model.to_json())

model.fit(training_x, training_y, epochs=20, batch_size=100, callbacks=callbacks_list, verbose=2)

# serialize weights to HDF5
model.save_weights("weights.hdf5")
print("Saved model to disk")

loss, accuracy = model.evaluate(testing_x, testing_y)
print("Trafność klasyfikacji to: {acc}%".format(acc=accuracy*100))