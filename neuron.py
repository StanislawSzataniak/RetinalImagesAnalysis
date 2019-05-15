import cv2
import numpy as np
from scipy import optimize
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

neighbourhoodSize = 40
inputSize = (2*neighbourhoodSize + 1)**2 * 2 + 1
numberOfProbes = 5000
healthyFolder = 'healthy/'
resultsFolder = 'results/'
FoVFolder = 'FoV/'
standardsFolder = 'standards/'


class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = inputSize
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

# Propagate inputs though network

def normalizeMatrix(matrix):
    matrix -= np.amin(matrix)
    matrix /= np.amax(matrix)

def getFromMatrix(x,y, matrix):
    if (x < 0 or x >= matrix.shape[0] or y < 0 or y >= matrix.shape[1]):
        return 0.0
    return matrix[x, y]

def findNeighbours(x,y, matrix):
    neighbourhoodArray = np.zeros((neighbourhoodSize*2 + 1, neighbourhoodSize*2 + 1))

    for i in range(len(neighbourhoodArray)):
        for j in range(len(neighbourhoodArray[i])):
            neighbourhoodArray[i, j] = getFromMatrix(x - neighbourhoodSize + i, y - neighbourhoodSize + j, matrix)
    return neighbourhoodArray

def calculateParams(x,y, gradientMagnitude, maskValue, img):
    imgNeighbours = findNeighbours(x, y, img)
    gradientNeighbours = findNeighbours(x, y, gradientMagnitude)

    imgNeighboursFlatten = np.ndarray.flatten(imgNeighbours)
    gradientNeighboursFlatten = np.ndarray.flatten(gradientNeighbours)
    a = np.concatenate(([maskValue], imgNeighboursFlatten, gradientNeighboursFlatten))
    return a

def determineParams(points, gradientMagnitude, mask, img):
    X = np.empty((numberOfProbes, inputSize))
    for i, point in enumerate(points):
        X[i] = calculateParams(x, y, gradientMagnitude, maskValues[i], img)
    return X

def loadImg(file):
    image = cv2.imread(healthyFolder + file + '.jpg')
    standard = cv2.imread(standardsFolder + file + '.tif', cv2.IMREAD_GRAYSCALE )
    FoV = cv2.imread(FoVFolder + file + '_mask.tif', cv2.IMREAD_GRAYSCALE )
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1)
    gradientMagnitude = np.sqrt((gx * gx + gy * gy))

    enhanced = enhanced.astype(float)
    FoV = FoV.astype(float)
    standard = standard.astype(float)
    normalizeMatrix(enhanced)
    normalizeMatrix(gradientMagnitude)
    normalizeMatrix(standard)
    normalizeMatrix(FoV)

    return enhanced, gradientMagnitude, standard, FoV


img, gradientMagnitude, standard, mask = loadImg('10_h')

points = []
expectedResults = []
maskValues = []

for i in range(numberOfProbes):
    x = np.random.randint(img.shape[0])
    y = np.random.randint(img.shape[1])
    points.append((x, y))
    expectedResults.append([standard[x, y]])
    maskValues.append(mask[x, y])

params = determineParams(points, gradientMagnitude, maskValues, img)

NN = Neural_Network()
T = Trainer(NN)
T.train(params, expectedResults)