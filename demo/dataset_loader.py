import glob
import math
import os
import random
from PIL import Image


_DATASET_DIRECTORY = '256_ObjectCategories'


class DatasetLoader:

    def __init__(self, trainingSampleRate, imageDimensions, maxLabels=None,
                 preload=False):
        unloadedDataset, labelsToValues, valuesToLabels = \
            _loadDatasetImagePaths(maxLabels)
        self._labelsToValues = labelsToValues
        self._valuesToLabels = valuesToLabels

        self._unloadedTraining, self._unloadedTest = _splitDataset(
            unloadedDataset,
            trainingSampleRate)
        self._imageDimensions = imageDimensions

        self._preloaded = preload
        if preload:
            self._loadedTraining = next(DatasetIterator(
                self._unloadedTraining,
                len(self._unloadedTraining),
                self._imageDimensions))
            self._loadedTest = next(DatasetIterator(
                self._unloadedTest,
                len(self._unloadedTest),
                self._imageDimensions))

    def numLabels(self):
        return len(self._labelsToValues)

    def trainingIterator(self, batchSize):
        if self._preloaded:
            return PreloadedDatasetIterator(
                self._loadedTraining, batchSize)
        else:
            return DatasetIterator(
                self._unloadedTraining, batchSize, self._imageDimensions)

    def testData(self):
        if self._preloaded:
            return self._loadedTest
        else:
            iteratorWithWholeDatasetInOneBatch = DatasetIterator(
                self._unloadedTest,
                len(self._unloadedTest),
                self._imageDimensions)
            return next(iteratorWithWholeDatasetInOneBatch)

    def outputToLabel(self, outputVal):
        return self._valuesToLabels[outputVal]


def _loadDatasetImagePaths(maxLabels=None):
    rootPath = _DATASET_DIRECTORY
    labels = _loadDatasetLabels(rootPath)
    if maxLabels:
        labels = labels[:maxLabels]

    labelsToVals, valsToLabels = _generateLabelMapping(labels)
    inputOutputPairs = _loadDatasetInputPathLabelPairs(rootPath, labelsToVals)
    return inputOutputPairs, labelsToVals, valsToLabels


def _loadDatasetLabels(rootPath):
    isDirectory = lambda x: os.path.isdir(os.path.join(rootPath, x))
    return sorted(filter(isDirectory, os.listdir(rootPath)))


def _generateLabelMapping(labels):
    labelsToVals = {
        labels[i]: _generateLabel(i, len(labels))
        for i in range(len(labels))
    }
    valsToLabels = { i: label for label in labels for i in range(len(labels)) }
    return labelsToVals, valsToLabels


def _generateLabel(labelNum, totalLabels):
    label = [ 0.0 for i in range(totalLabels) ]
    label[labelNum] = 1.0
    return label


def _loadDatasetInputPathLabelPairs(rootPath, labelsToVals):
    labelNames = labelsToVals.keys()
    allLabelPaths = {
        labelName: [ path for path in _labelPointGlob(rootPath, labelName) ]
        for labelName in labelNames
    }
    # Flatten dictionary into a list with input/output pairs
    return [
        (path, labelsToVals[labelName])
        for labelName, paths in allLabelPaths.items()
        for path in paths
    ]


def _labelPointGlob(rootPath, label):
    return glob.glob(os.path.join(rootPath, label, '*.jpg'))


def _splitDataset(dataset, trainingSampleRate):
    random.shuffle(dataset)
    numTrainingSamples = int(trainingSampleRate * len(dataset))
    return dataset[:numTrainingSamples], dataset[numTrainingSamples:]


class PreloadedDatasetIterator:

    def __init__(self, loadedData, batchSize):
        self._loadedData = loadedData
        self._batchSize = batchSize
        self._currentBatch = 0

    def totalBatches(self):
        return int(math.ceil(len(self._loadedData) / self._batchSize))

    def __iter__(self):
        return self

    def __next__(self):
        startIndex = self._currentBatch * self._batchSize
        if startIndex > len(self._loadedData):
            raise StopIteration()

        endIndex = min(
            startIndex + self._batchSize, len(self._loadedData))

        data = self._loadedData[startIndex:endIndex]
        self._currentBatch += 1
        return data


class DatasetIterator:

    def __init__(self, unloadedData, batchSize, imageDimensions):
        self._unloadedData = unloadedData
        self._batchSize = batchSize
        self._imageDimensions = imageDimensions
        self._currentBatch = 0

    def totalBatches(self):
        return int(math.ceil(len(self._unloadedData) / self._batchSize))

    def __iter__(self):
        return self

    def __next__(self):
        startIndex = self._currentBatch * self._batchSize
        if startIndex > len(self._unloadedData):
            raise StopIteration()

        endIndex = min(
            startIndex + self._batchSize, len(self._unloadedData))

        data = self._loadBatchImages(startIndex, endIndex)
        self._currentBatch += 1
        return data

    def _loadBatchImages(self, startIndex, endIndex):
        data = self._unloadedData[startIndex:endIndex]
        inputs = [
            loadImagePixels(path, self._imageDimensions, makeGreyscale=False)
            for path, label in data
        ]
        labels = [label for path, label in data]
        return inputs, labels


def loadImagePixels(filename, imageDimensions, makeGreyscale=False):
    img = loadAndScaleImage(filename, imageDimensions, makeGreyscale)
    return flattenImage(img)


def loadAndScaleImage(filename, dimensions, makeGreyscale):
    img = Image.open(filename)
    if makeGreyscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return img.resize(dimensions)


def flattenImage(img):
    if img.mode == 'L':
        return [pixel for pixel in img.getdata()]
    elif img.mode == 'RGB':
        return [colVal for rgbTuple in img.getdata() for colVal in rgbTuple]
    else:
        raise ValueError('Unsupported image mode: {}'.format(img.mode))
