import os, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros


def load_mnist(dataset="training", num = 20000, digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = num

    images = zeros((N, rows*cols+1), dtype=uint8)
    labels = zeros(N, dtype=int8)
    for i in range(num):
        feature  = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        feature.insert(0, 1)
        images[i] = array(feature)
        labels[i] = lbl[ind[i]]

    return images, labels


class Logistic:

    def __init__ (self, step, numlabel, dimension):
        self.step = step
        self.numlabel = numlabel
        self.weights = np.zeros(dimension)

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-1 * (self.weights.dot(data))))

    def train(self, traindata, label):
        Itertime = 0
        teacher = np.zeros(len(label))
        for i in range(len(label)):
            if label[i] == self.numlabel:
                teacher[i] = 1

        while Itertime < 2000:
            output = 1.0 / (1 + np.exp(-1.0 * (self.weights.dot(traindata.T) ) ) )
            self.weights = self.weights + self.step * ( (teacher - output).dot(traindata) )
            Itertime = Itertime + 1

    def predict(self, x):
        return self.sigmoid(x)


class softmax:

    def __init__(self, step, dimension, nclass, data, labels):
        self.step = step
        self.dimension = dimension
        self.nclass = nclass
        self.data = data
        self.labels = labels

        self.weights = np.zeros((nclass, dimension))
	self.teacher = np.zeros((nclass, len(labels)))
        for i in range(len(labels)):
            self.teacher[labels[i]][i] = 1


    def train(self, Round):
        Itertime = 0
        while Itertime < Round:
            print Itertime
            output = self.predict(self.data)
            self.weights = self.weights + self.step * (self.teacher - output).dot(self.data)
            Itertime = Itertime + 1  

    def predict(self, data):
        numerator = np.exp(self.weights.dot(data.T))
        denominator = np.sum(numerator, axis = 0)
	return numerator/denominator


def Lrun1():
    for i in range(10):
        model = Logistic(10e-8, i, 785)
        print "Classify number",i
        model.train(traindata, trainlabels)
        N = 0
        for index in range(len(testdata)):
            v = model.predict(testdata[index])
            if v >= 0.5 and testlabels[index] == i:
                N += 1
            if v < 0.5 and testlabels[index] != i:
                N += 1
        print "Classify number ",i," accuracy is : " , (N/2000.0)*100, "%"

def Lrun2():
    models = []
    for i in range(10):
        model = Logistic(10e-9, i, 785)
        model.train(traindata, trainlabels)
        print "trainging", i, "finished"
        models.append(model)
        N = 0

    for index in range(len(testdata)):
        data = testdata[index]
        m = -1.0
        ind = -1
        for j in range(10):
            v = models[j].predict(data)
            if v > m:
                m = v
                ind = j
        if testlabels[index] == ind:
            N = N + 1
    print "Overall Accuracy is : " , (N/2000.0)*100, "%"

def Srun1():
    for i in range(2,11):
        model = softmax(10e-9, 785, 10, traindata, trainlabels)
        model.train(100*i)
        output = model.predict(traindata)
        output = np.argmax(output, axis = 0)
        N = 0
        for j in range(len(trainlabels)):
            if output[j] == trainlabels[j]:
                N = N + 1
        print "Iteration : ", i*100, " Accuracy is : " , (N/20000.0)*100, "%"

def Srun2():

    model = softmax(10e-9, 785, 10, traindata, trainlabels)
    model.train(500)
    output = model.predict(testdata)
    output = np.argmax(output, axis = 0)
    N = 0
    for j in range(len(testlabels)):
        if output[j] == testlabels[j]:
            N = N + 1
    print " Accuracy is : " , (N*1.0/len(testlabels))*100, "%"
            
    
        

# read data
traindata, trainlabels = load_mnist('training', 20000)
testdata, testlabels = load_mnist('testing', 2000)

# run logistic
# Lrun1()
# Lrun2()

# run softmax
# Srun1()
Srun2()
