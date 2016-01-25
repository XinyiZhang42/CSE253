import os, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from scipy import stats

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

    images = zeros((N, rows*cols+1), dtype=float)
    labels = zeros(N, dtype=int8)
    for i in range(num):
        feature  = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        feature = stats.zscore(feature)
        features = zeros(rows*cols+1)
        features[0] = 1
        features[1:] = feature[:]
        images[i] = array(features)
        labels[i] = lbl[ind[i]]

    return images, labels

class NN:

    def __init__(self, step, dimension, nclass, data, labels):
        self.step = step
        self.dimension = dimension
        self.nclass = nclass
        self.data = data
        self.labels = labels

        self.teacher = np.zeros((nclass, len(labels)))
        for i in range(len(labels)):
            self.teacher[labels[i]][i] = 1

        self.nlayer1 = 50;
        self.nlayer2 = 50;

        #self.weights0 = np.zeros((self.nlayer1, dimension))
        #self.weights1 = np.zeros((nclass, self.nlayer1+1))
        self.weights0 = np.random.normal(0, 0.1, size = (self.nlayer1, dimension))
        self.weights1 = np.random.normal(0, 0.1, size = (self.nlayer2, self.nlayer1+1))
        self.weights2 = np.random.normal(0, 0.1, size = (self.nclass, self.nlayer2+1))

        self.pweights0 = np.zeros((self.nlayer1, dimension))
        self.pweights1 = np.zeros((nclass, self.nlayer1+1))

    def train(self, iterations, _lambda = 0.001):

        racks = 30
        count = 0

        while count < iterations:

            for l in range(racks):

                leng = len(self.labels)
                piece = leng/racks
                start = l*piece
                end = start + piece
                #print len(self.teacher)
                data = self.data[start:end]
                teacher = self.teacher[:, start:end]

                z1 = self.sigmoid( (self.weights0).dot(data.transpose()) )
                #z1 = np.tanh( (self.weights0).dot(data.transpose()) )
                #z1 = (self.weights0).dot(data.transpose())
                #z1[z1 < 0] = 0

                #oldz1 = z1
                z1 = np.insert(z1, 0, 1, axis = 0)
                #self.z1 = z1

                z2 = self.sigmoid( (self.weights1).dot(z1) )
                z2 = np.insert(z2, 0, 1, axis = 0)
                mm = len(z2)
                nn = len(z2[0])

                output = self.entropy(z2)

                self.weights2 = self.weights2 + self.step * ((teacher - output).dot(z2.transpose()))

                delta1 = ( z2 * (np.ones((mm, nn)) - z2) * ( self.weights2.transpose().dot(teacher - output))    )

                self.weights1 = self.weights1 + (self.step *  delta1.dot(z1.transpose() ) )[1:]

                z1 = z1[1:]
                m = len(z1)
                n = len(z1[0])

                self.weights0 = self.weights0 + (self.step * ( z1 * (np.ones((m, n)) - z1) * ( self.weights1.dot(delta1))    ).dot(data)  )
                #self.weights0 = self.weights0 + (self.step * ( (    (np.ones((m, n)) - (np.tanh(z1)) ) * (np.ones((m, n)) + (np.tanh(z1)) )  ) * ( self.weights1.transpose().dot(teacher - output))   ).dot(data) )[1:]
                #z1[z1 > 0] = 1
                #self.weights0 = self.weights0 + (self.step * ( z1 * ( self.weights1.transpose().dot(teacher - output))    ).dot(data) )[1:]

                #add regularization
                #self.weights1 = self.weights1 + self.step * ((teacher - output).dot(z1.transpose())) - 2 * _lambda * self.weights1
                #self.weights0 = self.weights0 + (self.step * ( z1 * (np.ones((m, n)) - z1) * ( self.weights1.transpose().dot(teacher - output))    ).dot(data) )[1:] - 2 * _lambda * self.weights0

                #add momentum
                #mm = 0.9
                #self.weights1 = self.weights1 + self.step * ((teacher - output).dot(z1.transpose())) + mm * self.pweights1
                #self.pweights1 = self.step * ((teacher - output).dot(z1.transpose()))
                #self.weights0 = self.weights0 + (self.step * ( z1 * (np.ones((m, n)) - z1) * ( self.weights1.transpose().dot(teacher - output))    ).dot(data) )[1:] + (mm * self.pweights0)
                #self.pweights0 = (self.step * ( z1 * (np.ones((m, n)) - z1) * ( self.weights1.transpose().dot(teacher - output))    ).dot(data) )[1:]
                #print " count:", count, "rack : ", l
            count = count + 1

    def entropy(self, data):
        numerator = np.exp(self.weights2.dot(data))
        denominator = np.sum(numerator, axis = 0)
        return numerator/denominator

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-1 * x))

    def predict(self, data):
        z1 = self.sigmoid( (self.weights0).dot(data.transpose()) )
        z1 = np.insert(z1, 0, 1, axis = 0)
        output = self.entropy(z1)
        return output

    def predict2(self, data):
        z1 = np.tanh( (self.weights0).dot(data.transpose()) )
        z1 = np.insert(z1, 0, 1, axis = 0)
        output = self.entropy(z1)
        return output

    def predict3(self, data):
        z1 = (self.weights0).dot(data.transpose())
        z1[z1 < 0] = 0
        z1 = np.insert(z1, 0, 1, axis = 0)
        output = self.entropy(z1)
        return output

    def predict4(self, data):
        z1 = self.sigmoid( (self.weights0).dot(data.transpose()) )
        z1 = np.insert(z1, 0, 1, axis = 0)
        z2 = self.sigmoid( (self.weights1).dot(z1) )
        z2 = np.insert(z2, 0, 1, axis = 0)
        output = self.entropy(z2)
        return output

    def E(self, weight, data):
        numerator = np.exp(weight.dot(data))
        denominator = np.sum(numerator, axis = 0)
        y = numerator/denominator
        y = np.log(y)
        return sum(self.teacher*y)

    def checkgradient(self):
        epsilon = 10e-5 * np.ones((10,21))
        hidden_gradient = (self.E(self.weights1 + epsilon, self.z1) - self.E(self.weights1 - epsilon, self.z1)) / (2*10e-5)
        print hidden_gradient

    def scanweight(self):
        print self.weights0


def run():
    model = NN(10e-5, 785, 10, traindata, trainlabels)
    for i in range(1,101):
        print "--------------------------------",10*i,"-------------------------------"
        model.train(10*i)
        #model.checkgradient()

        output = model.predict4(traindata)
        output = np.argmax(output, axis = 0)
        N = 0
        for j in range(len(trainlabels)):
            if output[j] == trainlabels[j]:
                N = N + 1
        print " Accuracy is : " , (N*1.0/len(trainlabels))*100, "%"

        output = model.predict4(testdata)
        output = np.argmax(output, axis = 0)
        N = 0
        for j in range(len(testlabels)):
            if output[j] == testlabels[j]:
                N = N + 1
        print " Accuracy is : " , (N*1.0/len(testlabels))*100, "%"


# read data
traindata, trainlabels = load_mnist('training', 60000)
testdata, testlabels = load_mnist('testing', 10000)
#print len(traindata[0])
#print traindata[0]

run()
