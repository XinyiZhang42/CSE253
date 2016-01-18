import numpy
import random
import matplotlib.pyplot as plt

train_data = []
label = ["sepal length", "sepal width", "petal length", "petal width"]
means = []
stds = []

class trainModel:

    def __init__ (self, learning_rate, dimension):
        self.learning_rate = learning_rate
        self.dimension = dimension
        self.threshold = 0
        self.weights = numpy.array([0 for i in range(dimension)])

    def train(self, train_data):
        self.ERROR = 0
        Itertimes = 0
        while Itertimes < 1000: #and not self.check(train_data):
            row = train_data[random.randint(0, len(train_data)-1)]
            teacher = row[self.dimension]
            x = row[:self.dimension]
            output = self.predict(x)
            self.weights = self.weights + self.learning_rate * (teacher - output) * x
            self.threshold = self.threshold - self.learning_rate * (teacher - output)
            Itertimes = Itertimes + 1

    def predict(self, x):
        v = sum(x[:self.dimension] * self.weights)
		
        if v >= self.threshold:
            return 1
        else:
            return 0

    def check(self, data):
        err = 0
        for row in data:
            v = self.predict(row)
            if v != row[self.dimension]:
                err = err + 1
        if err == self.ERROR and err < 3:
            return True
        else:
            self.ERROR = err
            return False

def read(filepath):
    file = open(filepath,"r")
    data = []
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.strip('\n')
        row = line.split(',')
        for i in xrange(4):
		    row[i] = float(row[i])
        row[4] = 1 if row[4] == "Iris-setosa" else 0
        data.append(row)
    return data

def zscore(data):
    mydata = numpy.vstack(data)
    for i in xrange(4):
        mean = numpy.mean(mydata.T[i])
        std = numpy.std(mydata.T[i])
        means.append(mean)
        stds.append(std)
        for j in xrange(len(mydata.T[i])):
            mydata[j][i] = (mydata[j][i] - mean)/std
    return mydata

def plot_data():   
    setosa_data = []
    versicolor_data = []
    for ele in train_data:
        if ele[4] == 1:
            setosa_data.append(ele)
        else:
            versicolor_data.append(ele)
    setosa_data = numpy.vstack(setosa_data)
    versicolor_data = numpy.vstack(versicolor_data)

    # plot my data
    for i in xrange(3):
        for j in xrange(3 - i):
            plt.scatter(setosa_data.T[i], setosa_data.T[i+j+1], color = 'blue')
            plt.scatter(versicolor_data.T[i], versicolor_data.T[i+j+1], color = 'green')
            plt.xlabel(label[i])
            plt.ylabel(label[i+j+1])
            plt.show()

def Runtest(learning_rate):
    print "Running as Learning Rate : ", learning_rate
    model = trainModel(learning_rate, 4)
    model.train(train_data)
    data = read("iris_test.data")
    mydata = numpy.vstack(data)
    for i in xrange(4):
        for j in xrange(len(mydata.T[i])):
            mydata[j][i] = (mydata[j][i] - means[i])/stds[i]
 
    miss = 0
    for ele in mydata:
        label = model.predict(ele)
        if label != ele[-1]:
            miss = miss+1

    print "ERROR rate : ", miss * 100.0 / len(mydata), "%"


train_data = read("iris_train.data")
train_data = zscore(train_data)

Runtest(2)
Runtest(1)
Runtest(.5)
Runtest(.25)
