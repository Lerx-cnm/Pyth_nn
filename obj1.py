import numpy, random, os
from sklearn import preprocessing
import pandas as pd

class Percept:

    def __init__(self):
        self.ind = 1
        self.lr = 0.1
        self.bias = -1.5
        self.bweight = random.random()
        self.weights = [random.random(), random.random(), random.random(), random.random()
        , random.random(), random.random(), random.random(), random.random()
        , random.random(), random.random()]
        # self.outputP = 0
    def getData(self):
        df = pd.read_csv('./housepricedata.csv')
        dataset = df.values
        x = dataset[:,0:10]
        self.y = dataset[:,10]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.x_scale = min_max_scaler.fit_transform(x)
        self.max = min_max_scaler.data_max_
        self.min = min_max_scaler.data_min_
    def calc(self, inputs, output):
        outputP = inputs[0]*self.weights[0] + inputs[1]*self.weights[1] + inputs[2]*self.weights[2]
        + inputs[3]*self.weights[3] + inputs[4]*self.weights[4] + inputs[5]*self.weights[5] + inputs[6]*self.weights[6]
        + inputs[7]*self.weights[7] + inputs[8]*self.weights[8] + inputs[9]*self.weights[9] + self.bias*self.bweight
        self.ind += 1
        outputP = 1/(1+numpy.exp(-outputP)) #sigmoid function
        error = output - outputP
        self.bweight += error * self.bias*self.lr
        for i in range(10):
            # print(i)
            self.weights[i] += error * inputs[i]* self.lr

    def learn(self):
        for i in range(51):
            if i % 50 == 0:
                print(i)
            for i in range(1460):
                # print(i)
                self.calc(self.x_scale[i], self.y[i])

    def guess(self, input):
        inputs = self.norm(input)
        # print(inputs)
        # print(inputs[0], self.weights[0])
        outputP = inputs[0]*self.weights[0] + inputs[1]*self.weights[1] + inputs[2]*self.weights[2]
        + inputs[3]*self.weights[3] + inputs[4]*self.weights[4] + inputs[5]*self.weights[5] + inputs[6]*self.weights[6]
        + inputs[7]*self.weights[7] + inputs[8]*self.weights[8] + inputs[9]*self.weights[9] + self.bias * random.random()   # if outputP > 0:
        #     outputP = 1
        # else:
        #     outputP = 0
        # print(outputP)
        if outputP <= 0:
            print(outputP)
        else:
            outputP = 1/(1+numpy.exp(-outputP)) #sigmoid function
            print(outputP)
        
    def norm(self, inputs):
        print(self.min)
        array = []
        for i in range(10):
            norm_inp = (inputs[i]-self.min[i])/(self.max[i]-self.min[i])
            array.append(norm_inp)
        return array


test = Percept()
test.getData()
test.learn()
input = [1400, 1, 1, 100, 0, 0, 3, 2, 1, 250]
# input = [0,0,0,0,0,0,0,0,0,0]
test.guess(input)