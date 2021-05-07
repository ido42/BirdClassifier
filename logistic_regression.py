import numpy as np

class logistic_regression:

    def __init__(self, input_size, learning_rate, num_classes, reg_lambda):
        self.input_size = input_size
        self.num_classes = num_classes  # the number of bird species
        self.weights = np.random.rand(self.input_size[1] ,
                                      self.num_classes)  # one vs all, each column for one class
        self.l_rate = learning_rate
        self.reg_lambda = reg_lambda

    def gradient_descent(self, inputs, labels):  # label rows are one hot
        #ones = np.ones((self.input_size[0], 1))
        #inputs = np.concatenate((ones, inputs), axis=1)  # with bias
        for c in range(self.num_classes):
            pred = np.matmul(inputs, self.weights[:, c])
            sigmoid = np.nan_to_num(1 / (1 + np.exp(-pred)))
            while any(abs(labels[:, c] - sigmoid) > np.ones(np.shape(labels[:, c])) * 0.02):
                grad = np.dot(inputs.transpose(),sigmoid-labels[:,c])
                self.weights[:,c]=self.weights[:,c]-self.l_rate*grad/self.input_size[0]
                self.weights[1:,c]=self.weights[1:,c]-self.weights[1:,c]*self.l_rate*self.reg_lambda/self.input_size[0]
                pred = np.matmul(inputs, self.weights[:, c])
                sigmoid = np.nan_to_num(1 / (1 + np.exp(-pred)))

    def cost(self,inp,label):
        pred = np.matmul(inp, self.weights)
        sigmoid = np.nan_to_num(1 / (1 + np.exp(-pred)))
        error=-np.sum(label*np.log(sigmoid))
        return error

    def classify(self,inp):#while classifying one input at a time,input is a row vector
        #inp=np.append([1],inp)
        #inp=np.reshape(inp,(1,len(inp)))
        inp = (inp - inp.mean()) / np.sqrt(inp.var()) #normalize the vect
        classes_probabilities=1/(1+np.exp(-np.matmul(inp,self.weights)))# each element has the probability of relative class, this is a row vector
        try:
            bird_num=int(np.where(classes_probabilities==np.max(classes_probabilities))[0])
            bird_class_encoded=np.zeros((1,self.num_classes))
            bird_class_encoded[:,bird_num]=1
            Fail=False
        except:
            bird_class_encoded=None
            Fail=True
        return bird_class_encoded,Fail

