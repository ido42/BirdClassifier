import numpy as np

class logistic_regression:

    def __init__(self, input_size, learning_rate, num_classes, reg_lambda):
        self.input_size = input_size
        self.num_classes = num_classes  # the number of bird species
        self.weights = np.random.rand(self.input_size[1] + 1,
                                      self.num_classes)  # one vs all, each column for one class
        self.l_rate = learning_rate
        self.reg_lambda = reg_lambda

    def gradient_descent(self,inputs,labels): # label rows are one hot
        ones=np.ones((self.input_size[0],1))
        inputs=np.concatenate((ones,inputs),axis=1)  #with bias
        pred=np.matmul(inputs,self.weights[:,0])
        sigmoid=np.nan_to_num(np.exp(pred)/(1+np.exp(pred)))
        results=[]
        for i in range(self.num_classes):
            while any(abs(labels[:,i]-sigmoid)>np.ones(np.shape(labels[:,i]))0.001):
                pred=np.matmul(inputs,self.weights[:,0])
                sigmoid=np.nan_to_num(np.exp(pred)/(1+np.exp(pred)))
                der_cost=(self.l_rate/self.input_size[0])np.matmul(np.transpose(inputs),(sigmoid-labels[:,i]))
                self.weights[:,0]=self.weights[:,0]-der_cost
            results.append(sigmoid)
        return(results)

    def classify(self,inp):#while classifying one input at a time,input is a row vector
        inp=np.append([1],inp)
        inp=np.reshape(inp,(1,len(inp)))
        inp = (inp - inp.mean()) / np.sqrt(inp.var()) #normalize the vect
        classes_probabilities=1/(1+np.exp(-np.matmul(inp,self.weights)))# each element has the probability of relative class, this is a row vector
        bird_num=int(np.where(classes_probabilities==np.max(classes_probabilities))[0])
        bird_class_encoded=np.zeros((1,self.num_classes))
        bird_class_encoded[:,bird_num]=1
        return (bird_class_encoded)

    def gradient_descent(self, inputs, labels):  # label rows are one hot
        ones = np.ones((self.input_size[0], 1))
        inputs = np.concatenate((ones, inputs), axis=1)  # with bias
        pred = np.matmul(inputs, self.weights[:, 0])
        sigmoid = np.nan_to_num(1 / (1 + np.exp(-pred)))
        results = []
        for i in range(self.num_classes):
            while any(abs(labels[:, i] - sigmoid) > np.ones(np.shape(labels[:, i])) * 0.01):
                pred = np.matmul(inputs, self.weights[:, i])
                sigmoid = np.nan_to_num(1 / (1 + np.exp(-pred)))
                der_cost = (self.l_rate / self.input_size[0]) * np.matmul(np.transpose(inputs),
                                                                          (sigmoid - labels[:, i]))
                regularized_cost = -self.weights[1:, i] * self.reg_lambda * self.l_rate / self.input_size[0]
                self.weights[:, i] = self.weights[:, i] - der_cost
                self.weights[1:, i] = self.weights[1:, i] - regularized_cost
                results.append(der_cost)
        return (results)