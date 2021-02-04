# This is a .py file with the name "network.py"
import numpy as np
import matplotlib.pyplot as plt
import layer
from layer import Linear, Softmax

class Network:
    def __init__(self):
        self.filePathLocal_labels = 'Dataset/batches.meta'
        self.filePathLocal_batch = 'Dataset/data_batch_1'
        self.filePathLocal_data_TRAIN = 'Dataset/data_batch_1'
        self.filePathLocal_data_VALIDATION = 'Dataset/data_batch_2'
        self.filePathLocal_data_TEST = 'Dataset/test_batch'
        self.K = 10                     # number of labels/classes
        self.d = 3072                   # number of dimensions of an image 32x32x3
        self.mu, self.sigma = 0, 0.01   # mean and standard deviation
        self.batch_length = 100         # n=100 samples picked as a subset ... N=10000=number of images in the training set
        self.lambda_cost = 0
        self.h = 1e-6
        self.eps = 1e-6
    
    # filePathList[0] = filePathLocal_data_TRAIN, filePathList[1] = filePathLocal_data_VALIDATION
    # consindering that all TRAIN, VALIDATION and TEST data are in different files
    def ReadData(self, cifar, filePathList):
        # Top-level: Read in and store the training, validation and test data.
        # cifar_batch = CIFAR_IMAGES()
        [self.train_X, self.train_Y, self.train_y] = cifar.load_batch_a1(filePathList[0])
        [self.validation_X, self.validation_Y, self.validation_y] = cifar.load_batch_a1(filePathList[1])
        [self.test_X, self.test_Y, self.test_y] = cifar.load_batch_a1(filePathList[2])
        
    def ReadData_Exercise4(self, cifar, filePathList):
        # Top-level: Read in and store the training, validation and test data.
        # cifar_batch = CIFAR_IMAGES()
        # 'filePathList[0] = ['Dataset/data_batch_1', 'Dataset/data_batch_2', 'Dataset/data_batch_3', 'Dataset/data_batch_4', 'Dataset/data_batch_5']
        list_X_all = []
        list_Y_all = []
        list_y_all = []

        for file in filePathList[0]:
            [temp_train_X, temp_train_Y, temp_train_y] = cifar.load_batch_a1(file)
            list_X_all.append(temp_train_X)
            list_Y_all.append(temp_train_Y)
            list_y_all.append(temp_train_y)   

        self.X_all = list_X_all[0]
        self.Y_all = list_Y_all[0]
        self.y_all = list_y_all[0]
        for i in range(len(list_X_all)): 
            if i != 0:
                self.X_all = np.hstack((self.X_all, list_X_all[i]))
                self.Y_all = np.hstack((self.Y_all, list_Y_all[i]))
                self.y_all = np.append(self.y_all, list_y_all[i])
            
        [self.train_X, self.train_Y, self.train_y] = [self.X_all[:, :45000], self.Y_all[:, :45000], self.y_all[:45000]]
        [self.validation_X, self.validation_Y, self.validation_y] = [self.X_all[:, 45000:], self.Y_all[:, 45000:], self.y_all[45000:]]
        [self.test_X, self.test_Y, self.test_y] = cifar.load_batch_a1(filePathList[1])
    
    # USED
    def MeanStd_Train_X(self, train_X):
        # Top-level: Compute the mean and standard deviation vector for the training data and then normalize the 
        # training, validation and test data w.r.t. these mean and standard deviation vectors.
        mean_train_X = self.train_X.mean(axis=1)
        #mean_train_X_broadcast = np.tile(mean_train_X, (10000, 1))
        mean_train_X_broadcast = np.tile(mean_train_X, (train_X.shape[1], 1))
        std_train_X = self.train_X.std(axis=1)
        #std_train_X_broadcast = np.tile(std_train_X, (10000, 1))
        std_train_X_broadcast = np.tile(std_train_X, (train_X.shape[1], 1))
        #plt.hist(std_train_X)
        return (mean_train_X_broadcast, std_train_X_broadcast)
    
    def NormalizeData(self, inputData, trainX_Broadcast_MeanStd):
        # http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting
        normalizedInputData = (inputData - trainX_Broadcast_MeanStd[0].transpose())\
        /trainX_Broadcast_MeanStd[1].transpose()
        return normalizedInputData
    
    def NormalizeData_Per_DataSet(self, inputData):
        mean_X = inputData.mean(axis=1)
        mean_X_broadcast = np.tile(mean_X, (inputData.shape[1], 1))
        std_X = inputData.std(axis=1)
        
        std_X_broadcast = np.tile(std_X, (inputData.shape[1], 1))
        
        # http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting
        normalizedInputData = (inputData - mean_X_broadcast.transpose())/std_X_broadcast.transpose()
        return normalizedInputData
    
    def NormalizeData_Broadcast(self, inputData, trainX):
        # http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting
        mean_train_X = self.train_X.mean(axis=1)
        mean_train_X_broadcast = np.tile(mean_train_X, (inputData.shape[1], 1))
        
        std_train_X = self.train_X.std(axis=1)
        std_train_X_broadcast = np.tile(std_train_X, (inputData.shape[1], 1))
        
        normalizedInputData = (inputData - mean_train_X_broadcast.transpose())/std_train_X_broadcast.transpose()
        return normalizedInputData
    
    def Initialize_W_b(self, initial_sizes, sigma1, sigma2):
        # Top-Level: After reading in and pre-processing the data, you can initialize the parameters of the model 
        # W and b as you now know what size they should be. W has size Kxd and b is Kx1. Initialize each entry to have 
        # Gaussian random values with zero mean and standard deviation .01. 
        # You should use the Matlab function randn to create this data.
        mu = initial_sizes[0] # check init for definitions
        d = initial_sizes[1]
        m = initial_sizes[2]
        K = initial_sizes[3]
        
        #sigma1 = int(np.sqrt(d))
        #sigma2 = int(np.sqrt(m))
        
        W1 = np.random.normal(mu, sigma1, (m, d))
        W2 = np.random.normal(mu, sigma2, (K, m))
        
        b1 = np.zeros((m, 1))
        b2 = np.zeros((K, 1))
        
        return (W1, W2, b1, b2)
    
    def EvaluationClassifier(self, layers, X, W, b):
        # W = [W1, W2]
        # X = each column of X corresponds to an image and it has size (d x N) >> here N will be smaller since 
        # it will be selected as subset of images n=100 can be selected
        # b = [b1, b2]
        # layers = [linearLayer1, reluLayer, linearLayer2, softmaxLayer]

        # 1st layer signs                                  # S1 = (m, n) 
        S1 = layers[0].Forward(X, W[0], b[0])
        #print('S1 = {}'.format(S1.shape))
        
        # signs after ReLu filtering                       # H = (m, n)
        H = layers[1].Forward(S1)
        #print('H = {}'.format(H.shape))
        
        # middle (2nd) layer signs                         # S = (K, n)
        S = layers[2].Forward(H, W[1], b[1])
        #print('S = {}'.format(S.shape))
        
        # P = probabilities of each class to the corresponding images  # P = (K, n)
        P = layers[3].Forward(S)
        #print('P = {}'.format(P.shape))
        
        return P, H
    
    def EvaluationClassifier_loop(self, layers, X, W, b):
        # W = [W1, W2]
        # X = each column of X corresponds to an image and it has size (d x N) >> here N will be smaller since 
        # it will be selected as subset of images n=100 can be selected
        # b = [b1, b2]
        # layers = [linearLayer1, reluLayer, linearLayer2, softmaxLayer]
        linear_layer = 0
        inputData = X
        
        results = []
        
        #layer.
        for lyr in layers:
            if type(lyr) == layer.Linear:
                inputData = lyr.Forward(inputData, W[linear_layer], b[linear_layer])
                linear_layer += 1
            elif type(lyr) == layer.ReLU:
                inputData = lyr.Forward(inputData)
                results.append(inputData)
            elif type(lyr) == layer.Softmax:
                inputData = lyr.Forward(inputData)
                results.append(inputData)

        #print('P = {}'.format(P.shape))
        H = results[0]
        P = results[1]
        
        return P, H
    
    def Cost(self, X, Y, W, b, lambda_cost):
        # for k-layer NN FORWARD pass & COST calculations: Lecture4-Pg.33 (46)
        cost_sum = 0
        
        b1_broadcast = np.tile(b[0], (1, X.shape[1]))
        S1 = np.dot(W[0], X) + b1_broadcast
        
        H = S1 * (S1 > 0)
        
        b2_broadcast = np.tile(b[1], (1, H.shape[1]))
        S = np.dot(W[1], H) + b2_broadcast
        
        # Softmax
        exp_s = np.exp(S)
        P = exp_s / np.sum(exp_s, axis=0)
        
        cross_entropy_loss = -np.log(np.dot(Y.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y.shape[1]
        
        total_cost = sum_cross_entropy_loss/N + lambda_cost*(np.power(W[0], 2).sum() + np.power(W[1], 2).sum())
        
        return total_cost
    
    def Cost_loop(self, X, Y, W, b, lambda_cost):
        cost_sum = 0
        
        b1_broadcast = np.tile(b[0], (1, X.shape[1]))
        S1 = np.dot(W[0], X) + b1_broadcast
        
        H = S1 * (S1 > 0)
        
        b2_broadcast = np.tile(b[1], (1, H.shape[1]))
        S = np.dot(W[1], H) + b2_broadcast
        
        # Softmax
        exp_s = np.exp(S)
        P = exp_s / np.sum(exp_s, axis=0)
        
        cross_entropy_loss = -np.log(np.dot(Y.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y.shape[1]
        
        total_cost = sum_cross_entropy_loss/N + lambda_cost*np.power(W[0], 2).sum() + lambda_cost*np.power(W[1], 2).sum()
        
        return total_cost
      
    
    def ComputeAccuracy(self, k, y):
        # k = predictions=the label with the max(probability) = Nx1
        # y = ground_truth_labels = Nx1
        # N = batch_length = number of images used
        N = k.shape[0]
        return np.sum(k == y)/N
    