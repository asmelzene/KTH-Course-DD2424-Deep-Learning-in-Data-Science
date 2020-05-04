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
        
    def Initialize_W_b_ex(self, d, m, K, sigma1, sigma2):
        # Top-Level: After reading in and pre-processing the data, you can initialize the parameters of the model 
        # W and b as you now know what size they should be. W has size Kxd and b is Kx1. Initialize each entry to have 
        # Gaussian random values with zero mean and standard deviation .01. 
        # You should use the Matlab function randn to create this data.
        mu = 0
        
        #sigma1 = int(np.sqrt(d))
        #sigma2 = int(np.sqrt(m))
        
        W1 = np.random.normal(mu, sigma1, (m, d))
        W2 = np.random.normal(mu, sigma2, (K, m))
        
        b1 = np.zeros((m, 1))
        b2 = np.zeros((K, 1))
        
        return (W1, W2, b1, b2)
    
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
    
    # layers = [linearLayer1, reluLayer, linearLayer2, softmaxLayer]
    def Cost_ex(self, layers, X, Y, W, b, lambda_cost):
        cost_sum = 0
        for l in layers:
            if type(l) is Linear:
                linCost = l.Cost(lambda_cost, W)
                print(linCost.shape)
                cost_sum += linCost
            elif type(l) is Softmax:
                softCost = l.Cost(X, Y, b)
                print(softCost.shape)
                cost_sum += softCost
        
        return cost_sum
    
    def Cost_xx(self, X, Y, W, b, lambda_cost):
        cost_sum = 0
        for i in range(len(layers)):
            if type(l) is Linear:
                linCost = l.Cost(lambda_cost, W)
                print(linCost.shape)
                cost_sum += linCost
            elif type(l) is Softmax:
                softCost = l.Cost(X, Y, b)
                print(softCost.shape)
                cost_sum += softCost
        
        return cost_sum
    
    def Cost_yy(self, X, Y, W, b, lambda_cost):
        cost_sum = 0
        lin1 = Linear()
        
        for i in range(len(W)):
                linCost = lin1.Cost(lambda_cost, W[i])
                cost_sum += linCost

        soft1 = Softmax()
        softCost = soft1.Cost(X, Y, b)
        cost_sum += softCost
        
        return cost_sum
    
    def Cost(self, X, Y, W, b, lambda_cost):
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
    
    
    def Task2_test(self):
        print("Train_X_Normalized: " + self.train_X_normalized.mean(axis=1))
        print("Train_X_Normalized: " + \
              str(sum(self.train_X_normalized.mean(axis=1))/len(self.train_X_normalized.mean(axis=1))))
        print(str(max(self.train_X_normalized.mean(axis=1))))
        print(str(min(self.train_X_normalized.mean(axis=1))))
        #print(mean_train_X_broadcast.shape)    
        
    def Task4(self, comp):
        # comp = COMPUTATIONS()
        # Top-level: Check the function runs on a subset of the training data given a random initialization of the 
        # network's parameters: P = EvaluateClassifier(trainX(:, 1:100), W, b)
        self.train_X_batch = self.train_X[:, 0:self.batch_length]
        self.P_batch = comp.EvaluationClassifier(self.train_X_batch, self.W, self.b)
        self.train_Y_batch = self.train_Y[:, 0:self.batch_length]
        self.train_y_batch = self.train_y[0:self.batch_length]
        #print(P.shape) # (10, 100) (K, n)

    def Task5(self, comp):
        # comp = COMPUTATIONS()
        # Write the function that computes the cost function given by equation
        # J = ComputeCost(X, Y, W, b, lambda)
        # J = cost = ComputeCost(X, Y, W, b, lambda) >> in the assignment document.
        # ComputeCost(Y, P, W, lambda) >> we use only those since X, b have already been used to calculate P
        #J = comp.ComputeCost(train_Y_batch, P_batch, W, lambda_cost)
        # b=(10, 1)    ...   W=(10, 3072)    ...  X=(3072, 100)
        # b_broadcast_test = np.tile(b, (1, train_X_batch.shape[1]))  >> b_broadcast_test.shape=(10, 100)        
        self.J = comp.ComputeCost(self.train_Y_batch, self.train_X_batch, self.b, self.W, self.lambda_cost)

    def Task6(self, comp):
        # comp = COMPUTATIONS()
        # Write a function that computes the accuracy of the network's predictions given by equation(4) on a set of data.
        # k = predictions
        self.k = np.argmax(self.P_batch, axis=0)
        self.acc = comp.ComputeAccuracy(self.k, self.train_y_batch)
        print("Accuracy (Task6): " + str(self.acc))  # acc = 0.13
        
    def Task7(self, comp):
        # comp = COMPUTATIONS()
        # Write the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b, 
        # that is equations (10, 11).
        start_time = datetime.datetime.now()
        [self.grad_W, self.grad_b] = comp.ComputeGradients(self.train_Y_batch, self.P_batch, self.train_X_batch, self.lambda_cost, self.W)
        [self.grad_W_num, self.grad_b_num] = comp.ComputeGradsNum(self.train_X_batch, self.train_Y_batch, self.P_batch, self.W, self.b, self.lambda_cost, self.h)
        end_time = datetime.datetime.now()
        print("Calculation time: "+ str(end_time - start_time))
        
    def Task7_test1(self):
        #print(train_X_batch.shape)
        #print(grad_W_num.shape)
        #print(grad_b_num.shape) # grad_b_num.shape=(10, 1)     grad_b.shape=(10,)
        #print("sum_grad_b_difference = " + str(np.abs(grad_b - grad_b_num.transpose()).sum()))
        
        ######### W #######
        print("\ngrad_W_difference_MEAN = "+ str(np.abs(self.grad_W - self.grad_W_num).mean()))
        print("grad_W_difference_MIN = "+ str(np.abs(self.grad_W - self.grad_W_num).min()))
        print("grad_W_difference_MAX = "+ str(np.abs(self.grad_W - self.grad_W_num).max()))

        #print("sum_grad_W_difference = " + str(np.abs(grad_W - grad_W_num).sum()))

        print("\ngrad_W_MIN = " + str(np.min(np.abs(self.grad_W))))
        print("grad_W_num_MIN = " + str(np.min(np.abs(self.grad_W_num))))
        print("grad_W_MAX = " + str(np.max(np.abs(self.grad_W))))
        print("grad_W_num_MAX = " + str(np.max(np.abs(self.grad_W_num))))
        
        ######### b #######
        print("\ngrad_b_difference_MEAN = "+ str(np.abs(self.grad_b - self.grad_b_num.transpose()).mean()))
        print("grad_b_difference_MIN = "+ str(np.abs(self.grad_b - self.grad_b_num.transpose()).min()))
        print("grad_b_difference_MAX = "+ str(np.abs(self.grad_b - self.grad_b_num.transpose()).max()))
        print("\ngrad_b_MIN = " + str(min(np.abs(self.grad_b))))
        print("grad_b_num_MIN = " + str(min(np.abs(self.grad_b_num))))
        print("\ngrad_b_MAX = " + str(max(np.abs(self.grad_b))))
        print("grad_b_num_MAX = " + str(max(np.abs(self.grad_b_num))))
            
    def Task7_test2(self):
        diff2_W = np.abs(self.grad_W - self.grad_W_num)
        sum2_W = np.abs(self.grad_W) + np.abs(self.grad_W_num)
        
        eps2_W = np.full_like(sum2_W, 1e-6)
        #print(eps2)
        #print(eps2.shape)

        sum2_W_max = np.zeros_like(sum2_W)

        for r in range(sum2_W.shape[0]):
            for c in range(sum2_W.shape[1]):
                if(sum2_W[r, c] > eps2_W[r, c]):
                    sum2_W_max[r, c] = sum2_W[r, c]
                else:
                    sum2_W_max[r, c] = eps2_W[r, c]
     
        self.gradient_W_error_check_2 = diff2_W/sum2_W_max
        
        print("gradient_W_error_check_2-MAX: " + str(np.max(self.gradient_W_error_check_2)))
        print("gradient_W_error_check_2-MIN: " + str(np.min(self.gradient_W_error_check_2)))
        print("gradient_W_error_check_2-MEAN: " + str(np.mean(self.gradient_W_error_check_2)))
        print("gradient_W_error_check_2-STD: " + str(np.std(self.gradient_W_error_check_2)))
        #print(sum2_W_max)
        #print(np.max(sum2_W_max))
        #print(np.min(sum2_W_max))

        #self.grad_b_reshape = self.grad_b.reshape(self.grad_b_num[0], self.grad_b_num[1])
        
        diff2_b = np.abs(self.grad_b - self.grad_b_num.transpose())
        sum2_b = np.abs(self.grad_b) + np.abs(self.grad_b_num.transpose())
        
        eps2_b = np.full_like(sum2_b, 1e-6)
        
        sum2_b_max = np.zeros_like(sum2_b)
        
        self.sum2_b_test = sum2_b
        self.eps2_b_test = eps2_b
        
        for r in range(sum2_b.shape[1]):
            if(sum2_b[0][r] > eps2_b[0][r]):
                sum2_b_max[0][r] = sum2_b[0][r]
            else:
                sum2_b_max[0][r] = eps2_b[0][r]

        #print(sum2_b_max)
        #print(np.max(sum2_b_max))
        #print(np.min(sum2_b_max))

        self.gradient_b_error_check_2 = diff2_b/sum2_b_max
 
        print("\ngradient_b_error_check_2-MAX: " + str(np.max(self.gradient_b_error_check_2)))
        print("gradient_b_error_check_2-MIN: " + str(np.min(self.gradient_b_error_check_2)))
        print("gradient_b_error_check_2-MEAN: " + str(np.mean(self.gradient_b_error_check_2)))
        print("gradient_b_error_check_2-STD: " + str(np.std(self.gradient_b_error_check_2)))
        
        '''
        self.gradient_W_error_check_2 = np.abs(self.grad_W[0, 0] - \
                                          self.grad_W_num[0, 0])/max(self.eps, np.abs(self.grad_W[0, 0]) +\
                                                                               np.abs(self.grad_W_num[0, 0]))
        print("\ngradient_W_error_check_2: " + str(self.gradient_W_error_check_2))
        
        self.gradient_b_error_check_2 = np.abs(self.grad_b[0] - \
                                          self.grad_b_num[0])/max(self.eps, \
                                                            np.abs(self.grad_b[0]) + np.abs(self.grad_b_num[0]))
        print("\ngradient_b_error_check_2: " + str(self.gradient_b_error_check_2))
        '''
    
    def Task8(self, comp, GDparams):
        # Once you have the gradient computations debugged you are now ready to write the code to perform the mini-batch 
        # gradient descent algorithm to learn the network's parameters
        #comp = COMPUTATIONS()
        #GDparams = [n_batch, eta, n_epocs, lambda_cost] 
        TrainData = [self.train_X_normalized, self.train_Y, self.train_y]
        ValidationData = [self.validation_X_normalized, self.validation_Y, self.validation_y]
        TestData = [self.test_X_normalized, self.test_Y, self.test_y]
        
        print("************ CostCalculations ************\n")
        print("GDparams: [n_batch={}, eta={}, n_epocs={}, lambda_cost={}]".\
              format(GDparams[0], GDparams[1], GDparams[2], GDparams[3]))
        
        W_final, b_final = comp.CostCalculations(GDparams, TrainData, ValidationData, TestData)
        return W_final, b_final
    