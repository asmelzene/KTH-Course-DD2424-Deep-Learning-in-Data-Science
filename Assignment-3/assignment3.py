import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import dataset_cifar
import datetime

def load_data(dataset, data_type = 'Test'):
    if data_type == 'Test':
        filePathLocal_labels = 'Dataset/batches.meta'
        filePathLocal_data_ALL = ['Dataset/data_batch_1', 'Dataset/data_batch_2', 
                    'Dataset/data_batch_3', 'Dataset/data_batch_4', 'Dataset/data_batch_5']
        filePathLocal_data_TEST = 'Dataset/test_batch'

        filePathList = (filePathLocal_data_ALL, filePathLocal_data_TEST)
        dataset.ReadData_Exercise4(dataset, filePathList)
        dataset.Normalize_ALL()
        
    elif data_type == 'Validation':
        filePathLocal_labels = 'Dataset/batches.meta'
        filePathLocal_data_TRAIN = 'Dataset/data_batch_1'
        filePathLocal_data_VALIDATION = 'Dataset/data_batch_2'
        filePathLocal_data_TEST = 'Dataset/test_batch'

        filePathList = (filePathLocal_data_TRAIN, filePathLocal_data_VALIDATION, filePathLocal_data_TEST)

        # Read TRAIN, VALIDATION, TEST data into numpy arrays (numpy.ndarray) from local files
        dataset.ReadData(dataset, filePathList)
        # X = (d, N), Y = (K, N), y = (N,)   # N=number of total images in X
        # X = (3072, 10000), Y = (10, 10000), y = (10000, 1)

        # Normalize all INPUT data by using their own MEAN and STD
        dataset.Normalize_ALL()

class Network:
    def __init__(self):
        self.mu, self.sigma = 0, 0.01   # mean and standard deviation
        self.batch_length = 100         # n=100 samples picked as a subset ... N=10000=number of images in the training set
        self.lambda_cost = 0
        self.h = 1e-6
        self.eps = np.finfo(float).eps     # or 1e-6 can be used
        self.step_no_points = []
        
        self.W_layers = {}
        self.b_layers = {}
        self.gamma_layers = {}
        self.beta_layers = {}
        self.Mu_layers = {}
        self.Var_layers = {}
        self.Mu_avg_layers = {}
        self.Var_avg_layers = {}
        self.X_layers = {}
        self.S_layers = {}
        self.S_bn_layers = {}
        self.grad_W = {}
        self.grad_b = {}
        self.grad_gamma = {}
        self.grad_beta = {}
        self.nodes_layer = {}
        self.init_type = 'Xavier'   # 'Xavier', 'He', 'Fixed'
        self.BN = True
        self.alpha = 0.9
    
    def Initialize_W_b_ALL(self, dataset, dimension = 'ALL'):
        # INPUT + Hidden Layers + OUTPUT (Input will be considered as layer-0)
        if dimension == 'ALL':
            list_nodes = [dataset.test_X.shape[0]] + self.hidden_layers + [dataset.test_Y.shape[0]]
        else:
            list_nodes = [dimension] + self.hidden_layers + [dataset.test_Y.shape[0]]
            
        self.nodes_layer = { i : list_nodes[i] for i in range(0, len(list_nodes)) }
        self.n_layers = len(self.nodes_layer) - 1  ## We start counting the layers from 1 not 0
        # e.g. (3072, 50, 50, 10)  ... hidden_layers = [50, 50]

        np.random.seed(400)
        for i in range(1, self.n_layers + 1): 
            n_output = self.nodes_layer[i]
            n_input = self.nodes_layer[i - 1]
            if self.init_type == 'Xavier':
                sigma1 = 1 / np.sqrt(n_input)
            elif self.init_type == 'He':
                sigma1 = 2 / np.sqrt(n_input)
            else:            # 'Fixed'
                sigma1 = self.init_type #self.sigma

            self.W_layers[i] = np.random.normal(self.mu, sigma1, (n_output, n_input))    # (mu, std, size)
            self.b_layers[i] = np.zeros((n_output, 1))
            
            if self.BN and i < self.n_layers:
                self.gamma_layers[i] = np.ones((n_output, 1))                            # size = (m, n)
                self.beta_layers[i] = np.zeros((n_output, 1))
    
    def ComputeGradsNumSlow_BN(self, X, Y):
        grad_W = {}; grad_b = {}; grad_gamma = {}; grad_beta = {}; training = True
        for layer_no in range(1, self.n_layers + 1):
            grad_W[layer_no] = (np.zeros(self.W_layers[layer_no].shape))
            grad_b[layer_no] = (np.zeros(self.b_layers[layer_no].shape))
        
            if self.BN and layer_no < self.n_layers:
                grad_gamma[layer_no] = (np.zeros(self.gamma_layers[layer_no].shape))
                grad_beta[layer_no] = (np.zeros(self.beta_layers[layer_no].shape))

        for layer_no in range(1, self.n_layers + 1):
            for i in range(self.b_layers[layer_no].shape[0]):
                self.b_layers[layer_no][i] += self.h
                cost_p, _ = self.Cost(X, Y, training)
                self.b_layers[layer_no][i] -= 2 * self.h
                cost_n, _ = self.Cost(X, Y, training)
                self.b_layers[layer_no][i] += self.h
                    
                grad_b[layer_no][i] = (cost_p - cost_n) / (2*self.h)

                for j in range(self.W_layers[layer_no].shape[1]):
                    self.W_layers[layer_no][i, j] += self.h
                    cost_p, _ = self.Cost(X, Y, training)
                    self.W_layers[layer_no][i, j] -= 2 * self.h
                    cost_n, _ = self.Cost(X, Y, training)
                    self.W_layers[layer_no][i, j] += self.h

                    grad_W[layer_no][i, j] = (cost_p - cost_n) / (2*self.h)
                              
        if self.BN:
            for layer_no in range(1, self.n_layers):
                for i in range(self.gamma_layers[layer_no].shape[0]):          
                    self.gamma_layers[layer_no][i] += self.h
                    cost_p, _ = self.Cost(X, Y, training)
                    self.gamma_layers[layer_no][i] -= 2 * self.h
                    cost_n, _ = self.Cost(X, Y, training)
                    self.gamma_layers[layer_no][i] += self.h

                    grad_gamma[layer_no][i] = (cost_p - cost_n) / (2*self.h)

            for layer_no in range(1, self.n_layers):
                for i in range(self.beta_layers[layer_no].shape[0]):
                    self.beta_layers[layer_no][i] += self.h
                    cost_p, _ = self.Cost(X, Y, training)
                    self.beta_layers[layer_no][i] -= 2 * self.h
                    cost_n, _ = self.Cost(X, Y, training)
                    self.beta_layers[layer_no][i] += self.h

                    grad_beta[layer_no][i] = (cost_p - cost_n) / (2*self.h)

        Grads_NUMERICAL = (grad_W, grad_b, grad_gamma, grad_beta)
                                  
        return Grads_NUMERICAL
    
    def ComputeGradients(self, X, Y, training = True):             
        # for k-layer NN aggregated backward pass calculations: Lecture4-Pg.34(47) to 36(49)
        n_layers = self.n_layers      
        n_images = X.shape[1]   
        # W_layers = we will pass all W values except the last layer, 
        # so we pass all W values to this function except the last one
        self.vector_ones = np.ones((n_images, 1));      self.vector_ones_T = np.ones((1, n_images))     
        
        ### FINAL LAYER
        P = self.EvaluationClassifier(X, training)
        G = -(Y - P)                                                                                # eq.21
        
        dL_dW = np.divide(np.dot(G, self.X_layers[n_layers - 1].T), n_images)                       # eq.22_1.1
        dL_dB = np.dot(G, self.vector_ones) / n_images                                              # eq.22_2.1

        self.grad_W[n_layers] = dL_dW + 2 * self.lambda_cost * self.W_layers[n_layers]              # eq.22_1
        self.grad_b[n_layers] = dL_dB                                                               # eq.22_2
        
        G = np.dot(self.W_layers[n_layers].T, G)                                                    # eq.23
        G = G * (self.X_layers[n_layers - 1] > 0)                                                   # eq.24 ... ReLu backward
        
        ### HIDDEN LAYERS
        layer_no = n_layers - 1
        while layer_no > 0:                            
            if self.BN:
                grad_gamma = np.dot((G * self.S_bn_layers[layer_no]), self.vector_ones) / n_images   # eq.25
                grad_beta = np.dot(G, self.vector_ones) / n_images                                   # eq.25
                G = G * (np.dot(self.gamma_layers[layer_no], self.vector_ones_T))                    # eq.26
                G = self.BatchNormBackPass_BN(G, layer_no)                                           # eq.27

            dL_dW = np.divide(np.dot(G, self.X_layers[layer_no - 1].transpose()), n_images)
            dL_dB = np.dot(G, self.vector_ones) / n_images
            
            self.grad_W[layer_no] = dL_dW + 2 * self.lambda_cost * self.W_layers[layer_no]           # eq.28 ... grad_W = dJ_dW
            self.grad_b[layer_no] = dL_dB                                                            # eq.28 ... grad_b = dJ_db
            
            if self.BN:
                self.grad_gamma[layer_no] = grad_gamma
                self.grad_beta[layer_no] = grad_beta
            
            if layer_no > 1:
                G = np.dot(self.W_layers[layer_no].T, G)                                             # eq.29
                G = G * (self.X_layers[layer_no - 1] > 0)                                            # eq.30

            layer_no -= 1

    def BatchNormBackPass_BN(self, G, layer_no):                                                     # eq.11
            n = G.shape[1];
            vector_ones = np.ones((n, 1));                   vector_ones_T = np.ones((1, n)) # >> 1nT = np.ones(1, n)
            
            # ***** CAUTION!: when sqrt is used, it gives an overflow exception!!
#             sigma1 = np.sqrt(self.Var_avg_layers[layer_no] + self.eps) 
            sigma1 = np.power(self.Var_avg_layers[layer_no] + self.eps, -0.5)                       # eq.31
            sigma2 = np.power(self.Var_avg_layers[layer_no] + self.eps, -1.5)                       # eq.32

            G1 = G * np.dot(sigma1, vector_ones_T)                                                  # eq.33 
            G2 = G * np.dot(sigma2, vector_ones_T)                                                  # eq.34
            D = self.S_layers[layer_no] - np.dot(self.Mu_avg_layers[layer_no], vector_ones_T)       # eq.35
            c = np.dot((G2 * D), vector_ones)                                                       # eq.36
            G = G1 - np.dot(np.dot(G1, vector_ones), vector_ones_T)/ n - (D * np.dot(c, vector_ones_T))/ n # eq.37

            return G
    
    #def GradientComparison_Analytical_Numerical(self, train_X_Norm, train_Y, param_list):
    def GradientComparison_Analytical_Numerical(self, dataset, param_list):
        # param_list = [lambda_cost, n_images, n_dimension, h, mu, nodes_layer, init_type, gradNumCheck, sigma_fix]
        # param_list = [lambda_cost, n_images, n_dimension, h, mu, hidden_layers, init_type, BN]
        lambda_cost = param_list[0]       # 0 but maybe 0.01 can be used
        n_images = param_list[1]          # 1 or 2, few images to test
                                          # network1.train_X.shape[1]
                                          # input size, for the input, it is equal to the number of the images, 
                                          # for the first layer, it's number of the nodes and so on
                                          # for the last layer, it's number of the nodes in the previous layer
        n_dimension = param_list[2]       # 20 ... small dimension for testing only, 
                                          # full dimension=network1.train_X.shape[0]=3072, takes too long to calculate
        self.h = param_list[3]        # 1e-5
        self.mu = param_list[4]       # 0        
        self.hidden_layers = param_list[5]       # nodes per hidden layer e.g. [50, 30] for 3-layer NN
        self.init_type = param_list[6]           # default='Xavier'  ... for Fixed sigma default=1e-2=0.01
        self.BN = param_list[7]       # default=True
        
        self.Initialize_W_b_ALL(dataset, n_dimension)
        
        X_batch = dataset.train_X_Norm[0:n_dimension, 0:n_images]
        Y_batch = dataset.train_Y[:, 0:n_images]
        #y_batch = network1.train_y[0:n_images]

        start_time = datetime.datetime.now()
        self.ComputeGradients(X_batch, Y_batch)
            
        Grads_NUMERICAL = self.ComputeGradsNumSlow_BN(X_batch, Y_batch)

        end_time = datetime.datetime.now()

        print("Calculation time of GradsNumSlow: " + str(end_time - start_time))

        for i in range(1, self.n_layers + 1):
            print('\n***** LAYER: {}'.format(i))
            print('W: {} .. W_numerical: {}'.format(self.grad_W[i].shape, Grads_NUMERICAL[0][i].shape))
            print('b: {} .. b_numerical: {}'.format(self.grad_b[i].shape, Grads_NUMERICAL[1][i].shape))
            self.CompareGradients(self.grad_W[i], Grads_NUMERICAL[0][i], 'W')
            self.CompareGradients(self.grad_b[i], Grads_NUMERICAL[1][i], 'b')
            
            if i < self.n_layers and self.BN:
                print('\ngamma: {} .. gamma_numerical: {}'.format(self.grad_gamma[i].shape, Grads_NUMERICAL[2][i].shape))
                print('beta: {} .. beta_numerical: {}'.format(self.grad_beta[i].shape, Grads_NUMERICAL[3][i].shape))
                self.CompareGradients(self.grad_gamma[i], Grads_NUMERICAL[2][i], grad_type = "gamma")
                self.CompareGradients(self.grad_beta[i], Grads_NUMERICAL[3][i], grad_type = "beta")

        return self.W_layers, self.b_layers, self.grad_W, self.grad_b, Grads_NUMERICAL
    
    def CompareGradients(self, gradAnalytic, gradNumerical, grad_type = "W"):
        print("\nGradient difference Comparisons for: {}".format(grad_type))
        grad_diff_mean = np.abs(gradAnalytic - gradNumerical).mean()
        grad_diff_min = np.abs(gradAnalytic - gradNumerical).min()
        grad_diff_max = np.abs(gradAnalytic - gradNumerical).max()
        
        #print("MEAN: {}\t\tMIN: {}\t\tMAX: {}\t\t".format(grad_diff_mean, grad_diff_min, grad_diff_max))
        print("MEAN: {}".format(grad_diff_mean))
    
    def EvaluationClassifier(self, X, training = False):  
        n_images = X.shape[1];  n_layers = self.n_layers;   alpha = self.alpha
        self.X_layers[0] = X    # for gradient calculations (training), we need X_layers to be returned
                                # BUT not for Cost calculations
        
        if self.BN:
            for i in range(1, n_layers):                                        # HIDDEN LAYERS
                self.S_layers[i] = np.dot(self.W_layers[i], self.X_layers[i - 1]) + self.b_layers[i]     # eq.5 & eq.9 & eq.12
                if training:
                    self.Mu_layers[i] = np.mean(self.S_layers[i], axis = 1, keepdims = True)             # eq.13
                    self.Var_layers[i] = np.var(self.S_layers[i], axis = 1, keepdims = True)             # eq.14

                    eps_vector = np.full((len(self.Var_layers[i]), 1), self.eps)                
                    var_eps = np.sqrt(self.Var_layers[i] + eps_vector)

                    self.S_bn_layers[i] = (self.S_layers[i] - self.Mu_layers[i])/var_eps                 # eq.6 & eq.15
                    # BELOW gives the same result as ABOVE, so I just use the above equation
#                 diag_Var = np.diagflat(self.Var_layers[i] + eps_vector);
#                 diag_var_sqrt_inv = np.linalg.inv(np.sqrt(diag_Var))
#                 self.S_bn_layers[i] = np.dot(diag_var_sqrt_inv, self.S_layers[i] - self.Mu_layers[i])

                    if self.Mu_avg_layers.get(i) is None:
                        self.Mu_avg_layers[i] = self.Mu_layers[i]
                    else:                                                                                # eq.38
                        self.Mu_avg_layers[i] = self.alpha * self.Mu_avg_layers[i] + (1 - self.alpha) * self.Mu_layers[i]

                    if self.Var_avg_layers.get(i) is None:
                        self.Var_avg_layers[i] = self.Var_layers[i]
                    else:                                                                                # eq.39
                        self.Var_avg_layers[i] = self.alpha * self.Var_avg_layers[i] + (1 - self.alpha) * self.Var_layers[i]
                else:
                    eps_vector = np.full((len(self.Var_avg_layers[i]), 1), self.eps)                
                    var_eps = np.sqrt(self.Var_avg_layers[i] + eps_vector)
                    self.S_bn_layers[i] = (self.S_layers[i] - self.Mu_avg_layers[i])/var_eps

                S_shift = self.gamma_layers[i] * self.S_bn_layers[i] + self.beta_layers[i]               # eq.7 & eq.16

                self.X_layers[i] = np.maximum(0, S_shift) # self.X_layers[i] = S_shift * (S_shift > 0)   # eq.8 & eq.17
      
            # FINAL layer
            S = np.dot(self.W_layers[n_layers], self.X_layers[n_layers - 1]) + self.b_layers[n_layers]   # eq.9 & eq.18
            exp_S = np.exp(S)  
            P = exp_S / np.sum(exp_S, axis=0)                                                           # eq.10 & eq.19 (SOFTMAX)

        else:
            for i in range(1, n_layers):                            # HIDDEN LAYERS
                S = np.dot(self.W_layers[i], self.X_layers[i - 1]) + self.b_layers[i] 
                self.X_layers[i] = np.maximum(0, S)
                     
            # FINAL layer 
            S = np.dot(self.W_layers[n_layers], self.X_layers[n_layers - 1]) + self.b_layers[n_layers]
            exp_s = np.exp(S)
            P = exp_s / np.sum(exp_s, axis=0)

        return P
    
    def Cost(self, X, Y, training = False, debug = False):
        # for k-layer NN FORWARD pass & COST calculations: Lecture4-Pg.33 (46)
        cost_sum = 0;                           n_images = Y.shape[1];
        
        P = self.EvaluationClassifier(X, training)   # training = True
        
#         for W_temp in self.W_layers:
#             cost_sum += np.power(W_temp, 2).sum()
            
        # If we don't pass the layer and make the calculations on self.W_layers[layer] but W_temp,
        # it gives different results... not for the accuracy but for the cost...
        for layer in range(1, len(self.W_layers) + 1):
            cost_sum += np.power(self.W_layers[layer], 2).sum()
        
        cross_entropy_loss = -np.log(np.dot(Y.T, P))          # Y.transpose()
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        total_loss = sum_cross_entropy_loss/n_images
        
        total_cost = total_loss + self.lambda_cost * cost_sum

        if debug:
            print('total_loss: {}, cost_sum: {}'.format(total_loss, cost_sum))
        
        return total_cost, total_loss
    
    def ComputeAccuracy(self, X, y):
        # k = predictions=the label with the max(probability) = Nx1
        # y = ground_truth_labels = Nx1
        # N = batch_length = number of images used
        P = self.EvaluationClassifier(X)
        k = np.argmax(P, axis=0)
        N = k.shape[0]
        return np.sum(k == y)/N
        
    def Coarse_Search(self, l_min, l_max, n_size):
        l_list = pow(10, np.random.uniform(l_min, l_max, n_size))
        # import math
        # pow(10, -2.345) = 0.0045185594437492215
        # round(math.log(0.0010933478237383542, 10), 4) = -2.345
        # 0.004768558843785876 - 0.004768558843785876*0.2
        return l_list
    
    def Fine_Search(self, l_min, l_max, n_size):
        lambda_fine = np.arange(l_min, l_max, (l_max - l_min)/n_size)
        return lambda_fine
    
    def Plot_Train_Validation_Cost_Accurracy(self, Cost_Train, Cost_Validation, Acc_Train, Acc_Validation, Eta):
        plt.rcParams['figure.figsize'] = (15.0, 5.0)
        plt.subplot(1,3,1)
        plt.plot(self.step_no_points, Cost_Train, 'g-', label='Train')
        if len(Cost_Validation) != 0:
            plt.plot(self.step_no_points, Cost_Validation, 'r-', label='Validation')
        plt.title('Cost Comparison')
        plt.xlabel('step_no')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid('on')
        
        plt.subplot(1,3,2)
        plt.plot(self.step_no_points, Acc_Train, 'g-', label='Train')
        if len(Acc_Validation) != 0:
            plt.plot(self.step_no_points, Acc_Validation, 'r-', label='Validation')
        plt.title('Accuracy Comparison')
        plt.xlabel('step_no')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid('on')
        
        plt.subplot(1,3,3)
        plt.plot(Eta, 'g-', label='Eta')
        plt.title('Eta Change')
        plt.xlabel('Time')
        plt.ylabel('Eta')
        plt.legend()
        plt.grid('on')

class Exercises:
    def GradientSanityCheck(self, network1, dataset, param_list, BN = True):
        # param_list = [n_batch_size, n_epocs, lambda_cost, eta, hidden_layers, n_sanity_batch, init_type, alpha, debug_on, n_dimension] 
        # [100, 400, 0.005, 0.001, [50, 50, 30], 1, 'He', 0.9, False, 20] 
        n_batch_size = param_list[0];    n_epocs = param_list[1];                 network1.lambda_cost = param_list[2]
        eta = param_list[3];             network1.hidden_layers = param_list[4];
        n_sanity_batch = param_list[5];  network1.init_type = param_list[6];      alpha = param_list[7];
        debug_on = param_list[8];        n_dimension= param_list[9]
        
        network1.BN = BN
        network1.Initialize_W_b_ALL(dataset);             self.alpha = alpha;   n_layers = network1.n_layers
                   
        #n_dimension = dataset.train_X_Norm.shape[0];              # 32x32x3=3072     
        n_total_images = n_batch_size * n_sanity_batch
        
        Accuracy_epocs_train = np.zeros(n_epocs)  # accuracy array - will keep accuracy per epoc (iteration)

        J_train_sum = 0; cost_record = 0; J_epocs_train = []; L_epocs_train = []
        start_time = datetime.datetime.now()
        for e in range(n_epocs):
            for batch in range(n_sanity_batch):
                index_list = list(range(batch * n_batch_size, (batch + 1) * n_batch_size))
                X_batch = dataset.train_X_Norm[0:n_dimension, index_list]
                Y_batch = dataset.train_Y[:, index_list]
                y_batch = dataset.train_y[index_list]
                
                network1.ComputeGradients(X_batch, Y_batch)

                # go through hyperparamaters in all layers
                for layer in range(1, n_layers + 1):
                    network1.W_layers[layer] -= eta * network1.grad_W[layer]
                    network1.b_layers[layer] -= eta * network1.grad_b[layer]
                    
                    if layer < n_layers and network1.BN:
                    # Mu and Var exist in k-1 layers
                        network1.gamma_layers[layer] -= eta * network1.grad_gamma[layer]
                        network1.beta_layers[layer] -= eta * network1.grad_beta[layer]
            
            J_train, L_train = network1.Cost(X_batch, Y_batch, debug = False)

            ### NOTE: If the total images are used, the cost goes down to 1.24
#             J_train, _ = network1.Cost(train_X_Norm[0:n_dimension, 0:n_total_images], 
#                             dataset.train_Y[:, 0:n_total_images], NetParams, lambda_cost, training = False)
#             J_epocs_train[e] = J_train
            
            ### NOTE: If the smooth cost is used with X_batch, the cost goes down to 1.64 ONLY
            J_train_sum += J_train
            smooth_cost = J_train_sum/(cost_record + 1)
            cost_record += 1
            J_epocs_train.append(smooth_cost)

            A_train = network1.ComputeAccuracy(dataset.train_X_Norm[0:n_dimension, 0:n_total_images], dataset.train_y[0:n_total_images])
            
            Accuracy_epocs_train[e] = A_train
            network1.step_no_points.append(e)
            
            if debug_on:
                print('******************   epoch_no: {}   *******************************'.format(e))
                print('J_train: {} ... L_train: {} ... smooth_cost : {} ... A_train: {}'.format(
                    round(J_train, 6), round(L_train, 6), round(smooth_cost, 6), round(A_train, 6)))

            #grad.Plot_Train_Validation_Cost_Accurracy(J_epocs_train, Accuracy_epocs_train)

        end_time = datetime.datetime.now()
        
        ## d = 20,   N=100    >>> Cost from 2.54  to 2.38    >>> Accuracy from 0.08 to 0.15    >> time: 0.15 seconds
        ## d = 3072, N=100    >>> Cost from 2.439 to 1.199   >>> Accuracy from 0.15 to 0.74    >> time: 1.23 seconds
        ## d = 3072, N=10000  >>> Cost from 2.347 to 1.323   >>> Accuracy from 0.1906 to 0.543 >> time: 3.08 minutes

#         print(J_epocs_train)
#         print(Accuracy_epocs_train)
        
        print("Calculation time of GradientSanityCheck: {} ... finished at: {}".format(end_time - start_time, datetime.datetime.now()))

        network1.Plot_Train_Validation_Cost_Accurracy(J_epocs_train, [], Accuracy_epocs_train, [], [])

    def Train_Cyclical_BN(self, network1, dataset, param_list, BN = True):
        n_batch_size = param_list[0];    n_cycles = param_list[1];     network1.lambda_cost = param_list[2]
        eta_min = param_list[3];         eta_max = param_list[4];      network1.hidden_layers = param_list[5];
        k_cycle = param_list[6];         network1.init_type = param_list[7];    alpha = param_list[8];
        total_records = param_list[9] + 1
                   
        n_epochs = 2 * k_cycle * n_cycles
        n_steps = k_cycle * dataset.train_X.shape[1] / n_batch_size
        J_train_sum = 0;    L_train_sum = 0;     cost_record = 0
        step_no = 0;        cycle_no = 0;        n_images = dataset.train_X.shape[1]

        network1.BN = BN;
        network1.Initialize_W_b_ALL(dataset);             network1.alpha = alpha;   n_layers = network1.n_layers

        J_epocs_train = np.zeros(total_records);           L_epocs_train = np.zeros(total_records)
        Accuracy_epocs_train = np.zeros(total_records)     # accuracy array
        J_epocs_val = np.zeros(total_records);             L_epocs_val = np.zeros(total_records)
        Accuracy_epocs_val = np.zeros(total_records);      eta_train = []

        n_total_batch = n_images // n_batch_size    # per epoch
        plot_points = n_total_batch * n_epochs/(total_records - 1)

        np.random.seed(400)
        cost_record = 0
        for e in range(n_epochs):
            shuffled_idx = np.random.permutation(n_images)
            for id in range(n_images // n_batch_size):                
                batch_range = range(id * n_batch_size, ((id + 1) * n_batch_size))
                batch_idx = shuffled_idx[batch_range]
                
                X_batch = dataset.train_X_Norm[:, batch_idx]
                Y_batch = dataset.train_Y[:, batch_idx]

                network1.ComputeGradients(X_batch, Y_batch)
                
                if (step_no >= 2 * cycle_no * n_steps) & (step_no <= (2 * cycle_no + 1) * n_steps):
                    eta = eta_min + (step_no - 2 * cycle_no * n_steps) / n_steps * (eta_max - eta_min)
                elif (step_no >= (2 * cycle_no + 1) * n_steps) & (step_no <= 2 * (cycle_no + 1) * n_steps):
                    eta = eta_max - (step_no - (2 * cycle_no + 1) * n_steps) / n_steps * (eta_max - eta_min)

                eta_train.append(eta)

                for layer in range(1, n_layers + 1):
                    network1.W_layers[layer] -= eta * network1.grad_W[layer]
                    network1.b_layers[layer] -= eta * network1.grad_b[layer]
                    
                    if layer < n_layers and network1.BN:               # Mu and Var exist in k-1 layers
                        network1.gamma_layers[layer] -= eta * network1.grad_gamma[layer]
                        network1.beta_layers[layer] -= eta * network1.grad_beta[layer]

                if step_no % plot_points == 0 or step_no == n_total_batch * n_epochs - 1:
                    # ****** TRAINING ******
                    J_train, L_train = network1.Cost(X_batch, Y_batch)

                    J_train_sum += J_train;                        L_train_sum += L_train
                    smooth_cost = J_train_sum/(cost_record + 1);   smooth_loss = L_train_sum/(cost_record + 1)
                    J_epocs_train[cost_record] = smooth_cost;      L_epocs_train[cost_record] = smooth_loss

                    A_train = network1.ComputeAccuracy(dataset.train_X_Norm, dataset.train_y)
                    Accuracy_epocs_train[cost_record] = A_train

                    # ****** VALIDATION ******
                    J_val, L_val = network1.Cost(dataset.validation_X_Norm, dataset.validation_Y) #

                    J_epocs_val[cost_record] = J_val;            L_epocs_val[cost_record] = L_val

                    A_validation = network1.ComputeAccuracy(dataset.validation_X_Norm, dataset.validation_y)
                    Accuracy_epocs_val[cost_record] = A_validation
                    print("e: {} ... step_no: {} .. cost_record: {} ... time: {}".format(e, step_no, 
                                                                                         cost_record, datetime.datetime.now()))
                    print('J_train: {} ... smooth_cost : {} ... J_validation: {}'.format(J_train, smooth_cost, J_val))
                    cost_record += 1;      network1.step_no_points.append(step_no)
                
                step_no += 1

                if (step_no ) % (2 * n_steps) == 0:
                        cycle_no += 1 

        network1.Plot_Train_Validation_Cost_Accurracy(J_epocs_train, J_epocs_val, Accuracy_epocs_train, 
                                             Accuracy_epocs_val, eta_train)

        NetworkResults_final = (J_epocs_train, J_epocs_val, Accuracy_epocs_train, Accuracy_epocs_val)

        return NetworkResults_final

    
##### NOTE: Below part should be executed on a separate Jupyter Notebook (each part in a different cell)
"""
import numpy as np
import assignment3
import dataset_cifar

######################################################### Part1: GRADIENT COMPARISON
dataset = dataset_cifar.CIFAR_IMAGES()
network1 = assignment3.Network()
assignment3.load_data(dataset, data_type = 'Validation')

#param_list = [lambda_cost, n_images, n_dimension, h, mu, hidden_layers, init_type, BN]
param_list = [0, 3, 10, 1e-5, 0, [50, 50], 1e-2, True]

W_layers, b_layers, grad_W, grad_b, Grads_NUMERICAL = \
           network1.GradientComparison_Analytical_Numerical(dataset, param_list)

######################################################### Part2: GRADIENT Sanity Check
dataset = dataset_cifar.CIFAR_IMAGES()
assignment3.load_data(dataset, data_type = 'Validation')
# param_list = [n_batch_size, n_epocs, lambda_cost, eta, hidden_layers, n_sanity_batch, 
#                                                                       init_type, alpha, debug_on, n_dimension] 
param_list = [100, 200, 0.005, 0.1, [50, 50], 1, 'He', 0.9, False, 3072]
network1 = assignment3.Network()
exercise1 = assignment3.Exercises()
exercise1.GradientSanityCheck(network1, dataset, param_list, BN = True)

######################################################### Part3: TRAIN CYCLICAL With and Without BN
dataset = dataset_cifar.CIFAR_IMAGES()
assignment3.load_data(dataset, data_type = 'Test')
#param_list = [n_batch_size, n_cycles, lambda_cost, eta_min, eta_max, hidden_layers, k_cycle, init_type, 
#                                                                                           alpha, plot_points]
param_list = [100, 2, 0.005, 1e-5, 1e-1, [50, 50], 5, 'He', 0.9, 10]
# param_list = [100, 2, 0.005, 1e-5, 1e-1, [50, 30, 20, 20, 10, 10, 10, 10], 5, 'He', 0.9, 10]
network1 = assignment3.Network()
exercise1 = assignment3.Exercises()

NetworkResults_final = exercise1.Train_Cyclical_BN(network1, dataset, param_list, BN = True)

A_train = network1.ComputeAccuracy(dataset.train_X_Norm, dataset.train_y)
A_validation = network1.ComputeAccuracy(dataset.validation_X_Norm, dataset.validation_y)
A_test = network1.ComputeAccuracy(dataset.test_X_Norm, dataset.test_y)

print('A_train: \t{} \nA_validation: \t{} \nA_test: \t{}'.format(round(A_train, 4), A_validation, A_test))

######################################################### Part4: LAMBDA COARSE SEARCH
coarse_results = {}
#coarse_list = network1.Coarse_Search(-5, -1, 5)
coarse_list = network1.Coarse_Search(-2.9612, -1.6133, 5)

for lambda_c in coarse_list:
    network1 = assignment3.Network()
    exercise1 = assignment3.Exercises()
    param_list = [100, 2, lambda_c, 1e-5, 1e-1, [50, 50], 5, 'He', 0.9, 10]

    NetworkResults_final = exercise1.Train_Cyclical_BN(network1, dataset, param_list, BN = True)

    A_test = network1.ComputeAccuracy(dataset.test_X_Norm, dataset.test_y)
    coarse_results[lambda_c] = A_test

# import operator
# sorted_coarse_results = sorted(coarse_results.items(), key=operator.itemgetter(0))
sorted_coarse_results = sorted(coarse_results.items(), key=lambda kv: kv[1])
sorted_coarse_results

######################################################### Part5: LAMBDA FINE SEARCH
fine_results = {}
fine_list = network1.Fine_Search(0.0041, 0.0081, 20)

for lambda_c in fine_list:
    network1 = assignment3.Network()
    exercise1 = assignment3.Exercises()
    param_list = [100, 2, lambda_c, 1e-5, 1e-1, [50, 50], 5, 'He', 0.9, 10]

    NetworkResults_final = exercise1.Train_Cyclical_BN(network1, dataset, param_list, BN = True)

    A_test = network1.ComputeAccuracy(dataset.test_X_Norm, dataset.test_y)
    fine_results[lambda_c] = A_test

sorted_fine_results = sorted(fine_results.items(), key=lambda kv: kv[1])
sorted_fine_results

######################################################### Part6: Sensitivity to Initialization
network1 = assignment3.Network()
exercise1 = assignment3.Exercises()
#param_list = [100, 2, 0.005, 1e-5, 1e-1, [50, 30, 20, 20, 10, 10, 10, 10], 5, 1e-4, 0.9, 10]
param_list = [100, 2, 0.005, 1e-5, 1e-1, [50, 50], 2, 1e-4, 0.9, 10]

NetworkResults_final = exercise1.Train_Cyclical_BN(network1, dataset, param_list, BN = False)

A_train = network1.ComputeAccuracy(dataset.train_X_Norm, dataset.train_y)
A_validation = network1.ComputeAccuracy(dataset.validation_X_Norm, dataset.validation_y)
A_test = network1.ComputeAccuracy(dataset.test_X_Norm, dataset.test_y)

print('A_train: \t{} \nA_validation: \t{} \nA_test: \t{}'.format(round(A_train, 4), A_validation, A_test))

"""

"""
########################################################## This part is needed to be saved as a .py file with the name "dataset_cifar.py" in the same folder with Assignment3.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing

class CIFAR_IMAGES:
    def __init__(self):
        self.label_size = 10
        self.image_size = 32 * 32 * 3
        #for one_hot_encoding
        self.label_encoder = preprocessing.LabelBinarizer()
        # file path of the images on your laptop
        self.filePath = 'Dataset/data_batch_1'
        self.unique_labels = []
    
    # below function corresponds to [X, Y, y] = LoadBatch(filename) in the assignment, 1st task.
    def load_batch_a1(self, filePathLocal):
        with open(filePathLocal, 'rb') as fileToOpen:
            dictBatch = pickle.load(fileToOpen, encoding='bytes')
            Y_one_hot_labels = self.label_encoder.fit_transform(dictBatch[b'labels'])
        #return [X, Y, y]
        return [np.divide(dictBatch[b'data'], 255).transpose(), Y_one_hot_labels.transpose(), np.array(dictBatch[b'labels'])]
           
        
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
    
    def load_meta(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_batch(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X = dict[b'data'].T
            y = dict[b'labels']
            Y = np.eye(10)[y].T
        return X, Y, y

    def normalization(self, X, X_train, type_norm):
        if type_norm == 'z-score':
            mean_X = np.mean(X_train, axis=1, keepdims=True)
            std_X = np.std(X_train, axis=1, keepdims=True)
            X = (X-mean_X)/std_X
        if type_norm == 'min-max':
            min_X = np.min(X_train, axis=1, keepdims=True)
            max_X = np.max(X_train, axis=1, keepdims=True)
            X = (X-min_X)/(max_X-min_X)
        return X
    
    def NormalizeData_Per_DataSet(self, inputData):
        mean_X = np.mean(inputData, axis=1, keepdims=True)
        std_X = np.std(inputData, axis=1, keepdims=True)
        normalizedInputData = (inputData - mean_X)/std_X

        return normalizedInputData
    
    def Normalize_ALL(self):
        self.train_X_Norm = self.NormalizeData_Per_DataSet(self.train_X)
        self.validation_X_Norm = self.NormalizeData_Per_DataSet(self.validation_X)
        self.test_X_Norm = self.NormalizeData_Per_DataSet(self.test_X)
        
    # Used to transfer a python model to matlab
    def save_as_mat(data, name="model"):
        #import scipy.io as sio
        sio.savemat(name+'.mat',{name:b})
 """