import numpy as np
import matplotlib.pyplot as plt
import layer
import network

import datetime
import time

class Gradient:
    def ComputeGradients_Linear_HiddenLayer(self, N, G, H, lambda_cost, W):
        # In assignment1, we used a bit different = ComputeGradients(self, Y, P, X, lambda_cost, W):
        # Y = ground_truth_labels_matrix
        # P = probabilities
        # X = image data
        # let's provide G to the function, so it will slightly be different than assisgnment1
        # G = -np.subtract(Y, P)
        # Also instead of providing Y, now we can provide N only
        # N = Y.shape[1]  # number of images in data (X)

        dL_dW = np.divide(np.dot(G, H.transpose()), N)
        dL_dB = np.divide(np.sum(G, axis=1), N)

        # grad_W = dJ_dW    ...   grad_b = dJ_db
        grad_W = dL_dW + 2 * lambda_cost * W
        grad_b = dL_dB
        
        # MEL
        # g = gW2          >>> pg. 29(42) - Lecture4.pdf
        # OR as below? 
        # np.dot(W.T, G)   >>> pg. 32(45) - Lecture4.pdf (Gbatch = W2.T@Gbatch)
        # since we use matrix notation here, I think we should go for the below one. check the matrix sizes after the execution
        G = np.dot(W.T, G)
        
        return (grad_W, grad_b, G)
    
    def ComputeGradients_Linear_FirstLayer(self, N, G, X, lambda_cost, W):
        # In assignment1, we used a bit different = ComputeGradients(self, Y, P, X, lambda_cost, W):
        # Y = ground_truth_labels_matrix
        # P = probabilities
        # X = image data
        # let's provide G to the function, so it will slightly be different than assisgnment1
        # G = -np.subtract(Y, P)
        # Also instead of providing Y, now we can provide N only
        # N = Y.shape[1]  # number of images in data (X)
        
        dL_dW = np.divide(np.dot(G, X.transpose()), N)
        dL_dB = np.divide(np.sum(G, axis=1), N)

        # grad_W = dJ_dW    ...   grad_b = dJ_db
        grad_W = dL_dW + 2 * lambda_cost * W
        grad_b = dL_dB
        
        return (grad_W, grad_b)
        
    def ComputeGradients_ReLU(self, G, H):
        G = G * (H > 0)       
        return G     
        
    def ComputeGradsNumSlow(self, X, Y, W, b, lambda_cost, h=1e-5):
        # W = (W1, W2)
        # b = (b1, b2)
        # X = images
        # Y = true labels of images
        # Someone has shared the python version of the code for ComputeGradsNum, I slightly modified it and using
        grad_W = [np.zeros(w.shape) for w in W]
        grad_b = [np.zeros(bi.shape) for bi in b]
        
        nw = network.Network()

        #for layer_no in [0, 1]:
        for layer_no in [0, 1]:
            #print('layer_no: {}\t datetime: {}'.format(layer_no, datetime.datetime.now()))
            for i in range(b[layer_no].shape[0]):
                Costs = []
                #print('i: {}\t datetime: {}'.format(i, datetime.datetime.now()))
                for m in [-1, 1]:
                    #print('m: {}\t datetime: {}'.format(m, datetime.datetime.now()))
                    b_try = [bi.copy() for bi in b]
                    b_try[layer_no][i] += m * h
                    Costs.append(nw.Cost(X, Y, W, b_try, lambda_cost))
                    #Costs.append(nw.Cost(layers, X, Y, W, b_try[-1], lambda_cost))
                    #Costs.append(nw.Cost(X, Y, W, b_try[-1], lambda_cost))
                    
                grad_b[layer_no][i] = (Costs[1] - Costs[0]) / (2 * h)

                for j in range(W[layer_no].shape[1]):
                        Costs = []
                        #print('j: {}\t datetime: {}'.format(j, datetime.datetime.now()))
                        for m in [-1, 1]:
                            #print('m2: {}\t datetime: {}'.format(m, datetime.datetime.now()))
                            W_try = [w.copy() for w in W]
                            W_try[layer_no][i, j] += m * h
                            Costs.append(nw.Cost(X, Y, W_try, b, lambda_cost))
                            #Costs.append(nw.Cost(layers, X, Y, W_try, b[-1], lambda_cost))
                            #Costs.append(nw.Cost(X, Y, W_try, b[-1], lambda_cost))
                            
                        grad_W[layer_no][i, j] = (Costs[1] - Costs[0]) / (2 * h)
                    
        return grad_W, grad_b 
    
    def ComputeGradsNumSlow_ex(self, layers, X, Y, W, b, lambda_cost, h=1e-5):
        # W = (W1, W2)
        # b = (b1, b2)
        # X = images
        # Y = true labels of images
        # Someone has shared the python version of the code for ComputeGradsNum, I slightly modified it and using
        grad_W = [np.zeros(w.shape) for w in W]
        grad_b = [np.zeros(bi.shape) for bi in b]
        
        nw = network.Network()

        #for layer_no in [0, 1]:
        for layer_no in range(len(layers)):
            for i in range(b[layer_no].shape[0]):
                Costs = []
                for m in [-1, 1]:
                    b_try = [bi.copy() for bi in b]
                    b_try[layer_no][i] += m * h
                    Costs.append(nw.Cost(layers, X, Y, W, b_try, lambda_cost))
                    #Costs.append(nw.Cost(layers, X, Y, W, b_try[-1], lambda_cost))
                    #Costs.append(nw.Cost(X, Y, W, b_try[-1], lambda_cost))
                    
                grad_b[layer_no][i] = (Costs[1] - Costs[0]) / (2 * h)

                for j in range(W[layer_no].shape[1]):
                        Costs = []
                        for m in [-1, 1]:
                            W_try = [w.copy() for w in W]
                            W_try[layer_no][i, j] += m * h
                            Costs.append(nw.Cost(layers, X, Y, W_try, b, lambda_cost))
                            #Costs.append(nw.Cost(layers, X, Y, W_try, b[-1], lambda_cost))
                            #Costs.append(nw.Cost(X, Y, W_try, b[-1], lambda_cost))
                            
                        grad_W[layer_no][i, j] = (Costs[1] - Costs[0]) / (2 * h)
                    
        return grad_W, grad_b 
    
    def ComputeGradsNum(self, layers, X, Y, W, b, lambda_cost, h=1e-5):
        # W = (W1, W2)
        # b = (b1, b2)
        # X = images
        # Y = true labels of images
        # Someone has shared the python version of the code for ComputeGradsNum, I slightly modified it and using
        grad_W = [np.zeros(w.shape) for w in W]
        grad_b = [np.zeros(bi.shape) for bi in b]
        
        nw = network()
        c = nw.Cost(layers, X, Y, W, b, lambda_cost)
        Costs = [c]

        for layer_no in range(len(layers)):
            for i in range(b[layer_no].shape[0]):
                Costs = []
                b_try = [bi.copy() for bi in b]
                b_try[layer_no][i] += h
                Costs.append(nw.Cost(layers, X, Y, W, b_try, lambda_cost))
                    
                grad_b[layer_no][i] = (Costs[1] - Costs[0]) / (2 * h)

                for j in range(W[layer_no].shape[1]):
                        W_try = [w.copy() for w in W]
                        W_try[layer_no][i, j] += m * h
                        Costs.append(nw.Cost(layers, X, Y, W_try, b, lambda_cost))
                            
                        grad_W[layer_no][i, j] = (Costs[1] - Costs[0]) / (2 * h)
                    
        return grad_W, grad_b
    
    def CompareGradients_W(self, gradAnalytic, gradNumerical):
        #print(train_X_batch.shape)
        #print(grad_W_num.shape)
        #print(grad_b_num.shape) # grad_b_num.shape=(10, 1)     grad_b.shape=(10,)
        #print("sum_grad_b_difference = " + str(np.abs(grad_b - grad_b_num.transpose()).sum()))
        
        ######### W #######
        print("\ngrad_W_difference_MEAN = "+ str(np.abs(gradAnalytic - gradNumerical).mean()))
        print("grad_W_difference_MIN = "+ str(np.abs(gradAnalytic - gradNumerical).min()))
        print("grad_W_difference_MAX = "+ str(np.abs(gradAnalytic - gradNumerical).max()))

        #print("sum_grad_W_difference = " + str(np.abs(grad_W - grad_W_num).sum()))

        print("\ngrad_W_MIN = " + str(np.min(np.abs(gradAnalytic))))
        print("grad_W_num_MIN = " + str(np.min(np.abs(gradNumerical))))
        print("grad_W_MAX = " + str(np.max(np.abs(gradAnalytic))))
        print("grad_W_num_MAX = " + str(np.max(np.abs(gradNumerical))))
        
    def CompareGradients_b(self, gradAnalytic, gradNumerical):
        #print(train_X_batch.shape)
        #print(grad_W_num.shape)
        #print(grad_b_num.shape) # grad_b_num.shape=(10, 1)     grad_b.shape=(10,)
        #print("sum_grad_b_difference = " + str(np.abs(grad_b - grad_b_num.transpose()).sum()))
        
        ######### b #######
        print("\ngrad_b_difference_MEAN = "+ str(np.abs(gradAnalytic - gradNumerical.transpose()).mean()))
        print("grad_b_difference_MIN = "+ str(np.abs(gradAnalytic - gradNumerical.transpose()).min()))
        print("grad_b_difference_MAX = "+ str(np.abs(gradAnalytic - gradNumerical.transpose()).max()))
        print("\ngrad_b_MIN = " + str(min(np.abs(gradAnalytic))))
        print("grad_b_num_MIN = " + str(min(np.abs(gradNumerical))))
        print("\ngrad_b_MAX = " + str(max(np.abs(gradAnalytic))))
        print("grad_b_num_MAX = " + str(max(np.abs(gradNumerical))))