import numpy as np
import matplotlib.pyplot as plt
import gradient

### MEL
### !!! using from requires extra attention if you use circular dependencies !!!
#from gradient import Gradient

class Linear:
    def __init__(self):
        self.gradLinear = gradient.Gradient()
        self.lambda_cost = 0 # wight regularization
    
    def Forward(self, X, W, b):
        # EvaluationClassifier
        # W = (Kxd) size, randomly initialized weights > then this will be updated by each batch
        # X = each column of X corresponds to an image and it has size (dxn) >> here n will be smaller since 
        # it will be selected as subset of images n=100 can be selected
        # b = (Kx1) size, randomly initialized bias > then this will be updated by each batch
        #print('b.shape: ' + str(b.shape))
        #print('X.shape: ' + str(X.shape))
        b_broadcast = np.tile(b, (1, X.shape[1]))
        #b_broadcast = np.broadcast_to(b, (b.shape[0], X.shape[1]))
        #print('b_broadcast.shape: ' + str(b_broadcast.shape))
        s = np.dot(W, X) + b_broadcast
        # p = probabilities of each class to the corresponding images
        # p = self.softmax(s)
        return s
    
    # MEL
    # use the Gradient class, once it works, remove this
    def ComputeGradients(self, Y, P, X, lambda_cost, W):
        # Y = ground_truth_labels_matrix
        # P = probabilities
            # X = image data
        G = -np.subtract(Y, P)
        N = Y.shape[1]  # number of images in data (X)

        dL_dW = np.divide(np.dot(G, X.transpose()), N)
        dL_dB = np.divide(np.sum(G, axis=1), N)

        # grad_W = dJ_dW    ...   grad_b = dJ_db
        grad_W = dL_dW + 2 * lambda_cost * W
        grad_b = dL_dB
        
    def Backward(self, N, G, X, lambda_cost, W, eta, layer_type='hidden'):
        #P = self.EvaluationClassifier(X, W, b)

        #n_batch = GDparams[0]  # e.g. n_batch=100
        #eta = GDparams[0]      # e.g. eta=0.001
        #n_epocs = GDparams[1]    # e.g. epocs=20
        
        if layer_type == 'hidden':
            (grad_W, grad_b, G) = self.gradLinear.ComputeGradients_Linear_HiddenLayer(N, G, X, lambda_cost, W2)
            #(grad_W, grad_b, G) = ComputeGradients_Linear_HiddenLayer(N, G, H, lambda_cost, W2)
        else:
            # means first layer
            (grad_W, grad_b) = self.gradLinear.ComputeGradients_Linear_FirstLayer(N, G, X, lambda_cost, W1)
            #(grad_W, grad_b) = ComputeGradients_Linear_FirstLayer(N, G, X_batch, lambda_cost, W1)      
        
        Wstar = W - eta * grad_W
        bstar_m = b - eta * grad_b
        bstar = bstar_m[:, :1]
        
        return (Wstar, bstar, G)
      
    def Cost(self, lambda_cost, W):
        return lambda_cost * np.power(W, 2).sum()
    
class ReLU:
    def __init__(self):
        self.gradReLU = gradient.Gradient()
        
    def Forward(self, s1):
        h = s1 * (s1 > 0)
        # h = max(0, s1) ... since s1 will return 1 (true) for values >0 and 0 for others, 
        # s1 * (s1 > 0) check is sufficient for this operation
        return h
                
    def Backward(self, G, H):        
        return self.gradReLU.ComputeGradients_ReLU(G, H)
    
class Softmax:
    def __init__(self):
        self.gradSoftmax = gradient.Gradient()
        
    def Forward(self, s):
        """ Standard definition of the softmax function """
        # think of making it as a STATIC method, @staticmethod
        exp_s = np.exp(s)
        p = exp_s / np.sum(exp_s, axis=0)
        return p

    #def Backward(self, Y, P):
        # Only passes back Y and P so no need to add this to the calculations
        
    def Cost(self, X, Y_matrix, b):
        # b = b_last # b at the last layer
        # Y_matrix = ground_truth_labels_matrix
        # P = probabilities
        b_broadcast = np.tile(b, (1, X.shape[1]))
        s = np.dot(W, X) + b_broadcast
        # P = probabilities of each class to the corresponding images
        P = self.softmax(s)
        # MEL
        # l_cross = 1 / n* np.sum(-np.log(np.sum(Y*P,axis=0)))
        # l_cross = -np.mean(....)
        cross_entropy_loss = -np.log(np.dot(Y_matrix.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y_matrix.shape[1]
        return sum_cross_entropy_loss/N
    
    def Cost_P(self, Y_matrix, P):
        cross_entropy_loss = -np.log(np.dot(Y_matrix.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y_matrix.shape[1]
        return sum_cross_entropy_loss/N
        