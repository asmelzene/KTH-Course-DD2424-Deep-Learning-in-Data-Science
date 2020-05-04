import numpy as np
import matplotlib.pyplot as plt

class COMPUTATIONS:
    def softmax(self, s):
        """ Standard definition of the softmax function """
        return np.exp(s) / np.sum(np.exp(s), axis=0)
    
    def ComputeGradsNum(self, X, Y, P, W, b, lamda, h=1e-6):
        """ Converted from matlab code """
        K = W.shape[0]
        d = X.shape[0]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros((K, 1))

        c = self.ComputeCost(Y, X, b, W, lamda);
        #c = ComputeCost(X, Y, W, b, lamda);  # provided by canvas
        #c = ComputeCost_2(Y, X, b, W, lambda)

        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(Y, X, b_try, W, lamda)
            # # provided by canvas
            #c2 = ComputeCost_2(Y, X, b_try, W, lambda)
            grad_b[i] = (c2-c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] += h
                #c2 = ComputeCost(X, Y, W_try, b, lamda) # provided by canvas
                c2 = self.ComputeCost(Y, X, b, W_try, lamda)
                #c2 = ComputeCost_2(Y, X, b, W_try, lambda)
                grad_W[i,j] = (c2-c) / h

        return [grad_W, grad_b]

    def ComputeGradsNumSlow(self, X, Y, P, W, b, lamda, h=1e-6):
        """ Converted from matlab code """
        K = W.shape[0]
        d = X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((K, 1));

        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] -= h
            c1 = self.ComputeCost(Y, X, b_try, W, lamda)
            #c1 = ComputeCost(X, Y, W, b_try, lamda) # provided by canvas

            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(Y, X, b_try, W, lamda)
            #c2 = ComputeCost(X, Y, W, b_try, lamda) # provided by canvas

            grad_b[i] = (c2-c1) / (2*h)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] -= h
                c1 = self.ComputeCost(Y, X, b, W_try, lamda)
                #c1 = ComputeCost(X, Y, W_try, b, lamda) # provided by canvas

                W_try = np.array(W)
                W_try[i,j] += h
                c2 = self.ComputeCost(Y, X, b, W_try, lamda) 
                #c2 = ComputeCost(X, Y, W_try, b, lamda) # provided by canvas

                grad_W[i,j] = (c2-c1) / (2*h)

        return [grad_W, grad_b]
    
    def EvaluationClassifier(self, X, W, b):
        # W = (Kxd) size, randomly initialized weights
        # X = each column of X corresponds to an image and it has size (dxn) >> here n will be smaller since 
        # it will be selected as subset of images n=100 can be selected
        # b = (Kx1) size, randomly initialized bias
        #print('b.shape: ' + str(b.shape))
        #print('X.shape: ' + str(X.shape))
        b_broadcast = np.tile(b, (1, X.shape[1]))
        #b_broadcast = np.broadcast_to(b, (b.shape[0], X.shape[1]))
        #print('b_broadcast.shape: ' + str(b_broadcast.shape))
        s = np.dot(W, X) + b_broadcast
        # p = probabilities of each class to the corresponding images
        p = self.softmax(s)
        return p
    
    def ComputeCost(self, Y_matrix, P, W, lambda_cost):
        # Y_matrix = ground_truth_labels_matrix
        # P = probabilities
        cross_entropy_loss = -np.log(np.dot(Y_matrix.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y_matrix.shape[1]
        return sum_cross_entropy_loss/N + lambda_cost*np.power(W, 2).sum() 
    
    def ComputeCost(self, Y_matrix, X, b, W, lambda_cost):
        b_broadcast = np.tile(b, (1, X.shape[1]))
        s = np.dot(W, X) + b_broadcast
        # P = probabilities of each class to the corresponding images
        P = self.softmax(s)
        
        # Y_matrix = ground_truth_labels_matrix
        # P = probabilities
        cross_entropy_loss = -np.log(np.dot(Y_matrix.transpose(), P))
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        N = Y_matrix.shape[1]
        return sum_cross_entropy_loss/N + lambda_cost*np.power(W, 2).sum()
    
    def ComputeAccuracy(self, k, y):
        # k = predictions=the label with the max(probability) = Nx1
        # y = ground_truth_labels = Nx1
        # N = batch_length = number of images used
        N = k.shape[0]
        return np.sum(k == y)/N
    
    def ComputeGradients(self, Y, P, X, lambda_cost, W):
        # Y = ground_truth_labels_matrix
        # P = probabilities
        # X = image data
        G = - np.subtract(Y, P)
        N = Y.shape[1]

        dL_dW = np.divide(np.dot(G, X.transpose()), N)
        dL_dB = np.divide(np.sum(G, axis=1), N)

        # grad_W = dJ_dW    ...   grad_b = dJ_db
        grad_W = dL_dW + 2 * lambda_cost * W
        grad_b = dL_dB
        return [grad_W, grad_b]
    
    def MiniBatchGD(self, X, Y, GDparams, W, b, lambda_cost):
        P = self.EvaluationClassifier(X, W, b)

        n_batch = GDparams[0]  # e.g. n_batch=100
        eta = GDparams[1]      # e.g. eta=0.001
        n_epocs = GDparams[2]    # e.g. epocs=20
        
        [grad_W, grad_b] = self.ComputeGradients(Y, P, X, lambda_cost, W)
        
        Wstar = W - eta * grad_W
        bstar_m = b - eta * grad_b
        bstar = bstar_m[:, :1]
        
        return [Wstar, bstar]
    
    def CostCalculations(self, GDparams, TrainData, ValidationData, TestData):
        #GDparams = [n_batch, eta, n_epocs, lambda_cost]
        #TrainData = [train_X_normalized, train_Y, train_y]
        #ValidationData = [validation_X_normalized, validation_Y, validation_y]
        
        lambda_cost = GDparams[3] # lambda_cost=0; lambda_cost=0; lambda_cost=0.1; lambda_cost=1; 
        n_epocs = GDparams[2] # n_epocs=40; n_epocs=40; n_epocs=40; n_epocs=40;
        
        eta = GDparams[1] # eta=0.1; eta=0.001; eta=0.001; eta=0.001;
        # n_batch: the size of the batch, in other words, the number of images in each batch
        n_batch = GDparams[0] # n_batch=100; n_batch=100; n_batch=100; n_batch=100;
        
        train_X_normalized = TrainData[0]
        train_Y = TrainData[1]
        train_y = TrainData[2]
        
        validation_X_normalized = ValidationData[0]
        validation_Y = ValidationData[1]
        validation_y = ValidationData[2]
        
        test_X_normalized = TestData[0]
        test_Y = TestData[1]
        test_y = TestData[2]
        
        K = 10  # number of labels/classes
        d = 3072 # number of dimensions of an image 32x32x3
        mu, sigma = 0, 0.01 # mean and standard deviation
        W = np.random.normal(mu, sigma, (K, d))
        b = np.random.normal(mu, sigma, (K, 1))

        # J_epocs = cost per epoch
        J_epocs_train = np.zeros(n_epocs)
        J_epocs_validation = np.zeros(n_epocs)
        # Accuracy_epocs = accuracy per epoch
        Accuracy_epocs_train = np.zeros(n_epocs)
        Accuracy_epocs_validation = np.zeros(n_epocs)

        start_time = datetime.datetime.now()
        total_batch = int(train_X_normalized.shape[1] / n_batch) # how many batches we will have
        # i.e if n_batch=10 and we have 200 images, then we will have 20 batches each having 10 images in it
        for e in range(n_epocs):
            #print('e: ' + str(e))
            for batch in range(total_batch):
                #print('batch: ' + str(batch))
                index_list = list(range(batch * 100, (batch+1) * 100))
                # shuffling the indexes (so image samples) of the selected batch
                np.random.shuffle(index_list)
                X_batch = train_X_normalized[:, index_list]
                Y_batch = train_Y[:, index_list]

                [W, b] = self.MiniBatchGD(X_batch, Y_batch, GDparams, W, b, lambda_cost)

            P_train = self.EvaluationClassifier(train_X_normalized, W, b)
            k_train = np.argmax(P_train, axis=0)  
            A_train = self.ComputeAccuracy(k_train, train_y)
            Accuracy_epocs_train[e] = A_train

            P_validation = self.EvaluationClassifier(validation_X_normalized, W, b)
            k_validation = np.argmax(P_validation, axis=0)  
            A_validation = self.ComputeAccuracy(k_validation, validation_y)
            Accuracy_epocs_validation[e] = A_validation

            J_train = self.ComputeCost(train_Y, train_X_normalized, b, W, lambda_cost)
            J_epocs_train[e] = J_train

            J_validation = self.ComputeCost(validation_Y, validation_X_normalized, b, W, lambda_cost)
            J_epocs_validation[e] = J_validation
            
        P_test = self.EvaluationClassifier(test_X_normalized, W, b)
        k_test = np.argmax(P_test, axis=0)
        acc_test = self.ComputeAccuracy(k_test, test_y)
        print("\nTest data final accuracy: "+ str(round(acc_test*100, 4)) + "%")
        
        self.Plot_Train_Validation_Cost_Accurracy(J_epocs_train, J_epocs_validation, Accuracy_epocs_train, Accuracy_epocs_validation)
        
        cifar = CIFAR_IMAGES()
        cifar.show_images_array(W)
        #print(W.shape)
        #print(W)
        
        end_time = datetime.datetime.now()
        print("Calculation time: "+ str(end_time - start_time))
        return W, b
    
    def Plot_Train_Validation_Cost_Accurracy(self, Cost_Train, Cost_Validation, Acc_Train, Acc_Validation):
        plt.subplot(1,2,1)
        plt.plot(Cost_Train, 'g-', label='Train')
        plt.plot(Cost_Validation, 'r-', label='Validation')
        plt.title('Cost Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid('on')
        
        plt.subplot(1,2,2)
        plt.plot(Acc_Train, 'g-', label='Train')
        plt.plot(Acc_Validation, 'r-', label='Validation')
        plt.title('Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid('on')