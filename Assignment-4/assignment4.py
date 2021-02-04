import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from copy import deepcopy
import datetime

def load_data(file_path):
    with open(file_path, 'r') as file:
        book_data = file.read()

    book_chars = set(book_data)
    K = len(book_chars)
    
    char_to_ind = dict([(char, ind) for ind, char in enumerate(book_chars)])
    ind_to_char = dict([(ind, char) for ind, char in enumerate(book_chars)])
    
    data_processed = {'book_data': book_data, 'book_chars': book_chars, 'K': K, 'char_to_ind': char_to_ind, 'ind_to_char': ind_to_char}
    
    return data_processed

def onehot_encode(char_to_ind, string_to_encode):
    seq_length = len(string_to_encode)     # length of the string to be encoded
    char_dict_length = len(char_to_ind)    # legth of the unique characters in the dictionary composed of the characters from the given file
    
    # each column vector represents a character in a string to be encoded
    # each row represents a character in the dictionary. if this character is matching with the character in the string then it will be 1 otherwise 0
    encoded_string = np.zeros((char_dict_length, seq_length))
    
    for i, char in enumerate(string_to_encode):
        encoded_string[char_to_ind[char], i] = 1
        
    return encoded_string

def Gradient_Check(X, Y, params):
#     K = len(data_processed['char_to_ind']); m = 5; eta = 0.1; seq_length = 25; sig = 0.01
#     parameters_grad_check = [K, m, eta, seq_length, sig]
    K = params[0]; m = params[1]; eta = params[2]; seq_length = params[3]; sig = params[4]
    rnn1 = RecurrentNeuralNetwork(K, m, eta, seq_length, sig)

    h0 = np.zeros((rnn1.m, 1))
    (A, H, P) = rnn1.forward(h0, X)
    (grad_W, grad_U, grad_V, grad_b, grad_c) = rnn1.backward(X, Y, P, H, A)
    grad_num_W, grad_num_U, grad_num_V, grad_num_b, grad_num_c = rnn1.ComputeGradsNumSlow(X, Y)

    rnn1.CompareGradients(grad_W, grad_num_W, grad_type = "W")
    rnn1.CompareGradients(grad_U, grad_num_U, grad_type = "U")
    rnn1.CompareGradients(grad_V, grad_num_V, grad_type = "V")
    rnn1.CompareGradients(grad_b, grad_num_b, grad_type = "b")
    rnn1.CompareGradients(grad_c, grad_num_c, grad_type = "c")

def Run_Training(data_processed, params):
    n_chars_book = len(data_processed['book_data'])
    X_chars = data_processed['book_data'][:n_chars_book - 1]
    Y_chars = data_processed['book_data'][1:n_chars_book]
    X = onehot_encode(data_processed['char_to_ind'], X_chars)
    Y = onehot_encode(data_processed['char_to_ind'], Y_chars)

    np.random.seed(400)
    start_time = datetime.datetime.now()
    #K = len(data_processed['char_to_ind']); m = 100; eta = 0.1; seq_length = 25; sig = 0.01
    K = len(data_processed['char_to_ind']); m = params[0]; eta = params[1]; seq_length = params[2]; sig = params[3]
    rnn1 = RecurrentNeuralNetwork(K, m, eta, seq_length, sig)

    # parameters = [n_epochs, eta, n_print_loss, n_synth_length, n_synth_steps, ind_to_char]
    # parameters = [7, 0.1, 1000, 200, 10000, data_processed['ind_to_char']]
    n_epochs = params[4]; n_print_loss = params[5]; n_synth_length = params[6]; n_synth_steps = params[7]
    parameters = [n_epochs, eta, n_print_loss, n_synth_length, n_synth_steps, data_processed['ind_to_char']]
    best_model = rnn1.Train_RNN_AdaGrad(X, Y, parameters)

    Plot_Cost(rnn1.l_smooth_loss)
    end_time = datetime.datetime.now()

    print("Calculation time of GradientSanityCheck: {} ... finished at: {}".format(end_time - 
                                                        start_time, datetime.datetime.now()))
    
    return rnn1, best_model

def Plot_Cost(l_smooth_loss):
    plt.rcParams['figure.figsize'] = (15.0, 5.0)
    plt.plot(l_smooth_loss, 'g-')
    plt.title('RNN - AdaGrad')
    plt.xlabel('update_step')
    plt.ylabel('Smooth Loss')
    plt.grid('on')

def Best_Model_Test_x(rnn_best, data_processed, n_synth_length = 1000):
    sampled_indexes = rnn_best.synthesize_seq(rnn_best.best_h_prev, rnn_best.best_x0, n_synth_length)
    print('\n*************************************** Best Model >> Synthesized text ({} characters):'.format(n_synth_length))
    print(''.join([data_processed['ind_to_char'][index] for index in sampled_indexes]))
    
def Best_Model_Test(rnn_train, rnn_best, data_processed, n_synth_length = 1000):
    sampled_indexes = rnn_best.synthesize_seq(rnn_train.best_h_prev, rnn_train.best_x0, n_synth_length)
    print('\n*************************************** Best Model >> Synthesized text ({} characters):'.format(n_synth_length))
    print(''.join([data_processed['ind_to_char'][index] for index in sampled_indexes]))
    
class RecurrentNeuralNetwork():
    def __init__(self, K, m = 100, eta = 0.1, seq_length = 25, sig = 0.01):
        self.m = m   # 100: number of hidden nodes = state_size
        self.K = K   # 80: number of unique labels = input_size = output_size
        
        self.eta = eta
        self.seq_length = seq_length
        self.sig = sig
        self.epsilon = np.finfo(np.float64).eps
        
        # weight matrices
        self.W = np.random.randn(self.m, self.m) * self.sig     # (state_size, state_size)
        self.U = np.random.randn(self.m, self.K) * self.sig     # (state_size, input_size)
        self.V = np.random.randn(self.K, self.m) * self.sig     # (output_size, state_size)
        # bias vectors
        self.b = np.zeros((self.m, 1))                          # state_size
        self.c = np.zeros((self.K, 1))                          # output_size

    def softmax(self, o):
        exp_o = np.exp(o)
        p = exp_o / np.sum(exp_o, axis=0)
        
        return p
        
    def synthesize_seq(self, h0, x0, n):
        """
        In the body of the function you will write code to implement the equations (1-4). There is just one major difference - you have to generate the next input vector xnext from the current input vector x. At each time step t when you generate a vector of probabilities for the labels, you then have to sample a label (i.e. an integer) from this discrete probability distribution. This sample will then be the (t + 1)th character in your sequence
        """
        h = h0    # h0 = the hidden stateat time 0)
        x = x0    # x0 = the first (dummy) input vector to your RNN (it can be some character like a full-stop)

        # n = the length of the sequence you want to generate
        # You should store each index you sample for 1 < t < n and let your function output the matrix Y (size Kxn) where Y is the one-hot encoding of each sampled character. 
        # Given Y you can then use the map container ind_to_char to convert it to a sequence of characters and view what text your RNN has generated.
        sampled_indexes = []
        for t in range(n):
            a = self.W @ h + self.U @ x + self.b # eq.1 ... h = h[t - 1]  ... (m, 1)
            h = np.tanh(a)                       # eq.2                   ... (m, 1)
            o = self.V @ h + self.c              # eq.3 ... h = h[t]      ... (K, 1)
            p = self.softmax(o)                  # eq.4                   ... (K, 1)

            # randomly select a character based on the output probability scores p
            # compute the vector containing the cumulative sum of the probabilities
            cp = np.cumsum(p)
            # generate a random draw, a, from a uniform distribution in the range 0 to 1
            a_rand = random.random()
            # find the index 1<=ii<=K such that cp(ii-1)<=a<=cp(ii) where we assume for notational convenience cp(0)=0
            ixs = np.where(cp - a_rand > 0)
            ii = ixs[0][0]
            sampled_indexes.append(ii)
            
            # onehot_encoding
            x = np.zeros((self.K, 1))            # (K, 1)
            x[ii] = 1     

        return sampled_indexes
    
    def forward(self, h0, encoded_input):
        A = np.zeros((self.m, self.seq_length))                   # (m, seq_length)
        H = np.zeros((self.m, self.seq_length + 1))               # (m, seq_length + 1)
        O = np.zeros((encoded_input.shape[0], self.seq_length))   # (K, seq_length)
        P = np.zeros((encoded_input.shape[0], self.seq_length))   # (K, seq_length)

        h_t = h0
        for t in range(self.seq_length):
            # H[:, t] here, corresponds to H[:, t - 1] in the formula
            A[:, t] = (self.W @ h_t + self.U @ encoded_input[:, t].reshape(-1, 1) + self.b).flatten()
            h_t = np.tanh(A[:, t]).reshape(-1, 1)
            H[:, t + 1] = h_t.flatten()
            O[:, t] = (self.V @ H[:, t + 1]).flatten() + self.c.flatten()
            P[:, t] = self.softmax(O[:, t].reshape(-1, 1))[:,0]
            
        return (A, H, P)
    
    def compute_loss(self, h0, X, Y):
        (_, _, P) = self.forward(h0, X)
        
        cel1 = np.dot(Y.T, P)
        cel1[cel1 == 0] = np.finfo(float).eps
        cross_entropy_loss = -np.log(cel1) 
        sum_cross_entropy_loss = np.trace(cross_entropy_loss)
        
        return sum_cross_entropy_loss
    
    def backward(self, X, Y, P, H, A):
        # Gradients
        grad_W = np.zeros_like(self.W)            # (m, m)
        grad_U = np.zeros_like(self.U)            # (m, K)
        grad_V = np.zeros_like(self.V)            # (K, m)
        grad_b = np.zeros_like(self.b)            # (m, 1)
        grad_c = np.zeros_like(self.c)            # (K, 1)
        
        grad_A = np.zeros_like(A)                 # (m, seq_length)
        grad_H = np.zeros_like(H)                 # (m, seq_length + 1)
        
        for t in reversed(range(self.seq_length)):
            dL_dO = P[:, t] - Y[:, t]
            grad_c[:, 0] += dL_dO
            # H[:, t + 1] is indeed H[:, t] since h0 is added first
            grad_V += dL_dO.reshape(-1, 1) @ H[:, t + 1].reshape(1, -1)   # (K,m)
            
            if t == self.seq_length - 1:
                grad_H[:, t] = dL_dO @ self.V                             # Derivative of the last hidden state
            else:
                grad_H[:, t] = dL_dO @ self.V + grad_A[:, t + 1] @ self.W
                
            grad_A[:, t] = grad_H[:, t] @ (np.diag(1 - np.tanh(A[:, t]) ** 2))
            grad_W += grad_A[:, t].reshape(-1, 1) @ H[:, t].reshape(1, -1)
            grad_U += grad_A[:, t].reshape(-1, 1) @ X[:, t].reshape(1, -1)
            grad_b[:, 0] += grad_A[:, t]
        
        grad_W = np.clip(grad_W, -5, 5)
        grad_U = np.clip(grad_U, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)            
            
        return (grad_W, grad_U, grad_V, grad_b, grad_c)
           
    def ComputeGradsNumSlow(self, X, Y, h = 1e-4):
        grad_num_W = np.zeros(self.W.shape)
        grad_num_U = np.zeros(self.U.shape) 
        grad_num_V = np.zeros(self.V.shape) 
        grad_num_b = np.zeros(self.b.shape)
        grad_num_c = np.zeros(self.c.shape)
        
        h0 = np.zeros((self.m, 1))
 
        for i in tqdm(range(self.W.shape[0])):
            for j in range(self.W.shape[1]):
                self.W[i, j] += h
                cost_p = self.compute_loss(h0, X, Y)
                self.W[i, j] -= 2 * h
                cost_n = self.compute_loss(h0, X, Y)
                self.W[i, j] += h
                
                grad_num_W[i, j] = (cost_p - cost_n) / (2 * h)
        
        for i in tqdm(range(self.U.shape[0])):
            for j in range(self.U.shape[1]):
                self.U[i, j] += h
                cost_p = self.compute_loss(h0, X, Y)
                self.U[i, j] -= 2 * h
                cost_n = self.compute_loss(h0, X, Y)
                self.U[i, j] += h
                
                grad_num_U[i, j] = (cost_p - cost_n) / (2 * h)
       
        for i in tqdm(range(self.V.shape[0])):
            for j in range(self.V.shape[1]):
                self.V[i, j] += h
                cost_p = self.compute_loss(h0, X, Y)
                self.V[i, j] -= 2 * h
                cost_n = self.compute_loss(h0, X, Y)
                self.V[i, j] += h
                
                grad_num_V[i, j] = (cost_p - cost_n) / (2 * h)
       
        for i in tqdm(range(self.b.shape[0])):
            self.b[i] += h
            cost_p = self.compute_loss(h0, X, Y)
            self.b[i] -= 2 * h
            cost_n = self.compute_loss(h0, X, Y)
            self.b[i] += h
            
            grad_num_b[i] = (cost_p - cost_n) / (2 * h)
       
        for i in tqdm(range(self.c.shape[0])):
            self.c[i] += h
            cost_p = self.compute_loss(h0, X, Y)
            self.c[i] -= 2 * h
            cost_n = self.compute_loss(h0, X, Y)
            self.c[i] += h
            
            grad_num_c[i] = (cost_p - cost_n) / (2 * h)
    
        grad_num_W = np.clip(grad_num_W, -5, 5)
        grad_num_U = np.clip(grad_num_U, -5, 5)
        grad_num_V = np.clip(grad_num_V, -5, 5)
        grad_num_b = np.clip(grad_num_b, -5, 5)
        grad_num_c = np.clip(grad_num_c, -5, 5)
        
        return grad_num_W, grad_num_U, grad_num_V, grad_num_b, grad_num_c
    
    def CompareGradients(self, gradAnalytic, gradNumerical, grad_type = "W"):
        print("\nGradient difference comparisons for: {}".format(grad_type))

        grad_abs_diff = np.abs(gradAnalytic - gradNumerical)
        grad_abs_max = np.maximum(np.abs(gradAnalytic), np.abs(gradNumerical))
        # grad_U was giving NaN values, so comparing with self.epsilon was added to overcome that issue
        #grad_check = grad_abs_diff / grad_abs_max
        grad_check = grad_abs_diff / np.maximum(grad_abs_max, self.epsilon)
        
        print("MEAN: {}\t\tMAX: {}\t\tMIN: {}".format(np.mean(grad_check), np.max(grad_check), np.min(grad_check)))
              
    def Train_RNN_AdaGrad(self, X, Y, parameters):
        # parameters = [n_epochs, eta, n_print_loss, n_synth_length, n_synth_steps, ind_to_char]
        # parameters = [7, 0.1, 1000, 200, 10000, data_processed['ind_to_char']]
        n_epochs = parameters[0]             # 7
        eta = parameters[1]                  # 0.1        
        n_print_loss = parameters[2]         # 1000
        n_synth_length = parameters[3]       # 200
        n_print_synth = parameters[4]        # 10000
        ind_to_char = parameters[5]
        
        book_length = X.shape[1]
        self.l_smooth_loss = []

        ada_grad_V = np.zeros(self.V.shape)
        ada_grad_W = np.zeros(self.W.shape)
        ada_grad_U = np.zeros(self.U.shape)
        ada_grad_b = np.zeros(self.b.shape)
        ada_grad_c = np.zeros(self.c.shape)
        
        self.best_loss = np.finfo(float).max
        best_model = None
        self.best_h_prev = None
        self.best_x0 = None
        
        update_step = 0
        for epoch in tqdm(range(n_epochs)):
            print("Epoch: {}".format(epoch))
            e = 0                             # reset the position in the book
            h_prev = np.zeros((self.m, 1))
            
            while e + self.seq_length < book_length - 1:
                start = e 
                end = start + self.seq_length
 
                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                if update_step % n_print_synth == 0:
                    sampled_indexes = self.synthesize_seq(h_prev, X_batch[:, 0].reshape(-1, 1), n_synth_length)
                    print('\n******************************************************************************** Synthesized text:')
                    print(''.join([ind_to_char[index] for index in sampled_indexes]))
                    print('*****************************************************************************\n')
               
                (A, H, P) = self.forward(h_prev, X_batch)
                (grad_W, grad_U, grad_V, grad_b, grad_c) = self.backward(X_batch, Y_batch, P, H, A)
                
                ada_grad_W += grad_W ** 2
                self.W += -(eta * grad_W) / np.sqrt(ada_grad_W + self.epsilon)
                
                ada_grad_U += grad_U ** 2
                self.U += -(eta * grad_U) / np.sqrt(ada_grad_U + self.epsilon)
                
                ada_grad_V += grad_V ** 2
                self.V += -(eta * grad_V) / np.sqrt(ada_grad_V + self.epsilon)
                
                ada_grad_b += grad_b ** 2
                self.b += -(eta * grad_b) / np.sqrt(ada_grad_b + self.epsilon)
                
                ada_grad_c += grad_c ** 2
                self.c += -(eta * grad_c) / np.sqrt(ada_grad_c + self.epsilon)
                
                loss = self.compute_loss(h_prev, X_batch, Y_batch)
                
                if update_step == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = .999 * smooth_loss + .001 * loss;
                    
                self.l_smooth_loss.append(smooth_loss)
                     
                if update_step % n_print_loss == 0:
                    print("update_step: {} ... smooth_loss: {}".format(update_step, smooth_loss))
                    
                h_prev = H[:, -1].reshape(-1, 1)
                e += self.seq_length
                update_step += 1
                
                if smooth_loss < self.best_loss:
                    self.best_loss = smooth_loss
                    best_model = deepcopy(self)
                    self.best_h_prev = deepcopy(h_prev)
                    self.best_x0 = deepcopy(X_batch[:, 0].reshape(-1, 1))
                    self.best_step = update_step
                    
        return best_model
    
    