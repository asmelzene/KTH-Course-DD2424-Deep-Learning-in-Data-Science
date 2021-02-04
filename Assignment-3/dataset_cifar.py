# This part is needed to be saved as a .py file with the name "dataset_cifar.py" in the same folder with Assignment3.py
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
        
    """ Used to transfer a python model to matlab """
    def save_as_mat(data, name="model"):
        #import scipy.io as sio
        sio.savemat(name+'.mat',{name:b})
        