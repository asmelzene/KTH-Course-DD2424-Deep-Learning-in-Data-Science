# This is a .py file with the name "dataset.py"
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
        
    # NOT USED
    def load_labels(self, filePathLocal, keyToRead=b'label_names'):
        self.filePath = filePathLocal
        with open(filePathLocal, 'rb') as fileToOpen:
            dictUniqueLabels = pickle.load(fileToOpen, encoding='bytes')
            #dictUniqueLabels_keys([b'num_cases_per_batch', b'label_names', b'num_vis'])
            #print(dictUniqueLabels.get(b'label_names'))
            #print(type(dictUniqueLabels[b'label_names'][0]))
            #<class 'bytes'>
            #[b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
            # labels = class names of each image. we convert the values from bytes to string
            self.unique_labels = [u_lbl.decode('ascii') for u_lbl in dictUniqueLabels[keyToRead]]
            #print(unique_labels)
            #['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            #print(type(self.unique_labels[0]))
            #<class 'str'>
            
    # NOT USED
    def load_batch(self, filePathLocal):
        # requires "import pickle"
        # filePathLocal=filepath for the the required batch of images
        with open(filePathLocal, 'rb') as fileToOpen:
            dictBatch = pickle.load(fileToOpen, encoding='bytes')
            # dictBatch.keys()
            # dictBatch_keys([b'batch_label', b'labels', b'data', b'filenames'])
        return dictBatch
    
    # below function corresponds to [X, Y, y] = LoadBatch(filename) in the assignment, 1st task.
    def load_batch_a1(self, filePathLocal):
        with open(filePathLocal, 'rb') as fileToOpen:
            dictBatch = pickle.load(fileToOpen, encoding='bytes')
            Y_one_hot_labels = self.label_encoder.fit_transform(dictBatch[b'labels'])
        #return [X, Y, y]
        return [np.divide(dictBatch[b'data'], 255).transpose(), Y_one_hot_labels.transpose(), np.array(dictBatch[b'labels'])]
           
        
    """ Used to transfer a python model to matlab """
    def save_as_mat(data, name="model"):
        #import scipy.io as sio
        sio.savemat(name+'.mat',{name:b})
        