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
    
    # dictBatchData is the dictionary having all info related to the images in the batch
    # dictBatch[b'data']=a numpy array with  having all data (b'data') information from the loaded batch
    # e.g. for the batch1, dictBatchData=(10000, 3072) matrix (row:image, Column:HxWxRGB binary))
    def show_images(self, dictBatchData):
        #import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,5) # 2, 5=?
        for i in range(2):
            for j in range(5):
                im  = dictBatchData[b'data'][5*i+j,:].reshape(32,32,3, order='F') # why do we use order='F'??
                #sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = np.divide(im, 255)
                # rotate to the counter-clockwise 90 degrees for 3 times because images are provided rotated once
                sim = np.rot90(sim, k=3)
                # below does the rotation, too
                #sim = sim.transpose(1,0,2)
                
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("img_" + str(5*i+j) + ": " + self.unique_labels[dictBatchData[b'labels'][5*i+j]])
                ax[i][j].axis('off')
        plt.show()
        
    def show_images_random(self, dictBatchData, n_row = 2, n_column = 5):
        #import matplotlib.pyplot as plt
        fig, ax = plt.subplots(n_row, n_column)
        total_images = n_row * n_column
        for plot_i, img_i in enumerate(np.random.choice(dictBatchData[b'data'].shape[0], total_images, replace=False)):
                im  = dictBatchData[b'data'][img_i,:].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                # better to use above calculation for the normalization since the values might not always 
                # be regular RGB (in 0-255 range)
                #sim = np.divide(im, 255)
                # rotate to the counter-clockwise 90 degrees for 3 times because images are provided rotated once
                sim = np.rot90(sim, k=3)
                # below does the rotation, too
                #sim = sim.transpose(1,0,2)
                
                plt.subplot(n_row, n_column, plot_i+1)
                plt.imshow(sim, interpolation='gaussian')
                plt.title("img_" + str(img_i) + ": " + self.unique_labels[dictBatchData[b'labels'][img_i]])
                plt.axis('off')
        plt.show()
    
    def show_images_array(self, W, n_row = 2, n_column = 5):
        # W is the weight array of size Cxd = 10x3072
        fig, ax = plt.subplots(n_row, n_column)
        for plot_i, img_i in enumerate(W):
            im  = img_i.reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))

            plt.subplot(n_row, n_column, plot_i+1)
            plt.imshow(sim, interpolation='gaussian')
            plt.title("img_" + str(plot_i))
            plt.axis('off')
        plt.show()
        
        
    """ Used to transfer a python model to matlab """
    def save_as_mat(data, name="model"):
        #import scipy.io as sio
        sio.savemat(name+'.mat',{name:b})
        