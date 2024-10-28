import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn 

import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss, CrossEntropyLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
    MSELoss,
)
import torch.nn.functional as F
# torch.set_default_device('cuda')

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


from utils import MultiClassImageDataset, Base_Model, preprocessing, \
    CreateQNN,ResNet50_Model, ProgressBar, ResNet50_Model_Binary, ResNet50_OrthoModel_Binary
    
from utils import CreateQNN_Ex, CreateOrthogonalQNN

# def CreateQNN(): 
#     feature_map = ZZFeatureMap(2)
#     ansatz = RealAmplitudes(2, reps=1)
#     qc = QuantumCircuit(2)
#     qc.compose(feature_map, inplace=True)
#     qc.compose(ansatz, inplace=True)
    
#     print(qc.draw())

#     # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
#     qnn = EstimatorQNN(
#         circuit=qc,
#         input_params=feature_map.parameters,
#         weight_params=ansatz.parameters,
#         input_gradients=True,
#     )
#     return qnn
def process_target(target):
    target = F.one_hot(target, num_classes=2)
    zero_indices = target == 0
    target[zero_indices] = -1

    return target

def main(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder"):
    
    BINARY = True
    ORTHO = True 
    LOAD = False
    MSE = True
    max_num_test = 100
    epochs = 80    # Set number of epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sns.set_style('darkgrid')
    root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"
    
    root_dir_binary = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Train/"
    
    
    image_transforms = preprocessing(IMAGE_SIZE, device)
    
    if BINARY:
        dataSet = MultiClassImageDataset(root_dir_binary, image_transforms["train"])
    else:
        dataSet = MultiClassImageDataset(root_dir, image_transforms["train"])
    
    train_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=1)
    d = next(iter(train_loader))
    print("Train Image" , d[0].shape)
    print("Train Label" , d[1])
    
    QC_NN = CreateQNN(NUM_QUBIT)
    QC_NN = CreateQNN_Ex(NUM_QUBIT)
    
    
    # QC_NN = CreateBaseQNN(NUM_QUBIT)
    if BINARY:
        if ORTHO:
            QC_NN = CreateOrthogonalQNN(NUM_QUBIT)
            classifierModel = ResNet50_OrthoModel_Binary(QC_NN, NUM_QUBIT).to(device)
            classifierModel.train()
            classifierModel.resnet_encoder.eval()
            SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder_Binary_Ortho2_"
            
        else:
            SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder_Binary"
            classifierModel = ResNet50_Model_Binary(QC_NN, NUM_QUBIT).to(device)
            if LOAD:
                print("Loading ..... ")
                classifierModel.load_state_dict(torch.load(SAVE_PATH+"19.pt", weights_only=True))
                print("Loaded model :  ",SAVE_PATH+"19.pt" )
            classifierModel.train()
            classifierModel.resnet_encoder.eval()
            
            

    else:
        multiClassBaseModel = Base_Model(QC_NN, NUM_QUBIT).to(device)
        multiClassResnetModel = ResNet50_Model(QC_NN, NUM_QUBIT).to(device)
        multiClassResnetModel.train()
        multiClassBaseModel.train()  # Set model to training mode
        multiClassResnetModel.resnet_encoder.eval()
        
        classifierModel = multiClassResnetModel
        classifierModel.resnet_encoder.eval()
        # classifierModel = multiClassResnetModel
    
    
    # Set optimizer for training.
    optimizer = optim.Adam(classifierModel.parameters(), lr=0.0002)
    # optimizer = optim.SPSA (classifierModel.parameters(), lr=0.0002)
    loss_func = CrossEntropyLoss()
    loss_func_MSE = MSELoss() 
    
    # max_val = 1000
    # p = ProgressBar(max_value=max_val, disable=False)

    # Start training
   
    loss_list = []  # Store loss history
    accuracies = []
    for epoch in range(epochs):
        total_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):
            target= target.to(device)
            
            data= data.to(device)
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = classifierModel(data)  # Forward pass
            if MSE == True:
                target = process_target(target)
                loss = loss_func_MSE(output, target.float())
            else:
                loss = loss_func(output, target)  # Calculate loss
            
            #check wights before and after optimizer step
            # print(classifierModel.full[0])
            # sum_w = classifierModel.full[2].weight.cpu().detach().numpy()
            
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            
            # check weights after update
            # sum_w2 = classifierModel.full[2].weight.cpu().detach().numpy()
            # print( "weight check",np.sum(sum_w - sum_w2))
            
            total_loss.append(loss.item())  # Store loss
            
            if batch_idx%40 == 0:
                print(batch_idx)
                torch.save(classifierModel.state_dict(), SAVE_PATH+str(epoch)+".pt")
                accuracy_test = 0
                for num_test, (test_data, test_target) in enumerate(test_loader):
                    if num_test == max_num_test:
                        break
                    test_target= test_target.to(device)
                    test_data= test_data.to(device)
                    
                    test_pred = np.argmax(classifierModel(test_data).cpu().detach().numpy())
                    test_target = test_target.cpu().detach().numpy()
                    # print(str(np.int32(test_target == test_pred)))
                    accuracy_test += np.int32(test_target == test_pred)
                accuracies.append(np.float32(accuracy_test/max_num_test))
                

        loss_list.append(sum(total_loss) / len(total_loss))
            # accuracy_test = np.bool(target == torch.argmax (prediction))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
        print("\taccuracy: ", str(np.float32(accuracy_test/max_num_test)))
    
    
    plt.figure("Accuracy vs Epoch") # Here's the part I need
    plt.title('accuracy')
    plt.plot(accuracies, )
    plt.savefig(SAVE_PATH+ "_accuracy.png")
    
    plt.figure("Loss vs Epoch") 
    plt.title('loss')
    plt.plot(loss_list)
    plt.savefig(SAVE_PATH+ "_loss.png")
    



if __name__ == "__main__":
    IMAGE_SIZE = 64
    BATCH_SIZE = 1024
    NUM_QUBIT = 4
    main(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT)
    
    