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


from utils import MultiClassImageDataset, Base_Model, preprocessing, CreateQNN,ResNet50_Model,ProgressBar,OrthoModel_Binary_ClQ
from utils import *

sns.set_style('darkgrid')

import random
TEST_CLASSICAL = True
IMAGE_SIZE = 64
if TEST_CLASSICAL:
    BATCH_SIZE = 1024
else:
    BATCH_SIZE = 256
NUM_QUBIT = 4
TEST_DATA_NUMBER= 100
SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_"
root_dir_binary = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Train/"
root_dir_binary_test = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Test/"

backup_acc_data = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_style('darkgrid')

image_transforms = preprocessing(IMAGE_SIZE, device)
dataSet = MultiClassImageDataset(root_dir_binary, image_transforms["train"])
dataSet_test = MultiClassImageDataset(root_dir_binary_test, image_transforms["train"])

train_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=dataSet_test, shuffle=True, batch_size=1)
d = next(iter(train_loader))
print("Train Image" , d[0].shape)
print("Train Label" , d[1])

# QC_NN = CreateQNN(NUM_QUBIT)
# QC_NN = CreateOrthogonalQNN4(NUM_QUBIT,  SAMPLER = False)
QC_NN = CreateOrthogonalQNN4(NUM_QUBIT,  SAMPLER = False)
Binary_classifier_classical = OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT, Train_Classical = True).to(device)
# multiClassBaseModel = Base_Model(QC_NN, NUM_QUBIT).to(device)
# multiClassResnetModel = ResNet50_Model(QC_NN, NUM_QUBIT).to(device)
# multiClassResnetModel.eval()
# multiClassResnetModel.resnet_encoder.eval()

# multiClassBaseModel.eval()  # Set model to training mode


classifierModel = Binary_classifier_classical
print(classifierModel)


# optimizer = optim.Adam(classifierModel.parameters(), lr=0.001)
loss_func = CrossEntropyLoss()

# Start training
loss_list = []  # Store loss history
total_loss= []
accuracy=[]
acc_data=[]
raw_data_pred =[]
raw_data_target =[]

loss_data= []

if TEST_CLASSICAL: 
    all_exp = [""]
    starti = [0]
    endi = [240]
    intervals = 1
else:
    all_exp = [ "", "c", "cc", "ccc", "cccc"]
    starti = [0, 0, 0, 0, 0]
    endi = [120, 120, 120, 220, 200]
    intervals = 10



# if backup_acc_data != None:
#     print("setting acc_data")
#     acc_data = backup_acc_data

all_exp_data = {}

for sd in [1000,1,3,1222,122,5,90,34,33]:

    random.seed(int(sd))
    test_loader = DataLoader(dataset=dataSet_test, shuffle=True, batch_size=1)
    acc_data = []
    loss_list = [] 
    

    
    for e, exp in enumerate(all_exp): 
        # if exp == "ccc":
        print(sd)
        #     end = 220
        start = starti[e]
        end = endi[e]
        for i in range(start, end, intervals):  
            if TEST_CLASSICAL:   
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_4_"+str(i)+".pt"
                
                classifierModel.Train_Classical = True
                classifierModel.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
                classifierModel.eval()
                SAVE_DATA_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_4_24"+str(exp)+str(i)
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_4_3_"+str(exp)+str(i)+".pt"
            else:
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_4_"+str(exp)+str(i)+".pt"
                classifierModel.Train_Classical = False
                classifierModel.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
                classifierModel.eval()
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_4_3_"+str(exp)+str(i)+".pt"
                SAVE_DATA_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_4_24"+str(exp)+str(i)

            images =[]
            label = []
            preds = []

            accuracy = [] 
            total_loss = []
            raw_data_pred = []
            raw_data_target = [] 
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx == TEST_DATA_NUMBER:
                    break
                target= target.to(device)
                data= data.to(device)

                if TEST_CLASSICAL:
                    output = classifierModel(data)  # Forward pass
                else:
                    # test both parts 
                    o = classifierModel.classical_encoder(data) 
                    norm = torch.norm(o, dim=1, keepdim=True)
                        # Normalize each data point in the batch
                    o = o / (norm+1e-8)
                    o = classifierModel.encode_amplitudes_NQubit(o, NUM_QUBIT)
                    o2 = classifierModel.quantum_part(o) 
                    if Binary_classifier_classical.Train_Classical == False:
                        output=o2
                    else: 
                        print("Train_Classical set to True")

                raw_data_pred.append(output.cpu().detach().numpy())
                raw_data_target.append(target.cpu().detach().numpy())

                loss = loss_func(output, target)  # Calculate loss
                
                preds.append(output.cpu().detach().numpy())
                label.append(target.cpu().detach().numpy())
                images.append(data.cpu().detach().numpy())
                total_loss.append(loss.item())
                predicted_labels = np.argmax(output.cpu().detach().numpy())
                target_labels = target.cpu().detach().numpy()
                # print(str(np.int32(test_target == test_pred)))
                accuracy_test = np.int32(target_labels == predicted_labels)
                accuracy.append(accuracy_test)

            

            # accuracy.append(np.argmax(pred[-1]))
            loss_list.append(sum(total_loss) / len(total_loss))
            acc_data.append(np.sum(accuracy)/len(accuracy))
            print("Testing ", SAVE_PATH, "Seed:", sd)
            print("\t\tTesting [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (1 + 1) , loss_list[-1]))
            print("\t\tTesting \t accuracy: ", np.sum(accuracy)/len(accuracy))

        np.save(SAVE_DATA_PATH+"Acc_Data.npy", acc_data)
        np.save(SAVE_DATA_PATH+"Loss_Data.npy", loss_list)
        np.save(SAVE_DATA_PATH+"Raw_Data_Pred.npy", raw_data_pred)
        np.save(SAVE_DATA_PATH+"Raw_Data_Target.npy", raw_data_target)

        all_exp_data[sd] = [acc_data, loss_list]
        all_exp_data["Raw_predictions"] = [raw_data_pred]
        all_exp_data["Raw_targets"] = [raw_data_target]


        
        print("Testing [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (1 + 1) , loss_list[-1]))
        print("Testing \t accuracy: ", np.sum(accuracy)/len(accuracy))

        np.save(SAVE_DATA_PATH+"all_exp_dict_100.npy", all_exp_data)
if TEST_CLASSICAL:
    np.save("/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_4_24_final_all_exp_dict_100.npy", all_exp_data)
else:
    np.save("/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_4_24_final_all_exp_dict_100.npy", all_exp_data)
