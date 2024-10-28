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
    CreateQNN,ResNet50_Model, ProgressBar, ResNet50_Model_Binary, \
        ResNet50_OrthoModel_Binary, ResNet50_OrthoModel_Binary_ClQ, OrthoModel_Binary_ClQ
    
from utils import CreateQNN_Ex, CreateOrthogonalQNN, CreateOrthogonalQNN4,CreateOrthogonalQNN8, CreateOrthogonalQNN6
from torch.utils.tensorboard import SummaryWriter

def process_target(target):
    target = F.one_hot(target, num_classes=4)
    zero_indices = target == 0
    target[zero_indices] = -1
    target =  target.to(torch.float32)
    return target

def process_target_1Hot(target):
    target = F.one_hot(target, num_classes=4)
    target =  target.to(torch.float32)
    return target


BINARY = True
ORTHO = True 
LOAD = False
MSE = False # False for cross entropy
RESNET = False
TRAIN_ENCODER = False
max_num_test = 100
epochs = 160
TRAIN_REGIME = True

def train_classical_encoder(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS, SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder"):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sns.set_style('darkgrid')
    root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"
    
    root_dir_binary = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Train/"
    root_dir_binary_test = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Test/"
    
    
    image_transforms = preprocessing(IMAGE_SIZE, device)
    
    if BINARY:
        dataSet = MultiClassImageDataset(root_dir_binary, image_transforms["train"])
        dataSet_test = MultiClassImageDataset(root_dir_binary_test, image_transforms["train"])
    else:
        dataSet = MultiClassImageDataset(root_dir, image_transforms["train"])
        dataSet_test = MultiClassImageDataset(root_dir, image_transforms["train"])
    
    train_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=dataSet_test, shuffle=True, batch_size=1)

    d = next(iter(train_loader))
    print("Train Image" , d[0].shape)
    print("Train Label" , d[1])

    d = next(iter(test_loader))
    print("Train Image" , d[0].shape)
    print("Train Label" , d[1])

    loss_list = []  # Store loss history
    accuracies = []

    writer = SummaryWriter(SAVE_PATH)
    
    if BINARY:
        if ORTHO:
            QC_NN = CreateOrthogonalQNN4(NUM_QUBIT)
            if RESNET:
                classifierModel = ResNet50_OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT).to(device)
                classifierModel.train()
                classifierModel.resnet_encoder.eval()
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_Binary_Ortho_Classical/CL_"+ str(NUM_QUBIT) + "_"
            else:
                classifierModel = OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT , Train_Classical = True).to(device)
                classifierModel.train()
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_"+ str(NUM_QUBIT) + "_"

        else:
            
            classifierModel = ResNet50_Model_Binary(QC_NN, NUM_QUBIT).to(device)
            classifierModel.train()
            classifierModel.resnet_encoder.eval()

    if LOAD:
        print("Loading ..... ")
        classifierModel.load_state_dict(torch.load(SAVE_PATH+"239.pt", weights_only=True))
        print("Loaded model :  ",SAVE_PATH+"239.pt" )
    
    writer = SummaryWriter(SAVE_PATH)   

    # Set optimizer for training.
    optimizer = optim.Adam(classifierModel.parameters(), lr=0.0002)

    loss_func = CrossEntropyLoss()
    
    # max_val = 1000
    # p = ProgressBar(max_value=max_val, disable=False)

    # Start training
   

    for epoch in range(EPOCHS):
        total_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):
            target= target.to(device)
            
            data= data.to(device)
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = classifierModel(data)  # Forward pass

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
        writer.add_scalar('Loss/train', loss_list[-1], epoch)

            # accuracy_test = np.bool(target == torch.argmax (prediction))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / EPOCHS, loss_list[-1]))
        print("\taccuracy: ", str(np.float32(accuracy_test/max_num_test)))
        writer.add_scalar('Accuracy/train', np.float32(accuracy_test/max_num_test), epoch)
    
    
    plt.figure("Accuracy vs Epoch") # Here's the part I need
    plt.title('accuracy')
    plt.plot(accuracies, )
    plt.savefig(SAVE_PATH+ "_accuracy.png")
    
    plt.figure("Loss vs Epoch") 
    plt.title('loss')
    plt.plot(loss_list)
    plt.savefig(SAVE_PATH+ "_loss.png")





def train_quantum(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS, SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder"):
    

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

    loss_list = []  # Store loss history
    accuracies = []

    writer = SummaryWriter(SAVE_PATH)
    
    if BINARY:
        if ORTHO:
            QC_NN = CreateOrthogonalQNN4(NUM_QUBIT)
            if RESNET:
                classifierModel = ResNet50_OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT).to(device)
                classifierModel.train()
                classifierModel.resnet_encoder.eval()
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_Binary_Ortho_Classical/CL_" + str(NUM_QUBIT) + "_"
            else:
                classifierModel = OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT, Train_Classical = False).to(device)
                classifierModel.train()
                SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/CL_"+ str(NUM_QUBIT) + "_"

        else:
            
            classifierModel = ResNet50_Model_Binary(QC_NN, NUM_QUBIT).to(device)
            classifierModel.train()
            classifierModel.resnet_encoder.eval()
    
    LOAD = True # load classical trained model
    if LOAD:
        print("Loading ..... ")
        classifierModel.load_state_dict(torch.load(SAVE_PATH+"239.pt", weights_only=True))
        print("Loaded model :  ",SAVE_PATH+"239.pt" )
        SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_"+ str(NUM_QUBIT) + "_"
        if True:
            classifierModel.load_state_dict(torch.load(SAVE_PATH+"cc119.pt", weights_only=True))
            SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical_Val/Q_"+ str(NUM_QUBIT) + "_cccc"
    
    writer = SummaryWriter(SAVE_PATH)   

    # Set optimizer for training.
   

    loss_func = CrossEntropyLoss()
    loss_func = MSELoss()

    classifierModel.classical_part.eval()
    classifierModel.classical_encoder.eval()
    classifierModel.quantum_part.train()

    optimizer = optim.Adam(classifierModel.quantum_part.parameters(), lr=0.0002)
    

    for epoch in range(EPOCHS):
        total_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):
            target= target.to(device)
            
            data= data.to(device)
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = classifierModel(data)  # Forward pass

            target = process_target_1Hot(target)
            target= target.to(device)
            output =  F.sigmoid(10*output)
            # print("Target vs output",output, target)

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
        writer.add_scalar('Loss/train', loss_list[-1], epoch)

            # accuracy_test = np.bool(target == torch.argmax (prediction))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / EPOCHS, loss_list[-1]))
        print("\taccuracy: ", str(np.float32(accuracy_test/max_num_test)))
        writer.add_scalar('Accuracy/train', np.float32(accuracy_test/max_num_test), epoch)
    
    
    plt.figure("Accuracy vs Epoch") # Here's the part I need
    plt.title('accuracy')
    plt.plot(accuracies, )
    plt.savefig(SAVE_PATH+ "_accuracy.png")
    
    plt.figure("Loss vs Epoch") 
    plt.title('loss')
    plt.plot(loss_list)
    plt.savefig(SAVE_PATH+ "_loss.png")






# def train_quantum_old(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder"):
    
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sns.set_style('darkgrid')
#     root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"
    
#     root_dir_binary = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Train/"
    
    
#     image_transforms = preprocessing(IMAGE_SIZE, device)
    
#     if BINARY:
#         dataSet = MultiClassImageDataset(root_dir_binary, image_transforms["train"])
#     else:
#         dataSet = MultiClassImageDataset(root_dir, image_transforms["train"])
    
#     train_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=BATCH_SIZE)
#     test_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=1)
#     d = next(iter(train_loader))
#     print("Train Image" , d[0].shape)
#     print("Train Label" , d[1])
    
#     if BINARY:
#         if ORTHO:
#             QC_NN = CreateOrthogonalQNN6(NUM_QUBIT)
#             if RESNET:
#                 classifierModel = ResNet50_OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT).to(device)
#                 classifierModel.train()
#                 classifierModel.resnet_encoder.eval()
#                 classifierModel.layer_to_quantum.eval()
#                 SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_Binary_Ortho_Classical/"
#             else:
#                 classifierModel = OrthoModel_Binary_ClQ(QC_NN, NUM_QUBIT).to(device)
#                 classifierModel.train()
#                 SAVE_PATH = "/home/aws_install/projects/QCML/outputs/Binary_Ortho_Classical/"

#             if LOAD:
#                 print("Loading ..... ")
#                 classifierModel.load_state_dict(torch.load(SAVE_PATH+"CL_.pt", weights_only=True))
#                 print("Loaded model :  ",SAVE_PATH+"CL_.pt" )
#             SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_Binary_Ortho_Classical/Q8_"
            
#         else:
#             SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder_Binary"
#             classifierModel = ResNet50_Model_Binary(QC_NN, NUM_QUBIT).to(device)
#             if LOAD:
#                 print("Loading ..... ")
#                 classifierModel.load_state_dict(torch.load(SAVE_PATH+"19.pt", weights_only=True))
#                 print("Loaded model :  ",SAVE_PATH+"19.pt" )
#             classifierModel.train()
#             classifierModel.resnet_encoder.eval()
            

#     # Set optimizer for training.
#     optimizer = optim.Adam(classifierModel.quantum_layer.parameters(), lr=0.0002)

#     loss_func = CrossEntropyLoss()
    
#     # Start training
#     loss_list = []  # Store loss history
#     accuracies = []
#     for epoch in range(epochs):
#         total_loss = []

#         for batch_idx, (data, target) in enumerate(train_loader):
#             target= target.to(device)
            
#             data= data.to(device)
#             optimizer.zero_grad(set_to_none=True)  # Initialize gradient
#             output = classifierModel(data)  # Forward pass
            
#             loss = loss_func(output, target)  # Calculate loss
            
#             #check wights before and after optimizer step
#             # print(classifierModel.full[0])
#             # sum_w = classifierModel.full[2].weight.cpu().detach().numpy()
            
#             loss.backward()  # Backward pass
#             optimizer.step()  # Optimize weights
            
#             # check weights after update
#             # sum_w2 = classifierModel.full[2].weight.cpu().detach().numpy()
#             # print( "weight check",np.sum(sum_w - sum_w2))
            
#             total_loss.append(loss.item())  # Store loss
            
#             if batch_idx%40 == 0:
#                 print(batch_idx)
#                 torch.save(classifierModel.state_dict(), SAVE_PATH+str(epoch)+".pt")
#                 accuracy_test = 0
#                 for num_test, (test_data, test_target) in enumerate(test_loader):
#                     if num_test == max_num_test:
#                         break
#                     test_target= test_target.to(device)
#                     test_data= test_data.to(device)
                    
#                     test_pred = np.argmax(classifierModel(test_data).cpu().detach().numpy())
#                     test_target = test_target.cpu().detach().numpy()
#                     # print(str(np.int32(test_target == test_pred)))
#                     accuracy_test += np.int32(test_target == test_pred)
#                 accuracies.append(np.float32(accuracy_test/max_num_test))
                

#         loss_list.append(sum(total_loss) / len(total_loss))
#             # accuracy_test = np.bool(target == torch.argmax (prediction))
#         print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
#         print("\taccuracy: ", str(np.float32(accuracy_test/max_num_test)))
    
    
#     plt.figure("Accuracy vs Epoch") # Here's the part I need
#     plt.title('accuracy')
#     plt.plot(accuracies, )
#     plt.savefig(SAVE_PATH+ "_accuracy.png")
    
#     plt.figure("Loss vs Epoch") 
#     plt.title('loss')
#     plt.plot(loss_list)
#     plt.savefig(SAVE_PATH+ "_loss.png")



if __name__ == "__main__":
    IMAGE_SIZE = 64
    BATCH_SIZE = 1024
    NUM_QUBIT = 4
    EPOCHS = 240

    import argparse
    import sys

    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('--classical', action='store_true')
    # cmd_line = ["--my_bool", "False"]
    # parsed_args = parser.parse(cmd_line)

    args = parser.parse_args() 
    TRAIN_ENCODER = args.classical

    if TRAIN_ENCODER:
        BATCH_SIZE = 1024
        EPOCHS = 240
        print("Training Classical encoder and part....")
        train_classical_encoder(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS)
    else:
        BATCH_SIZE = 1024
        EPOCHS = 240
        print("Training Quantum part....")
        train_quantum(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS)

    # if TRAIN_REGIME:
    #     BATCH_SIZE = 1024
    #     EPOCHS = 120
    #     train_classical_encoder(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS)
    #     BATCH_SIZE = 256
    #     EPOCHS = 120
    #     train_quantum(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, EPOCHS)

    
    