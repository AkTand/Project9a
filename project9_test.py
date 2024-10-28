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


from utils import MultiClassImageDataset, Base_Model, preprocessing, CreateQNN,ResNet50_Model,ProgressBar

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

def main(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT, SAVE_PATH = "/home/aws_install/projects/QCML/outputs/resnet50_encoder"):
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sns.set_style('darkgrid')
    root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"
    
    image_transforms = preprocessing(IMAGE_SIZE, device)
    dataSet = MultiClassImageDataset(root_dir, image_transforms["train"])
    
    train_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=dataSet, shuffle=True, batch_size=1)
    d = next(iter(train_loader))
    print("Train Image" , d[0].shape)
    print("Train Label" , d[1])
    
    QC_NN = CreateQNN(NUM_QUBIT)
    multiClassBaseModel = Base_Model(QC_NN, NUM_QUBIT).to(device)
    multiClassResnetModel = ResNet50_Model(QC_NN, NUM_QUBIT).to(device)
    multiClassResnetModel.eval()
    multiClassResnetModel.resnet_encoder.eval()

    multiClassBaseModel.eval()  # Set model to training mode

    
    classifierModel = multiClassResnetModel

    optimizer = optim.Adam(classifierModel.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()
    
    # max_val = 1000
    # p = ProgressBar(max_value=max_val, disable=False)

    # Start training
    epochs = 10  # Set number of epochs
    loss_list = []  # Store loss history
    total_loss= []
    classifierModel.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    classifierModel.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        target= target.to(device)
        print(batch_idx)
        data= data.to(device)
        output = classifierModel(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print("Testing [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (1 + 1) / epochs, loss_list[-1]))


if __name__ == "__main__":
    IMAGE_SIZE = 64
    BATCH_SIZE = 256
    NUM_QUBIT = 4
    main(IMAGE_SIZE, BATCH_SIZE, NUM_QUBIT)