import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

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
# The above code is importing the `models` module from the `torchvision` library in Python. This
# module contains pre-trained models for computer vision tasks that can be used for tasks such as
# image classification, object detection, and segmentation.
from torchvision import models
from collections import OrderedDict

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

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import Pauli

from torchvision.models import resnet50, ResNet50_Weights
from qiskit_algorithms.gradients.spsa.spsa_estimator_gradient import SPSAEstimatorGradient



def preprocessing (IMAGE_SIZE, device ):
    image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)) ]),
    }  
    
    return image_transforms



def CreateQNN(num_qubits = 4): 
    feature_map = ZZFeatureMap(num_qubits)
    ansatz = RealAmplitudes(num_qubits, reps=4)
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    print(qc.draw())

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN( observables = ["ZZZZ"], circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, input_gradients=True,)
    return qnn





def CreateBaseQNN(num_qubits = 4): 
    
    # from qiskit.primitives import Estimator as Estimator
    # from qiskit_aer import AerSimulator
    
    # from qiskit_aer.primitives import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    
    
    # from qiskit_ibm_runtime import EstimatorV2 as Estimator
    # from qiskit_aer import AerSimulator # former import: from qiskit import Aer
    # backend = AerSimulator()
    # estimator = Estimator(backend
    # from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    # estimator_manila = Estimator(mode=fake_manila, options=options)
    
    fake_manila = FakeManilaV2()
    options = {"simulator": {"seed_simulator": 42}}
    custom_estimator = Estimator(mode=fake_manila, options=options)
    
    
    
    # from qiskit.providers  import BasicAer
    
    # from qiskit import Aer, QuantumCircuit, execute
    # simulator = Aer.get_backend('qasm_simulator')
    # backend = AerSimulator()
    
        

        
        # # Run the sampler job locally using FakeManilaV2
        # fake_manila = FakeManilaV2()
        
        # # You can use a fixed seed to get fixed results.
        # options = {"simulator": {"seed_simulator": 42}}

    # from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    
    # from qiskit_ibm_runtime import QiskitRuntimeService
    # provider = QiskitRuntimeService(channel='ibm_quantum', token="set your own token here")
    # backend = provider.get_backend("ibm_kyoto")
    
    
    

    
    # input_parameters = ParameterVector("x", num_qubits)
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}
    print(thetas)
    circuit = QuantumCircuit(num_qubits)
    
    # amplitude encoding
    for i in range(num_qubits):
        circuit.ry(input_parameters[i], i)
        
    circuit.h(0)
    circuit.cx(1, 0)
    circuit.cx(2, 1)
    circuit.cx(3, 2)

    circuit.barrier()

    for k in range(0, 4):
        circuit.ry(thetas[k], k)

    circuit.barrier()

    # now starting with layers
    circuit.cx(3, 2)
    circuit.cx(2, 1)
    circuit.cx(1, 0)
    circuit.h(0)

    circuit.measure_all()

    
    print(circuit.draw())
    
    
    # custom_estimator = Estimator(backend=backend, observables = ['IIIZ','IIZI','IZII','ZIII'] )

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=input_parameters,
        weight_params=thetas,
        input_gradients=True,
    )
    return qnn





def CreateQNN_Ex(num_qubits = 4): 
    """
    The function `CreateQNN_Ex` creates a quantum neural network (QNN) with a specified number of
    qubits, using a feature map, ansatz, and observables for estimation.
    
    @param num_qubits The `num_qubits` parameter specifies the number of qubits in the quantum circuit.
    In this case, the default value is set to 4, but you can change it to any positive integer value
    when calling the `CreateQNN_Ex` function.
    
    @return The function `CreateQNN_Ex` is returning a Quantum Neural Network (QNN) object that is set
    up with a specific quantum circuit, feature map, ansatz, and observables. The QNN is configured for
    hybrid gradient backpropagation and is ready to be used for quantum machine learning tasks.
    """
    
    # The above code is importing the `EstimatorV2` class from the `qiskit_ibm_runtime` module in
    # Python. This class is used for estimating the resources required for running quantum circuits on
    # IBM Quantum systems.
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    # The above code is importing the `FakeManilaV2` class from the `qiskit_ibm_runtime.fake_provider`
    # module in the Qiskit IBM Runtime package. This class likely represents a fake provider for
    # simulating quantum computing resources in a development or testing environment.
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qiskit.quantum_info import Pauli

    
    fake_manila = FakeManilaV2()
    options = {"simulator": {"seed_simulator": 42}}
    custom_estimator = Estimator(mode=fake_manila, options=options)
    
    
    # The code snippet is defining a feature map called `feature_map` using the `ZZFeatureMap` class
    # with a specified number of qubits `num_qubits`.
    feature_map = ZZFeatureMap(num_qubits)
    # The above code snippet is defining a quantum circuit ansatz using RealAmplitudes for a given
    # number of qubits with 2 repetitions. This ansatz is commonly used in variational quantum
    # algorithms for quantum state preparation and optimization tasks.
    ansatz = RealAmplitudes(num_qubits, reps=2)
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # The above code is attempting to print the output of `qc.draw()`, which is likely a function or
    # method that generates a visual representation of a quantum circuit. However, the code snippet
    # provided is not complete and lacks the definition of `qc` or any relevant context to determine
    # the exact behavior.
    print(qc.draw())

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        # The above code in Python is creating a list called `observables` containing two strings as Observables ZZII and IIZZ:
        # "ZZII" and "IIZZ".
        # observables = ["ZIII","IZII","IIZI","IIIZ"],
        observables = [Pauli("ZZII"),Pauli("IIZZ")]
    )
    return qnn


# def encode_amplitudes(X):
#         print("\n\n testing: ",X)

#         A = X.clone()

#         A[:,0] = torch.arccos( X[:,0] )

#         t1 =  X[:,1] / torch.sin(A[:,0])
#         A[:,1] = torch.arccos( t1 )
        
#         t2 = X[:,2]/(torch.sin(A[:,0])*torch.sin(A[:,1])) 
#         A[:,2] = torch.arccos(t2) 

#         print("\n\n\n\n A: ")
#         return A


def RBS_Loader(thetas, NUM_QUBITS = 4):
    # thetas = encode_amplitudes(thetas)
    loader = qiskit.QuantumCircuit(NUM_QUBITS)
    loader.x(0)
    for i in range(NUM_QUBITS-1):
        loader.append(RBS(thetas[i]), [i, i+1])
    loader.draw("mpl")

    return loader


def RBS_Pyramid_Trainable(NUM_QUBITS = 4):


    # thetas = {k : Parameter('T_L1'+str(k))for k in range(3)}

    num_param = NUM_QUBITS * (NUM_QUBITS-1) / 2  - (NUM_QUBITS-1)
    num_param2 = (NUM_QUBITS * (NUM_QUBITS-1) / 2)

    print(num_param2, num_param )
    thetas = {k : Parameter('T_L1'+str(k))for k in range(int(num_param))}
    thetas_layer2 = {k : Parameter('T_L2'+str(k))for k in range(int(num_param2))}
    print(num_param2, num_param , thetas )

    ansatz = qiskit.QuantumCircuit(NUM_QUBITS)    

    # layer 1 without data loaders
    k = 0
    for i in range(NUM_QUBITS-2):
        for j in range (NUM_QUBITS-2-i):
            ansatz.append(RBS(thetas[k]), [j, j+1])
            k+=1
    
    # layer 2 full v
    k = 0
    for i in range(NUM_QUBITS-1):
        for j in range (NUM_QUBITS-1-i):
            ansatz.append(RBS(thetas_layer2[k]), [j, j+1])
            k+=1
    
    # for i in range(3):
    #     ansatz.append(RBS(thetas_layer2[i]), [i, i+1])
    # for i in range(2):
    #     ansatz.append(RBS(thetas_layer2[i]), [i, i+1])
    
    # ansatz.append(RBS(thetas_layer2[2]), [0, 1])

    return ansatz


def RBS_Pyramid_Trainable_3layer(NUM_QUBITS = 4):


    # thetas = {k : Parameter('T_L1'+str(k))for k in range(3)}

    num_param = NUM_QUBITS * (NUM_QUBITS-1) / 2  - (NUM_QUBITS-1)
    num_param2 = (NUM_QUBITS * (NUM_QUBITS-1) / 2)

    print(num_param2, num_param )
    thetas = {k : Parameter('T_L1'+str(k))for k in range(int(num_param))}
    thetas_layer2 = {k : Parameter('T_L2'+str(k))for k in range(int(num_param2))}
    thetas_layer3 = {k : Parameter('T_L3'+str(k))for k in range(int(num_param32))}
    print(num_param2, num_param , thetas )

    ansatz = qiskit.QuantumCircuit(NUM_QUBITS)    

    # layer 1 without data loaders
    k = 0
    for i in range(NUM_QUBITS-2):
        for j in range (NUM_QUBITS-2-i):
            ansatz.append(RBS(thetas[k]), [j, j+1])
            k+=1
    
    # layer 2 full v
    k = 0
    for i in range(NUM_QUBITS-1):
        for j in range (NUM_QUBITS-1-i):
            ansatz.append(RBS(thetas_layer2[k]), [j, j+1])
            k+=1

    # layer 3 full v
    k = 0
    for i in range(NUM_QUBITS-1):
        for j in range (NUM_QUBITS-1-i):
            ansatz.append(RBS(thetas_layer3[k]), [j, j+1])
            k+=1
    
    # for i in range(3):
    #     ansatz.append(RBS(thetas_layer2[i]), [i, i+1])
    # for i in range(2):
    #     ansatz.append(RBS(thetas_layer2[i]), [i, i+1])
    
    # ansatz.append(RBS(thetas_layer2[2]), [0, 1])

    return ansatz

def RBS_Trainable(NUM_QUBITS = 4):
    thetas = {k : Parameter('Theta'+str(k))for k in range(3)}

    ansatz = qiskit.QuantumCircuit(NUM_QUBITS)    
    for i in range(2):
            ansatz.append(RBS(thetas[i]), [i, i+1])
    
    ansatz.append(RBS(thetas[2]), [0, 1])
    ansatz.draw("mpl")

    return ansatz



def RBS(theta):
  qc = qiskit.QuantumCircuit(2, name=f"RBS")
  theta_param = qiskit.circuit.Parameter("θ")
  qc.h(0)
  qc.h(1)
  qc.cz(0, 1)
  qc.ry(theta_param, 0)
  qc.ry(-theta_param, 1)
  qc.cz(0, 1)
  qc.h(0)
  qc.h(1)
  qiskit_gate = qc.to_gate(parameter_map={theta_param: theta})
  return qiskit_gate


def CreateOrthogonalQNN(num_qubits = 4): 
    from qiskit.circuit import Parameter
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}

    print("\n\n\n\n", input_parameters)

    feature_map = RBS_Loader(input_parameters, num_qubits)
    ansatz = RBS_Trainable(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    print(qc.draw())

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        # observables = [Pauli("ZZII"),Pauli("IIZZ")]
    )
    return qnn


def CreateOrthogonalQNN8(num_qubits = 8): 
    from qiskit.circuit import Parameter
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}

    print("\n\n\n\n", input_parameters)

    feature_map = RBS_Loader(input_parameters, num_qubits)
    ansatz = RBS_Pyramid_Trainable(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    qc.draw("mpl")

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        observables = [Pauli("ZIIIIIII"),Pauli("IZIIIIII"),Pauli("IIZIIIII"), Pauli("IIIZIIII"), Pauli("IIIIZIII"), Pauli("IIIIIZII"), Pauli("IIIIIIZI")]
    )
    return qnn


def CreateOrthogonalQNN6(num_qubits = 6): 
    from qiskit.circuit import Parameter
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}

    print("\n\n\n\n", input_parameters)

    feature_map = RBS_Loader(input_parameters, num_qubits)
    ansatz = RBS_Pyramid_Trainable(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    qc.draw("mpl")

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        observables = [Pauli("ZIIIII"),Pauli("IZIIII"),Pauli("IIZIII"), Pauli("IIIZII"), Pauli("IIIIZI"),Pauli("IIIIIZ")]
    )
    return qnn



def CreateOrthogonalQNN4(num_qubits = 4, SAMPLER = False): 
    from qiskit.circuit import Parameter
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}

    print("\n\n\n\n", input_parameters)

    feature_map = RBS_Loader(input_parameters, num_qubits)
    # ansatz = RBS_Trainable(num_qubits)
    ansatz = RBS_Pyramid_Trainable(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    qc.draw("mpl")
    if SAMPLER: 
        from qiskit.primitives import Sampler as basicSampler
        shots = 1000
        sampler = basicSampler(options={"shots": shots, "seed": algorithm_globals.random_seed})
        qnn = SamplerQNN(
            circuit=qc,
            sampler = sampler,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            # observables = [Pauli("ZIII"),Pauli("IZII"),Pauli("IIZI"), Pauli("IIIZ")]
        )
    else:
        from qiskit.primitives import Estimator as basicEstimator
        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            observables = [Pauli("ZIII"),Pauli("IZII"),Pauli("IIZI"), Pauli("IIIZ")],
            gradient=SPSAEstimatorGradient(estimator=basicEstimator(), epsilon=0.01),
        )
    return qnn


def CreateOrthogonalQNN4_2(num_qubits = 4, SAMPLER = False): 
    from qiskit.circuit import Parameter
    input_parameters  = {k : Parameter('X'+str(k))for k in range(num_qubits)}
    thetas = {k : Parameter('Theta'+str(k))for k in range(num_qubits)}

    print("\n\n\n\n", input_parameters)

    feature_map = RBS_Loader(input_parameters, num_qubits)
    # ansatz = RBS_Trainable(num_qubits)
    ansatz = RBS_Pyramid_Trainable(num_qubits)
    # ansatz2 = RBS_Pyramid_Trainable(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    qc.draw("mpl")
    if SAMPLER: 
        from qiskit.primitives import Sampler as basicSampler
        shots = 1000
        sampler = basicSampler(options={"shots": shots, "seed": algorithm_globals.random_seed})
        qnn = SamplerQNN(
            circuit=qc,
            sampler = sampler,
            input_params=feature_map.parameters,
            weight_params= ansatz.parameters,
            input_gradients=True,
            # observables = [Pauli("ZIII"),Pauli("IZII"),Pauli("IIZI"), Pauli("IIIZ")]
        )
    else:
        from qiskit.primitives import Estimator as basicEstimator
        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            observables = [Pauli("ZIII"),Pauli("IZII"),Pauli("IIZI"), Pauli("IIIZ")],
            gradient=SPSAEstimatorGradient(estimator=basicEstimator(), epsilon=0.001),
        )
    return qnn



class MultiClassImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.image_paths = []
        self.class_folders = ['SeaLake', 'Industrial', 'Residential', 'Pasture']
        # self.class_folders = ['Industrial', 'Pasture', 'Residential', 'SeaLake'] # fixing the order of labels
        self.labels = []

        class_length = {}
        for label, class_folder in enumerate(self.class_folders):
            img_paths = glob.glob(os.path.join(root_dir, class_folder, '*.jpg'))
            class_length[class_folder] = int(len(img_paths))
            self.image_paths.extend(img_paths)
            self.labels.extend([label] * len(img_paths))
        
        plt.figure(figsize=(15,8)) 
        self.plot_from_dict(class_length, plot_title="Entire Dataset (before train/val/test split)")

    def plot_from_dict(self, dict_obj, plot_title, **kwargs):
        return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    

class MultiClassImageDataset6_all(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.image_paths = []
        self.class_folders = ['SeaLake', 'Industrial', 'Residential', 'Pasture', "Highway", "River"]
        # self.class_folders = ['Industrial', 'Pasture', 'Residential', 'SeaLake'] # fixing the order of labels
        self.labels = []

        class_length = {}
        root_dir_bckup =  root_dir 
        root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"

        for label, class_folder in enumerate(self.class_folders):    
            if class_folder in ['SeaLake', 'Industrial', 'Residential', 'Pasture']:
                root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/Small_Train/"
            else:
                root_dir = "/home/aws_install/projects/QCML/QNN4EO/Dataset/EuroSAT_RGB/"

            img_paths = glob.glob(os.path.join(root_dir, class_folder, '*.jpg'))
            class_length[class_folder] = int(len(img_paths))
            self.image_paths.extend(img_paths)
            self.labels.extend([label] * len(img_paths))
        
        plt.figure(figsize=(15,8)) 
        self.plot_from_dict(class_length, plot_title="Entire Dataset (before train/val/test split)")

    def plot_from_dict(self, dict_obj, plot_title, **kwargs):
        return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    

class MultiClassImageDataset6(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.image_paths = []
        self.class_folders = ['SeaLake', 'Industrial', 'Residential', 'Pasture', "Highway", "River"]
        # self.class_folders = ['Industrial', 'Pasture', 'Residential', 'SeaLake'] # fixing the order of labels
        self.labels = []

        class_length = {}
        for label, class_folder in enumerate(self.class_folders):
            img_paths = glob.glob(os.path.join(root_dir, class_folder, '*.jpg'))
            class_length[class_folder] = int(len(img_paths))
            self.image_paths.extend(img_paths)
            self.labels.extend([label] * len(img_paths))
        
        plt.figure(figsize=(15,8)) 
        self.plot_from_dict(class_length, plot_title="Entire Dataset (before train/val/test split)")

    def plot_from_dict(self, dict_obj, plot_title, **kwargs):
        return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    


# Define torch NN module
class Base_Model(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        super().__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=5)
        self.conv2 = Conv2d(16, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(16, 64) # 2704 for 64 , 16 for 16
        self.fc2 = Linear(64, NUM_QUBIT)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 10)  # 1-dimensional output from QNN
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return self.activation(x)
    


class ProgressBar(object):  # Python3+ '(object)' can be omitted
    def __init__(self, max_value, disable=True):
        self.max_value = max_value
        self.disable = disable
        self.p = self.pbar()

    def pbar(self):
        return tqdm(
            total=self.max_value,
            desc='Loading: ',
            disable=self.disable
        )

    def update(self, update_value):
        self.p.update(update_value)

    def close(self):
        self.p.close()



    
    # Define torch NN module
class ResNet50_Model(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        # self.resnet_encoder = resnet50(weights=weights)
        self.resnet_encoder = models.resnet50(pretrained=True)
        
        self.dropout = Dropout2d()
        self.fc1 = Linear(1000, 256) # 2704 for 64 , 16 for 16
        self.fc2 = Linear(256, NUM_QUBIT)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 10)  # 1-dimensional output from QNN
        # self.activation = nn.Sigmoid()
        
        
        
        self.full = nn.Sequential(OrderedDict([
          ('resnet50',  self.resnet_encoder),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
          ('fc2', self.fc2),
          ('qnn', self.qnn),
          ('fc3', self.fc3), 
        #   ('act', self.activation),
          
        ]))
        
        for param in self.full.resnet50.parameters():
            param.requires_grad = False
            
        

    def forward(self, x):
        x = self.full(x)
        return x
        


# This class is binary ResNet50 model implemented in Python.
class ResNet50_Model_Binary(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        """
        This function initializes a neural network model with a ResNet50 encoder, dropout layer, fully
        connected layers, a quantum neural network connector, and sets ResNet50 parameters to not
        require gradients.
        
        @param QC_NN It looks like the code snippet you provided is a part of a neural network model
        initialization in PyTorch. The `__init__` method of a custom neural network class is being
        defined here.
        @param NUM_QUBIT NUM_QUBIT is a parameter that represents the number of qubits in the quantum
        neural network (QNN) model. It is used to define the size of the input layer of the fully
        connected layer (`fc1`) in the neural network architecture you provided.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        # self.resnet_encoder = resnet50(weights=weights)
        self.resnet_encoder = models.resnet50(pretrained=True)
        
        self.dropout = Dropout2d()
        # self.fc1 = Linear(1000, 256) # 2704 for 64 , 16 for 16
        self.fc1 = Linear(1000, NUM_QUBIT)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc2 = Linear(2, 2)  # 1-dimensional output from QNN
        # self.activation = nn.Sigmoid()
        
        
        
        self.full = nn.Sequential(OrderedDict([
          ('resnet50',  self.resnet_encoder),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
          ('qnn', self.qnn),
          ('fc2', self.fc2), 
        #   ('act', self.activation),
          
        ]))
        
        for param in self.full.resnet50.parameters():
            param.requires_grad = False
            
        

    def forward(self, x):
        x = self.full(x)
        return x



class ResNet50_OrthoModel_Binary_ClQ(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        """
        This function initializes a neural network model with a ResNet50 encoder, dropout layer, fully
        connected layers, a quantum neural network connector, and sets ResNet50 parameters to not
        require gradients.
        
        @param QC_NN It looks like the code snippet you provided is a part of a neural network model
        initialization in PyTorch. The `__init__` method of a custom neural network class is being
        defined here.
        @param NUM_QUBIT NUM_QUBIT is a parameter that represents the number of qubits in the quantum
        neural network (QNN) model. It is used to define the size of the input layer of the fully
        connected layer (`fc1`) in the neural network architecture you provided.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        # self.resnet_encoder = resnet50(weights=weights)
        self.resnet_encoder = models.resnet50(pretrained=True)
        
        self.dropout = Dropout2d()
        self.fc1 = Linear(2048, NUM_QUBIT-1)  # 2-dimensional input to QNN

        self.clnn = Linear(NUM_QUBIT-1, NUM_QUBIT)   # will use eigen valyues from all qubit.
        self.qnn = TorchConnector(QC_NN) # Apply torch connector, weights chosen

        self.fc2 = Linear(NUM_QUBIT, 2)  # 1-dimensional output from QNN
        self.activation = nn.ReLU()
        
        self.feature_extractor = FeatureExtractor()

        self.encoding_hook = self.resnet_encoder.avgpool.register_forward_hook(self.feature_extractor )   
        
        self.full = nn.Sequential(OrderedDict([
          ('resnet50',  self.resnet_encoder),
          ('fc1', self.fc1),
          ('clnn', self.clnn),
          ('fc2', self.fc2), 
        ]))
        
        self.layer_to_quantum= nn.Sequential(OrderedDict([
          ('flatten',  nn.Flatten(start_dim=1)),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
        ]))
        self.quantum_layer = nn.Sequential(OrderedDict([
          ('clnn', self.qnn),
          ('fc2', self.fc2), 
        #   ("act", nn.Softmax()),   
        ]))

        self.classical_layers = nn.Sequential(OrderedDict([
          ('clnn', self.clnn),
          ('fc2', self.fc2), 
        #   ("act", nn.Softmax()),
          ]))
        
        for param in self.full.resnet50.parameters():
            param.requires_grad = False
    
    def encode_amplitudes(self, X):
        from torch import arccos, sin
        minVal = 1e-8
        A = X.clone()
        X = torch.clamp(X, min=-1+minVal, max=1-minVal)
        A[:,0] = arccos( X[:,0] )

        t1 =  X[:,1] / sin( arccos( X[:,0] ) )
        t1 = torch.clamp(t1, min=-1+minVal, max=1-minVal)
        A[:,1] = arccos( t1 )
        
        t2 = X[:,2] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) ) 
        t2 = torch.clamp(t2, min=-1+minVal, max=1-minVal)
        A[:,2] = arccos( t2 ) 

        return A

    def forward(self, x):

        self.encoding_full  =  self.resnet_encoder(x)
        encoding_bottleneck = self.feature_extractor.extracted_features

        x =  self.layer_to_quantum(encoding_bottleneck)
        norm = torch.norm(x, dim=1, keepdim=True)
        # Normalize each data point in the batch
        x = x / (norm+1e-8)
        x =  self.classical_layers(x)

        # print("X: \n", x, x.shape)
        # x = encode_amplitudes_8Qubit(x)
        # print("A: \n", x, x.shape)
        # x = self.quantum_layer(x)
        
        return x
    


def encode_amplitudes_8Qubit(self, X):

    from torch import arccos, sin
    minVal = 1e-8
    A = X.clone()
    X = torch.clamp(X, min=-1+minVal, max=1-minVal)
    A[:,0] = arccos( X[:,0] )

    t1 =  X[:,1] / sin( arccos( X[:,0] ) )
    t1 = torch.clamp(t1, min=-1+minVal, max=1-minVal)
    A[:,1] = arccos( t1 )
    
    t2 = X[:,2] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) ) 
    t2 = torch.clamp(t2, min=-1+minVal, max=1-minVal)
    A[:,2] = arccos( t2 ) 

    t3 = X[:,3] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) ) 
    t3 = torch.clamp(t3, min=-1+minVal, max=1-minVal)
    A[:,3] = arccos( t3 ) 

    t4 = X[:,4] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) ) 
    t4 = torch.clamp(t4, min=-1+minVal, max=1-minVal)
    A[:,4] = arccos( t4 ) 

    t5 = X[:,5] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) * sin( arccos( t4 ) ) ) 
    t5 = torch.clamp(t5, min=-1+minVal, max=1-minVal)
    A[:,5] = arccos( t5 ) 

    t6 = X[:,6] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) * sin( arccos( t4 ) )  * sin( arccos( t5 ) )) 
    t6 = torch.clamp(t6, min=-1+minVal, max=1-minVal)
    A[:,6] = arccos( t6 ) 

    
    return A



class OrthoModel_Binary_ClQ(Module):
    def __init__(self, QC_NN, NUM_QUBIT, Train_Classical):
        """
        This function initializes a neural network model with a ResNet50 encoder, dropout layer, fully
        connected layers, a quantum neural network connector, and sets ResNet50 parameters to not
        require gradients.
        
        @param QC_NN It looks like the code snippet you provided is a part of a neural network model
        initialization in PyTorch. The `__init__` method of a custom neural network class is being
        defined here.
        @param NUM_QUBIT NUM_QUBIT is a parameter that represents the number of qubits in the quantum
        neural network (QNN) model. It is used to define the size of the input layer of the fully
        connected layer (`fc1`) in the neural network architecture you provided.
        """
        super().__init__()
        self.NUM_QUBIT = NUM_QUBIT
        self.Train_Classical = Train_Classical
        self.conv1 = Conv2d(3, 64, kernel_size=3)
        self.dropout1 = Dropout2d()
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(64, 64, kernel_size=5)
        self.dropout2 = Dropout2d()
        self.maxpool2 = MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(64, 128, kernel_size=5)
        self.dropout3 = Dropout2d()
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.conv4 = Conv2d(128, 256, kernel_size=3)
        self.dropout4 = Dropout2d()
        self.maxpool4 = MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten(start_dim=1)
        latent_dim = 256
        self.fc1 = Linear(latent_dim, NUM_QUBIT-1) 

        self.classical_encoder = nn.Sequential(OrderedDict([
          ('conv1', self.conv1),
          ('drop1', self.dropout1), 
          ('maxpool1', self.maxpool1), 
          
          ('norm1', nn.BatchNorm2d(64)),
          ('relu1', self.relu),  

          ('conv2', self.conv2),
          ('drop2', self.dropout2), 
          ('maxpool2', self.maxpool2), 
          
          ('norm2', nn.BatchNorm2d(64)), 
          ('relu2', self.relu), 

          ('conv3', self.conv3),
          ('drop3', self.dropout3), 
          ('maxpool3', self.maxpool3),
          
          ('norm3', nn.BatchNorm2d(128)), 
          ('relu3', self.relu), 

          ('conv4', self.conv4),
          ('drop4', self.dropout4), 
          ('maxpool4', self.maxpool4), 
          
          ('relu4', self.relu), 

          ('flatten', self.flat),
          ('fc1', self.fc1), 

          ]))
        
        
        self.clnn = Linear(NUM_QUBIT-1, NUM_QUBIT) 
        self.qnn = TorchConnector(QC_NN)
        self.activation = nn.Softmax()

        self.classical_part = nn.Sequential(OrderedDict([
          ('clnn', self.clnn), 
        #   ("activation", self.activation)
          
          ]))
        
        self.quantum_part = nn.Sequential(OrderedDict([
          ('qnn', self.qnn), 
        #   ("activation", self.activation)
          
          ]))


    def encode_amplitudes_NQubit(self, X, N=4):

        from torch import arccos, sin
        minVal = 1e-8
        A = X.clone()
        X = torch.clamp(X, min=-1+minVal, max=1-minVal)

        all_t = []
        for i in range( N-1 ):
            new_t = X[:,i]
            for t in all_t:
                new_t = new_t/ sin(t)
            new_t = torch.clamp(new_t, min=-1+minVal, max=1-minVal)
            a_tmp = arccos( new_t )
            all_t.append(a_tmp)
            A[:,i] = a_tmp
        
        return A

    
    def encode_amplitudes_8Qubit(self, X):


        from torch import arccos, sin
        minVal = 1e-8
        A = X.clone()
        X = torch.clamp(X, min=-1+minVal, max=1-minVal)
        A[:,0] = arccos( X[:,0] )

        t1 =  X[:,1] / sin( arccos( X[:,0] ) )
        t1 = torch.clamp(t1, min=-1+minVal, max=1-minVal)
        A[:,1] = arccos( t1 )
        
        t2 = X[:,2] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) ) 
        t2 = torch.clamp(t2, min=-1+minVal, max=1-minVal)
        A[:,2] = arccos( t2 ) 

        t3 = X[:,3] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) ) 
        t3 = torch.clamp(t3, min=-1+minVal, max=1-minVal)
        A[:,3] = arccos( t3 ) 

        t4 = X[:,4] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) ) 
        t4 = torch.clamp(t4, min=-1+minVal, max=1-minVal)
        A[:,4] = arccos( t4 ) 

        t5 = X[:,5] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) * sin( arccos( t4 ) ) ) 
        t5 = torch.clamp(t5, min=-1+minVal, max=1-minVal)
        A[:,5] = arccos( t5 ) 

        t6 = X[:,6] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) * sin( arccos( t2 ) ) * sin( arccos( t3 ) ) * sin( arccos( t4 ) )  * sin( arccos( t5 ) )) 
        t6 = torch.clamp(t6, min=-1+minVal, max=1-minVal)
        A[:,6] = arccos( t6 ) 

        
        return A

    def forward(self, x):

        x =  self.classical_encoder(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        # Normalize each data point in the batch
        x = x / (norm+1e-8)
        if self.Train_Classical:
            x =  self.classical_part(x)
        else:
            x = self.encode_amplitudes_NQubit(x, self.NUM_QUBIT)
            x = self.quantum_part(x)
        
        return x
    





class ResNet50_OrthoModel_Binary_quantum(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        """
        This function initializes a neural network model with a ResNet50 encoder, dropout layer, fully
        connected layers, a quantum neural network connector, and sets ResNet50 parameters to not
        require gradients.
        
        @param QC_NN It looks like the code snippet you provided is a part of a neural network model
        initialization in PyTorch. The `__init__` method of a custom neural network class is being
        defined here.
        @param NUM_QUBIT NUM_QUBIT is a parameter that represents the number of qubits in the quantum
        neural network (QNN) model. It is used to define the size of the input layer of the fully
        connected layer (`fc1`) in the neural network architecture you provided.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        # self.resnet_encoder = resnet50(weights=weights)
        self.resnet_encoder = models.resnet50(pretrained=True)
        
        self.dropout = Dropout2d()
        # self.fc1 = Linear(1000, 256) # 2704 for 64 , 16 for 16
        self.fc1 = Linear(2048, NUM_QUBIT-1)  # 2-dimensional input to QNN
        # self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc2 = Linear(NUM_QUBIT-1, 2)  # 1-dimensional output from QNN
        self.activation = nn.ReLU()
        
        self.feature_extractor = FeatureExtractor()
        # self.encoding_hook = self.resnet_encoder.layer4.register_forward_hook(self.feature_extractor )   
        self.encoding_hook = self.resnet_encoder.avgpool.register_forward_hook(self.feature_extractor )   
        
        self.full = nn.Sequential(OrderedDict([
          ('resnet50',  self.resnet_encoder),
          ('fc1', self.fc1),
          ('qnn', self.qnn),
          ('fc2', self.fc2), 
        #   ('act', self.activation),
          
        ]))
        
        self.last_layers1 = nn.Sequential(OrderedDict([
          ('flatten',  nn.Flatten(start_dim=1)),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
        #   ('act', self.activation),
          
        ]))
        self.last_layers2 = nn.Sequential(OrderedDict([
          ('qnn', self.cnn),
          ('fc2', self.fc2), 
        #   ('act', self.activation),
          
        ]))
        
        for param in self.full.resnet50.parameters():
            param.requires_grad = False
            

    def forward(self, x):

        # extracted_features = output
        self.encoding_full  =  self.resnet_encoder(x)
        # encoding_hook = self.resnet_encoder.layer4.register_forward_hook(self.feature_extractor )  
        encoding_bottleneck = self.feature_extractor.extracted_features
        # x = self.full(x)
        x =  self.last_layers1(encoding_bottleneck)
        norm = torch.norm(x, dim=1, keepdim=True)
        # Normalize each data point in the batch
        x = x / norm
        # print("\n\n\n\n X: ",x, x.shape)
        # x = self.encode_amplitudes(x)
        # print("\n\n\n\n A: ",x, x.shape)
        x =  self.last_layers2(x)
        return x





class ResNet50_OrthoModel_Binary(Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        """
        This function initializes a neural network model with a ResNet50 encoder, dropout layer, fully
        connected layers, a quantum neural network connector, and sets ResNet50 parameters to not
        require gradients.
        
        @param QC_NN It looks like the code snippet you provided is a part of a neural network model
        initialization in PyTorch. The `__init__` method of a custom neural network class is being
        defined here.
        @param NUM_QUBIT NUM_QUBIT is a parameter that represents the number of qubits in the quantum
        neural network (QNN) model. It is used to define the size of the input layer of the fully
        connected layer (`fc1`) in the neural network architecture you provided.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        # self.resnet_encoder = resnet50(weights=weights)
        self.resnet_encoder = models.resnet50(pretrained=True)
        
        self.dropout = Dropout2d()
        # self.fc1 = Linear(1000, 256) # 2704 for 64 , 16 for 16
        self.fc1 = Linear(2048, NUM_QUBIT-1)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(QC_NN)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc2 = Linear(1, 2)  # 1-dimensional output from QNN
        self.activation = nn.ReLU()
        
        self.feature_extractor = FeatureExtractor()
        # self.encoding_hook = self.resnet_encoder.layer4.register_forward_hook(self.feature_extractor )   
        self.encoding_hook = self.resnet_encoder.avgpool.register_forward_hook(self.feature_extractor )   
        
        self.full = nn.Sequential(OrderedDict([
          ('resnet50',  self.resnet_encoder),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
          ('qnn', self.qnn),
          ('fc2', self.fc2), 
        #   ('act', self.activation),
          
        ]))
        
        self.last_layers1 = nn.Sequential(OrderedDict([
          ('flatten',  nn.Flatten(start_dim=1)),
          ('act_relu',  nn.ReLU()),
          ('fc1', self.fc1),
        #   ('act', self.activation),
          
        ]))
        self.last_layers2 = nn.Sequential(OrderedDict([
          ('qnn', self.qnn),
          ('fc2', self.fc2), 
        #   ('act', self.activation),
          
        ]))
        
        for param in self.full.resnet50.parameters():
            param.requires_grad = False
            


    # def encode_amplitudes(self, X):
    #     A = X.clone()
    #     lista = []
    #     print(X.shape)
    #     for i in range(X.shape[1]):
    #         aa = X[:,i]
    #         for a in lista:
    #             aa = aa * torch.asinh(a)
    #         a = torch.arccos(aa)
    #         print("\n\n",a)
    #         lista.append(a)
    #         A[:,i] = a

    #     return A

    def encode_amplitudes(self, X):
        from torch import arccos, sin
        minVal = 1e-8
        A = X.clone()
        X = torch.clamp(X, min=-1+minVal, max=1-minVal)
        A[:,0] = arccos( X[:,0] )

        t1 =  X[:,1] / sin( arccos( X[:,0] ) )
        t1 = torch.clamp(t1, min=-1+minVal, max=1-minVal)
        A[:,1] = arccos( t1 )
        
        t2 = X[:,2] / ( sin( arccos( X[:,0] ) ) * sin( arccos( t1 ) ) ) 
        t2 = torch.clamp(t2, min=-1+minVal, max=1-minVal)
        A[:,2] = arccos( t2 ) 

        return A



    def forward(self, x):

        # extracted_features = output
        self.encoding_full  =  self.resnet_encoder(x)
        # encoding_hook = self.resnet_encoder.layer4.register_forward_hook(self.feature_extractor )  
        encoding_bottleneck = self.feature_extractor.extracted_features
        # x = self.full(x)
        x =  self.last_layers1(encoding_bottleneck)
        norm = torch.norm(x, dim=1, keepdim=True)
        # Normalize each data point in the batch
        x = x / norm
        # print("\n\n\n\n X: ",x, x.shape)
        # x = self.encode_amplitudes(x)
        # print("\n\n\n\n A: ",x, x.shape)
        x =  self.last_layers2(x)
        return x
    


        
        
        
class Net1(nn.Module):
    def __init__(self, QC_NN, NUM_QUBIT):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        #self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(2304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)

        self.fc4 = nn.Linear(2304, 1*NUM_QUBIT)
        
        self.qc = TorchConnector(QC_NN)
        # self.qc = TorchCircuit.apply

        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(-1, 2304)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = F.relu(self.fc2(x))

        #x = F.relu(self.fc3(x))

        x = self.fc4(x)
        x = np.pi*torch.tanh(x)
        
        x = self.qc(x[0]) # QUANTUM LAYER
        
        x = F.relu(x)
        #print('output of QC = {}'.format(x))
        
#         # softmax rather than sigmoid
        x = self.fc5(x.float())
        #print('output of Linear(1, 2): {}'.format(x))
        # x = F.softmax(x, 1)

        #x = torch.sigmoid(x)
        #x = torch.cat((x, 1-x), -1)
        return x
    
    
    def predict(self, x):
        # apply softmax
        pred = self.forward(x)
#         print(pred)
        ans = torch.argmax(pred[0]).item()
        return torch.tensor(ans)
    


class FeatureExtractor:
    def __init__(self):
        self.extracted_features = None

    def __call__(self, module, input_, output):
        self.extracted_features = output
