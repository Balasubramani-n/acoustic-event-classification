#CORE CONFIGURATION FILE

#DATASET AND DATALOADERS 
annot_file = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
audio_dir = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
kfold = 10
num_test_folds = 1
drop_last = True 
batch_size = 32
#IMPORTS 
stmts = compile("""
#___________________________________________________________

#STANDARD IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import itertools as it
import sys 

#DEVICE AGNOSTIC CODE 

device = "cuda" if torch.cuda.is_available() else "cpu" 

#__________________________________________________________
""" , "" ,"exec")


exec(stmts) #USE THIS STATEMENT ACROSS THE MODULE TO EXECUTE COMMON STATEMENTS 

#____________________________STANDARD SIGNAL HYPER-PARAMETERS AND TRANSFORMS__________________________ 

sample_rate = 22050*4 
num_samples = 22050*4

#FOR DATA VISULIZATION
#MEL
n_fft  , hop_length , n_mels = 1024 , 512 , 64
#MFCC
n_mfcc = 64
#LFCC 
n_lfcc = 64 

#CAUTION : removing device agnostic for the transformations can affect various sections of the module
#(32 , 1 , 64 , 173)
melTransform = T.MelSpectrogram(
    sample_rate = sample_rate,
    n_fft = n_fft  ,                             #frame length
    hop_length = hop_length ,                          #half of the frame length
    n_mels = n_mels 
    ).to(device)
#(32 , 1 , 64 , 171)
mfccTransform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels , "center": False},
    ).to(device)
#(32 , 1 , 64 , 171)
lfccTransform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=n_lfcc,
    speckwargs={"n_fft": n_fft, "hop_length": hop_length, "center": False},
    ).to(device)
   

#______________________________MODELS________________________________
#use class names specified in MyModels.py , this may affect other sections of the module
currModel = "RNN_GRU"
epochs  = 50
"""
Set DIRECT_TRAIN to True if you want to train the model by running this file 
Set ENSEMBLE_TEST to True if you want to test ensemble model by running this file

"""
DIRECT_TRAIN = False #True  
ENSEMBLE_TEST = not DIRECT_TRAIN   

#valid spec options  "mfcc" , "lfcc" , "mel"  , "mfccDB" , "lfccDB" , "melDB" 
models = {
    "RNN_GRU" : {
                "params" : (171 , 32 , 8, 10),  #Class parameters
                "inDim" : (-1, 64, 171) ,       #input dim
                "spec" : "mfcc" ,              # Spectrogram type
                "optimizer" : "torch.optim.ASGD(model.parameters() , lr = 0.001)",#optimizers 
                "path" : "Trained\\RNN_GRU.pth", #model path
                "results" : "performace\\RNN_GRU" #result path
                },
    "CNN_Net" : {
                "params" : None ,  #Class parameters 
                "inDim" : (32 , 1 , 64 , 173) ,       #input dim
                "spec" : "melDB" ,              # spectrogram type  
                "optimizer" : "torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)",#optimizers 
                "path" : "Trained\\CNN_Net.pth", #model path
                "results" : "performace\\CNN_Net" #result path
                },
    }

specSet = set()
for k , v in models.items() :
    specSet.add(v['spec'])


if __name__ == "__main__" :
    if DIRECT_TRAIN :
        print("Training and testing")
        exec(open("trainingAndTesting.py").read())
    if ENSEMBLE_TEST :
        print("Ensemble test")
        exec(open("ensembleModel.py").read())
    print("___")

    
