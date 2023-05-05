#CONFIGURATION FILE

#DATASET DIRECTORY 
annot_file = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
audio_dir = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
kfold = 10
num_test_folds = 1 #80% training and 20% testing 

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

#DEVICE AGNOSTIC CODE 

device = "cuda" if torch.cuda.is_available() else "cpu" 

#__________________________________________________________
""" , "" ,"exec")


exec(stmts) #USE THIS STATEMENT ACROSS THE MODULE TO EXECUTE COMMON STATEMENTS 

#____________________________STANDARD SIGNAL HYPER-PARAMETERS AND TRANSFORMS__________________________ 

batch_size = 32
sample_rate = 22050*4 #SR/NS = 1 SEC  WE ARE GETTING 1 SEC WORTH OF AUDIO 
num_samples = 22050*4
epochs  = 200 
lr = 0.01

#CAUTION : removing device agnostic for the transformations can affect various sections of the module
#(32 , 1 , 64 , 173)
melTransform = T.MelSpectrogram(
    sample_rate = sample_rate,
    n_fft = 1024  ,                             #frame length
    hop_length = 512 ,                          #half of the frame length
    n_mels = 64 
    ).to(device)
#(32 , 1 , 64 , 171)
mfccTransform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=64,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64, "center": False},
    ).to(device)
#(32 , 1 , 64 , 171)
lfccTransform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=64,
    speckwargs={"n_fft": 1024, "hop_length": 512, "center": False},
    ).to(device)
   

#______________________________MODELS________________________________
#use class names in dict , this may affect other sections of the module
currModel = "RNN_GRU"

#valid spec options  "mfcc" , "lfcc" , "mel"
models = {
    "RNN_GRU" : {
                "params" : (171 , 32 , 8, 10),  #Class parameters
                "inDim" : (-1, 64, 171) ,       #input dim
                 "spec" : "mfcc" ,              #which spectrogram 
                 "toDB" : False ,               #conversion to decible
                "loss_fun" : "nn.CrossEntropyLoss().to(device)", # for optimizers modify the training section
                "optimizer" : "torch.optim.ASGD(model.parameters() , lr = 0.01)",
                 "path" : "C:\\Users\\nagav\\OneDrive\\Desktop\\ML_Moduled\\Trained\\RNN_GRU.pth", #model path
                 "results" : "\\performace\\RNN_GRU" #result path
                 },
    
    }

if __name__ == "__main__" :
    pass 

    
