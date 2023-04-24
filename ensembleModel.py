#MODULE1 : HYBRID MODEL(Model averaging)

"""
In this approach, you can average the predictions or outputs of multiple models to
create a combined prediction or output. This can be done at inference time by
passing the same input through multiple models and averaging their predictions,
or during training by averaging the outputs of multiple models during forward
pass or aggregating their gradients during backward pass.

"""
#hello world
TRAIN = True


import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader , Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


import us8k_dataset as ds1
import rnn_model as M1
import MyMetrics as met

trLossList , tstLossList  , trAcc , testAcc = [] , [] , [] , [] 
ACCURACY = [] 
BATCH_SIZE = 32
EPOCHS  = 10
LR = 0.001
ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\set6.csv"
AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050*4 
NUM_SAMPLES = 22050*4

INP = 171            #INPUTS 
HID = BATCH_SIZE     #NUMBER OF HIDDEN LAYERS 
NL  = 8              #NUMBER OF LAYERS 
NU  = 1              #NUMBER OF UNITS 
NCLS = 10            #NUMBER OF CLASSES
NH = 16

PT_FILE = "Trained\\rnnV3.pth"

# Define multiple models

# Define averaging function
def average_models(models, x):
    outputs = [model(x.view(*v)) for model , v in models.items()]
    combined_output = torch.mean(torch.stack(outputs), dim=0)
    return combined_output

def test_single_epoch(models , data_loader , loss_fun , device ) :
    for k , v  in models.items():
        k.eval() 
    with torch.inference_mode() :
        for inp , tar in data_loader :
            inp , tar = inp.to(device) , tar.to(device)
            logits = average_models(models, inp)
            loss = loss_fun(logits , tar)
            acc = met.acc(logits , tar)
        print(f"loss :{loss} testing accuracy {acc}")

def evaluate(models , data_loader , loss_fun ,  device , epochs ):

    for i in tqdm(range(epochs) , desc = "->"):
        test_single_epoch(models ,  data_loader  , loss_fun , device)
    print("Testing finished")

# Call the averaging function during inference
if __name__  == "__main__" :

    


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device using :{device}")
                                                           
    transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64, "center": False},
        )

    
    usd = ds1.UrbanSoundDataset(
        ANNOTATIONS_FILE ,
        AUDIO_DIR ,
        transform ,
        SAMPLE_RATE ,
        NUM_SAMPLES ,
        device
        )
    #creating a custom dataloader and splitting the data 
    
    data_loader = DataLoader(usd , batch_size = BATCH_SIZE,shuffle = True,drop_last=True)
    
    PT_FILE = "Trained\\rnnV3.pth"
    bgru1 = M1.RNN_GRU(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)
    bgru1.load_state_dict(torch.load(PT_FILE))

    PT_FILE = "Trained\\rnnV3.pth"
    bgru2 = M1.RNN_GRU(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)
    bgru2.load_state_dict(torch.load(PT_FILE))

    PT_FILE = "Trained\\rnnV3.pth"
    bgru3 = M1.RNN_GRU(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)
    bgru3.load_state_dict(torch.load(PT_FILE))
    
    models = { bgru1 : (-1, 64, 171) , bgru2 : (-1, 64, 171) , bgru3 : (-1, 64, 171) }

    
    if(TRAIN) :
        loss_fun = nn.CrossEntropyLoss().to(device)
        evaluate(models , data_loader , loss_fun ,  device , EPOCHS)

        
        
        



