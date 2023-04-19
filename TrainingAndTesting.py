#MODULE1 : TRAINING
#PENDING : SPLIT THE DATA INTO TRAIN ANS SPLIT AND VALIDATE THE PROCESS 

TRAIN = True
name = "rnnV3"

import torch
import torchaudio 
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader , Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv 

import us8k_dataset as ds1
import rnn_model as M1
import MyMetrics as met

trLossList , tstLossList  , trAcc , testAcc = [] , [] , [] , [] 
ACCURACY = [] 
BATCH_SIZE = 32
EPOCHS  = 10 
LR = 0.01
ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\set1.csv"
AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050*4 
NUM_SAMPLES = 22050*4

INP = 171            #INPUTS 
HID = BATCH_SIZE     #NUMBER OF HIDDEN LAYERS 
NL  = 8              #NUMBER OF LAYERS 
NU  = 1              #NUMBER OF UNITS 
NCLS = 10            #NUMBER OF CLASSES
NH = 16


PT_FILE = f"Trained\\{name}.pth"
DATA_FILE = f"performance\\lossValues_{name}.csv"

def test_single_epoch(model , data_loader , loss_fun , device ) :
    model.eval()
    with torch.inference_mode() :
        for inp , tar in data_loader :
            inp , tar = inp.to(device) , tar.to(device)
            logits = model(inp.view(-1, 64, 171))
            loss = loss_fun(logits , tar)
            acc = met.acc(logits , tar)
        tstLossList.append(loss.item())
        testAcc.append(acc)
        print(f"testing accuracy {acc}")

def train_single_epoch(model , data_loader , loss_fun , optimiser , device , scheduler) :
    model.train()
    for inp , tar in data_loader :
        inp , tar = inp.to(device) , tar.to(device)
        #calculate the loss
        #the input shape is (batch_size, sequence_length, input_size_1, input_size_2) = (32, 1, 64, 171).
        #print(inp.shape)
        logits = model(inp.view(-1, 64, 171))
        loss = loss_fun(logits , tar)
        acc = met.acc(logits , tar)

        
        #propagate backwards
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        #scheduler.step(loss)
        #ACCURACY.append(acc) 
    trLossList.append(loss.item())
    trAcc.append(acc)
    print(f"\ntraining accuracy {acc}")

def evaluate(model , training_data_loader , testing_data_loader  , loss_fun , optimiser , device , epochs , scheduler):

    for i in tqdm(range(epochs) , desc = "->"):
        train_single_epoch(model , training_data_loader , loss_fun , optimiser , device ,scheduler)
        test_single_epoch(model , training_data_loader , loss_fun , device)
    print("training finished")


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
    train_size = int(0.8 * len(usd))
    test_size = len(usd) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(usd , [train_size, test_size])

    print(len(train_dataset) , len(test_dataset))
    
    trDataLoader = DataLoader(train_dataset , batch_size = BATCH_SIZE,shuffle = True,drop_last=True)
    tstDataLoader = DataLoader(test_dataset , batch_size = BATCH_SIZE,shuffle = True,drop_last=True)

    print(tstDataLoader)
    
    model = M1.RNN_GRU(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)

    #model = M1.MultiLSTMBidirectionalModel(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)

    model.load_state_dict(torch.load(PT_FILE))
    
    print(model)


    if(TRAIN) :
        loss_fun = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.ASGD(model.parameters() , lr = LR)#ASGD , SGD
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        evaluate(model , trDataLoader , tstDataLoader , loss_fun , optimizer ,  device , EPOCHS , scheduler)

        torch.save(model.state_dict() , PT_FILE)
        print("trained RNN saved as "+PT_FILE)
        
        plt.plot(np.arange(len(trLossList)),np.array(trLossList))
        plt.plot(np.arange(len(tstLossList)),np.array(tstLossList))
        plt.legend({"Training loss :" : 14 , "Testing loss :":14})
        plt.xlabel('steps') 
        plt.ylabel('Loss')
        plt.title(f"Loss graph(class : {name})")
        plt.savefig(f'performance//TrainVsTest_{name}.png')
        plt.show()
        
        plt.plot(np.arange(len(trAcc)),np.array(trAcc))
        plt.plot(np.arange(len(testAcc)),np.array(testAcc))
        plt.legend({"Training acc :" : 14 , "Testing acc :":14})
        plt.xlabel('steps') 
        plt.ylabel('accuracy')
        plt.title(f"Accuracy graph(class : {name})")
        plt.savefig(f'performance//trAcc_Vs_testAcc_{name}.png')
        plt.show()
    
        fieldnames = ['Training values' , 'Testing values' , 'Training acc' , 'Testing acc']
        with open(DATA_FILE ,"a+",newline = "") as f :
            dict_writer = csv.writer(f)
            for a, b, c , d in zip(trLossList , tstLossList  , trAcc , testAcc):
                dict_writer.writerow([str(a) , str(b) , str(c) , str(d)])
