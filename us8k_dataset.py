#MODULE 1 : PROGRAM 1 DATASET (URBAN SOUND 8K)

import torch
import torchaudio
import torchvision
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset 
                    #THIS IS A BASE CLASS DATASETS WE NEED FOR OUR CUSTOM DATASETS
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time as t
import matplotlib.pyplot as plt 
#creating a custom datasets

class UrbanSoundDataset(Dataset) :
    def __init__(self , annotation_file , audio_dir , transformation , target_sample_rate , num_samples , device = "cpu") :
        self.annotation = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device 
        self.transformation = transformation.to(self.device)#apply this mel-spectrogram to audio file that we are loading -> modifying the getitem method
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        
    def __len__(self) :
        return len(self.annotation)

    def __getitem__(self,index):
                                                        # lst[idx] == lst.__getitem__(idx)
                                                        #what do we really want here
                                                        #getting and loading the waveform of the audio sample associated to the certain index
                                                        #at the same time .., return the label associated with it
                                                        #getting the path

        audio_sample_path = self.get_audio_sample_path(index)  #this is a private method
        label = self.get_audio_sample_label(index)             #loading audio files

                                                        #use the load functionallity of the torchaudio
                                                        #this function returns signal -> waveform or time series and sample rate

        signal , sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #this signal is Pytorch tensor(num_channels , samples) -> (2 , 16000) ->(1 , 16000)
        signal = self.resampleIfNecessary(signal, sr)
                                                        #We are unifying the process because the dataset contains mono audio samples they may be stero samples so they have two channels or they may have multiple channels
                                                        #we are not intrested in more than one channel
                                                        #take the initial signal which is loaded --> then mix it down to mono
        
        signal = self.mixDownIfNecessary(signal)
        #print(f"Before {signal.shape}")
        signal = self.cutIfNecessary(signal)
        signal = self.rightPadIfNecessary(signal)
        #print(f"After {signal.shape}")
                                                        #converting the signal to MFCC spectrogram 
        signal = self.transformation(signal)
        #signal = T.FrequencyMasking(freq_mask_param=80)

                                                        #Changing the signal to fixed length before making transformations
                                                        #we have different durations in length the problem is most deep learning architectures have data fixed in its shape
                                                        #We need to ensure that the signal we load before we produce and process and extract the mal-spectorgram is consistent
                                                        #The number of samples we need to have in a process 
        
        return signal , label

    def cutIfNecessary(self , signal) :
        if signal.shape[1] > self.num_samples :
            signal = signal[:,:self.num_samples]
        return signal

    def rightPadIfNecessary(self , signal) :
        length_signal = signal.shape[1]
        if length_signal < self.num_samples :
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0 , num_missing_samples)
            signal = F.pad(signal , last_dim_padding)
        return signal 
            
        
    def resampleIfNecessary(self, signal, sr) :
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr , self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def mixDownIfNecessary(self,signal):
        if signal.shape[0] > 1 :                                                # if not mono signal then mix it down 
            signal = torch.mean(signal , dim = 0 , keepdim = True)
        return signal 
        


    def get_audio_sample_path(self,index: int) :
        
        fold = f"fold{self.annotation.iloc[index ,5]}"                          #identify the fold
        path = os.path.join(self.audio_dir , fold , self.annotation.iloc[index , 0])
        return path

    def get_audio_sample_label(self, index : int) :
        return self.annotation.iloc[index , 6]
    
    

if __name__ == "__main__" :
    ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
    AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
    SAMPLE_RATE = 22050 #SR/NS = 1 SEC  WE ARE GETTING 1 SEC WORTH OF AUDIO 
    NUM_SAMPLES = 22050

    #setting up the device agnoistic code
    device = "cpu" if torch.cuda.is_available() else "cpu"
    print(f"The device :{device}")

                                                    #"""This is not enough for what we are going to do here because we want to build a classifier
                                                    #that we can discriminate among the different sounds in the us ds , but for that were not gonna use waveforms to train our model
                                                    #but rather were going to be using mel-spectorgrams"""

                                                    #We are using torchaudio TRANSFORMS ..., we are focusing on how to extract mel-spectrograms and extract them directly
                                                    #directly with in custom audio data set
    
    mel_spectrogram = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024  ,                             #frame length
        hop_length = 512 ,                          #half of the frame length
        n_mels = 64 
        )                                           #this is a callable object 
    
    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE ,
        AUDIO_DIR ,
        mel_spectrogram ,
        SAMPLE_RATE ,
        NUM_SAMPLES ,
        device
        )

    print(f"There are {len(usd)} samples in the datasets.")
    signal , label = usd[0]
    print(signal , label)
