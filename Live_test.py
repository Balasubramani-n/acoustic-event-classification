#MODULE1 : LIVE TEST

import torch
from torch import nn
import torchaudio 
import torchaudio.transforms as T
import torch.nn.functional as F
import pyaudio as pa 
import wave
import us8k_dataset as d1 
import rnn_model as m1 
import MyMetrics as met

LOSS_VALUES = []
ACCURACY = [] 
BATCH_SIZE = 32
EPOCHS  = 10
LR = 0.001
ANNOTATIONS_FILE = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\set8.csv"
AUDIO_DIR = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
SAMPLE_RATE = 22050*4 
NUM_SAMPLES = 22050*4

INP = 171            #INPUTS 
HID = BATCH_SIZE     #NUMBER OF HIDDEN LAYERS 
NL  = 8              #NUMBER OF LAYERS 
NU  = 1              #NUMBER OF UNITS 
NCLS = 10            #NUMBER OF CLASSES
NH = 16


def predict(model , inp):
    cm = "air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music".split(',')
    model.eval()
    with torch.inference_mode() :
        p = model(inp)
        pi = p[0].argmax(0)
        p = cm[pi] #predicted 
    return p 

if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device using :{device}")
    
    #instantiating our data set objects
    
                                                        
    transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64, "center": False},
        )

    
    usd = d1.UrbanSoundDataset(
        ANNOTATIONS_FILE ,
        AUDIO_DIR ,
        transform ,
        SAMPLE_RATE ,
        NUM_SAMPLES ,
        device
        )


    model = m1.RNN_GRU(input_size = INP, hidden_size = HID, num_layers = NL , output_size = NCLS).to(device)
    model.load_state_dict(torch.load("C:\\Users\\nagav\\OneDrive\\Desktop\\MLModuled\\Trained\\rnnV3.pth"))

    FRAMES_PER_BUFFER = 22050*4
    FORMAT = pa.paFloat32
    FORMAT = pa.paInt16
    CHANNELS = 1
    RATE = 22050*4
    p = pa.PyAudio()
    stream = p.open(
            format = FORMAT,
            channels = CHANNELS ,
            rate = RATE,
            input = True,
            frames_per_buffer = FRAMES_PER_BUFFER
        ) 

    print("start recording...,")
    seconds = 4
    frames = []
    tot_sec = int(input("Enter samples :"))
    try :
        for j in range(tot_sec) : #inf loop
            frames = []
            for i in range(0 , int(RATE/FRAMES_PER_BUFFER*seconds)):
                data = stream.read(FRAMES_PER_BUFFER)
                frames.append(data)
            #thread
            # mode
            vals = b"".join(frames)
            _path = f"C:\\Users\\nagav\\OneDrive\\Desktop\\MLModuled\\temp\\out{j}.wav"
            obj = wave.open(_path,"wb")
            obj.setnchannels(CHANNELS)
            obj.setsampwidth(p.get_sample_size(FORMAT))
            obj.setframerate(RATE)
            obj.writeframes(vals)
            obj.close()

            #signal = np.array(np.frombuffer(vals, dtype = np.float32))
            #signal = torch.from_numpy(signal)[None,:]
            #sr = torch.tensor([RATE]).to(device)
            #signal = torch.nan_to_num(signal, nan=0.5) 
            signal , sr = torchaudio.load(_path)
            signal = signal.to(device)
            sr = torch.tensor([sr]).to(device)
            
            print(j , signal.shape , signal)
            signal = usd.resampleIfNecessary(signal, sr)
            signal = usd.mixDownIfNecessary(signal)
            signal = usd.rightPadIfNecessary(signal)
            signal = usd.cutIfNecessary(signal)
            signal = usd.transformation(signal)
            inp = signal 
            #inp , tar = usd[index][0] , usd[index][1] #[batch_size , num_channels , fr , time]
            #C:\Users\nagav\OneDrive\Desktop\ML\AF
        
            p_  = predict(model,inp)
            print(p_)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
    else :
        stream.stop_stream()
        stream.close()
        p.terminate()

    print("recording stopped")



