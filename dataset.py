#DATASET and DATA VISUALIZATION 
import coreConfig as cc
exec(cc.stmts)
import librosa
import librosa.display
import warnings

class UrbanSoundDataset(Dataset) :    
    mfcc_t = cc.mfccTransform
    lfcc_t = cc.lfccTransform
    mel_t = cc.melTransform
    amp_to_DB = T.AmplitudeToDB().to(device)
    
    def __init__(self, spec = None , train = True, test_fold = [1]  , kfold = 10) :
        self.annotation = pd.read_csv(cc.annot_file)
        self.audio_dir = cc.audio_dir
        self.device = device 
        self.target_sample_rate = cc.sample_rate
        self.num_samples = cc.num_samples
        self.train = train
        self.spec = list(spec) if type(spec) == type(set()) else spec 

        self.subSet = [i for i in range(len(self.annotation)) if int(self.annotation.iloc[i ,5]) not in test_fold] if train else [i for i in range(len(self.annotation)) if int(self.annotation.iloc[i ,5]) in test_fold]
        
    def __len__(self) :
        return len(self.subSet) 

    def __getitem__(self,index):

        index = self.subSet[index]                                          
    
        audio_sample_path = self.get_audio_sample_path(index)  
        label = self.get_audio_sample_label(index)             
        signal , sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resampleIfNecessary(signal, sr)
        signal = self.mixDownIfNecessary(signal)
        signal = self.cutIfNecessary(signal)
        signal = self.rightPadIfNecessary(signal)

        d = dict()
        if ("mfcc" in self.spec) or (self.spec is None) :
            mfcc = self.mfcc_t(signal)
            d.update({"mfcc" : mfcc})
            
        if ("mfccDB" in self.spec) or (self.spec is None) :
            mfccDB = self.amp_to_DB(self.mfcc_t(signal))
            d.update({"mfccDB" : mfccDB})

        if ("lfcc" in self.spec) or (self.spec is None):
            lfcc = self.lfcc_t(signal)
            d.update({"lfcc" : lfcc})

        if ("lfccDB" in self.spec) or (self.spec is None) :
            lfccDB = self.amp_to_DB(self.lfcc_t(signal))
            d.update({"lfccDB" : lfccDB})

        if ("mel" in self.spec) or (self.spec is None) :
            mel = self.mel_t(signal)
            d.update({"mel" : mel})
        
        if ("melDB" in self.spec) or (self.spec is None) :
            melDB = self.amp_to_DB(self.mel_t(signal))
            d.update({"melDB" : melDB})

        if d is dict() :
            raise ValueError("Not a valid spec value(s)")

        if type(self.spec) == type(str()):
            return d.get(self.spec,d) , label
        else :
            return d , label 

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
        return torch.mean(signal , dim = 0 , keepdim = True) if signal.shape[0] > 1 else signal
        


    def get_audio_sample_path(self,index: int) :
        
        fold = f"fold{self.annotation.iloc[index ,5]}"                      
        path = os.path.join(self.audio_dir , fold , self.annotation.iloc[index , 0])
        return path

    def get_audio_sample_label(self, index : int) :
        return self.annotation.iloc[index , 6]


    def data_visualization(self , index) :
        initial_divice = self.device
        self.device = "cpu"
        
        index = self.subSet[index]                                          
    
        audio_sample_path = self.get_audio_sample_path(index)  
        label = self.get_audio_sample_label(index)             
        waveform, sample_rate = torchaudio.load(audio_sample_path)
        waveform = self.resampleIfNecessary(waveform, sample_rate)
        waveform = self.mixDownIfNecessary(waveform)
        waveform = self.cutIfNecessary(waveform)
        waveform = self.rightPadIfNecessary(waveform)


        try :
            labelName = self.get_class_names()[label]
            label = f"(label :{labelName} id :{index})"
            
            # Plot waveform
            plt.figure(figsize=(8, 2))
            plt.title('Waveform '+label)
            plt.plot(waveform.t().numpy())
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_Waveform_{labelName}.png")
            # Compute MEL spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y  = waveform.numpy()[0],
                sr = cc.sample_rate,
                n_fft = cc.n_fft ,
                hop_length = cc.hop_length,
                n_mels = cc.n_mels
            )

            # Plot MEL spectrogram
            plt.figure(figsize=(8, 2))
            plt.title('MEL Spectrogram '+label)
            librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', x_axis='time', sr=cc.sample_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_MEL Spectrogram_{labelName}.png")

            # Plot MEL spectrogram with dB scale
            plt.figure(figsize=(8, 2))
            plt.title('MEL Spectrogram with dB scale '+label)
            librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', sr=cc.sample_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_MEL Spectrogram with dB scale_{labelName}.png")
    
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(
            y=waveform.numpy()[0],
            sr=cc.sample_rate,
            n_mfcc=cc.n_mfcc,
            n_fft=cc.n_fft,
            hop_length=cc.hop_length
            )

            # Plot MFCCs
            plt.figure(figsize=(8, 2))
            plt.title('MFCCs '+label)
            librosa.display.specshow(mfccs, x_axis='time', sr=cc.sample_rate)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_MFCCs_{labelName}.png")

            # Plot MFCCs with dB scale
            plt.figure(figsize=(8, 2))
            plt.title('MFCCs with dB scale '+label)
            librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), x_axis='time', sr=cc.sample_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_MFCCs with dB scale_{labelName}.png")
            # Compute LFCCs
            lfccs = librosa.feature.mfcc(
                y=waveform.numpy()[0],
                sr=cc.sample_rate,
                n_mfcc=cc.n_mfcc,
                n_fft=cc.n_fft,
                hop_length=cc.hop_length,
                dct_type=2,
                norm='ortho'
            )

            # Plot LFCCs
            plt.figure(figsize=(8, 2))
            plt.title('LFCCs '+label)
            librosa.display.specshow(lfccs, x_axis='time', sr=cc.sample_rate)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_LFCCs_{labelName}.png")

            # Plot LFCCs with dB scale
            plt.figure(figsize=(8, 2))
            plt.title('LFCCs with dB scale '+label)
            librosa.display.specshow(librosa.power_to_db(lfccs, ref=np.max), x_axis='time', sr=cc.sample_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f"DP//id_{index}_LFCCs with dB scale_{labelName}.png")

            # Show all plots
            plt.show()
        
            
        except Warning as e:
            pass 
        self.device = initial_divice
        
    def get_class_names(self):
        return "air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music".split(", ")
        


if __name__ == "__main__":

    ds = UrbanSoundDataset(
                    spec = {"mfcc" , "mfccDB" } ,
                    train = True ,
                    test_fold = [0]
                    )

   
    ds.data_visualization(4)
