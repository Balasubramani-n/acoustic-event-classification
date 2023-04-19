#MODULE 1 : PROGRAM 2 MODEL : (CLASSICAL RNN BIDIRECTIONAL)
import torch
from torch import nn


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True , bidirectional = True)
        self.fc = nn.Sequential(
                nn.Linear(hidden_size*2,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.gru(x,h0)
        out = self.fc(out[:, -1, :])        
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        num_classes = output_size 
        super(RNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Define the LSTM layer with bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Define the output linear layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional LSTM

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional LSTM

        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Forward pass through the output linear layer
        out = self.fc(out)

        return out


import torch
import torch.nn as nn

class MultiLSTMBidirectionalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers , output_size ):

        num_units = 3
        num_layers = int(num_layers/3)+1
        num_classes = output_size
        
        super(MultiLSTMBidirectionalModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.lstms = nn.ModuleList()
        for i in range(num_units):
            if i == 0:
                lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            else:
                lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers , batch_first=True, bidirectional=True)
            self.lstms.append(lstm)
        self.fc = nn.Sequential(
                nn.Linear(hidden_size*2,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # *2 for bidirectional
        outs = []
        hiddens = []
        cells = []
        for i in range(self.num_units):
            lstm = self.lstms[i]
            if i == 0:
                out, (hn, cn) = lstm(x, (h0, c0)) # Store both hidden state and cell state
            else:
                out, (hn, cn) = lstm(out, (h0, c0)) # Store both hidden state and cell state
            outs.append(out)
            hiddens.append(hn)
            cells.append(cn)
        out = self.fc(outs[-1][:, -1, :])
        #return out, hiddens, cells # Return both hidden state and cell state
        return out 




if __name__ == '__main__' :
    model = RNN_GRU(input_size = 10 , hidden_size = 32, num_layers= 8,  output_size = 10)
    print(model.state_dict())
    print(model)

    
    
