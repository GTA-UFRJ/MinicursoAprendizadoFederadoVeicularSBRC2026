import torch 

class LSTM(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=1, 
                                  hidden_size=50, 
                                  num_layers=1, 
                                  batch_first=True)

        self.linear = torch.nn.Linear(50, 1)

    def forward(self, x):
        
        x, _ = self.lstm(x)

        return self.linear(x)