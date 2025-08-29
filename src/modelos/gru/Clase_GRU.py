import torch.nn as nn
class modelo_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, linear_size, dropout, bidirectional=True):
        super(modelo_GRU, self).__init__()
        self.gru = nn.GRU(input_size=int(input_size), 
                          hidden_size=int(hidden_size), 
                          num_layers=int(num_layers), 
                          batch_first=True, 
                          bidirectional=bool(bidirectional))

        self.dropout = nn.Dropout(dropout)

        if bidirectional:
            hidden_size *= 2
            
        self.batch_norm = nn.BatchNorm1d(hidden_size)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, linear_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_size, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.batch_norm(out)
        out = self.fc(out)
        return out
    