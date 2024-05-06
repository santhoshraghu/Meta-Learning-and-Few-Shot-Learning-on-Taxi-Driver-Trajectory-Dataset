import torch
import torch.nn as nn

class SiameseLSTM(nn.Module):
    """
    Neural network model for training Siamese Networks.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(SiameseLSTM, self).__init__()
        
        # Defining two separate LSTM layers with dropout
        self.lstm1 = nn.LSTM(input_size=input_dim, 
                             hidden_size=hidden_dim, 
                             num_layers=2,  
                             batch_first=True,
                             dropout=dropout) 
        
        self.lstm2 = nn.LSTM(input_size=input_dim, 
                             hidden_size=hidden_dim, 
                             num_layers=2,  
                             batch_first=True,
                             dropout=dropout)  
        
        # Fully connected layer with dropout
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass of the SiameseLSTM model.
        """
        # Separate input trajectory pairs
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]

        # Forward pass through LSTM branches
        out1, _ = self.lstm1(x1)
        out2, _ = self.lstm2(x2)

        # Extract final hidden states
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]

        # Concatenate hidden states
        combined = torch.cat((out1, out2), dim=1)

        # Pass through fully connected layer for prediction
        output = self.fc(combined)

        return output
