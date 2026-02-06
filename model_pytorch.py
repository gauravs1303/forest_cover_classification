import torch.nn as nn

class ForestCoverNet(nn.Module):
    """
    Neural Network for Forest Cover Type Classification
    
    REQUIRED SIGNATURE:
    - __init__(self, input_dim, num_classes=7)
    - forward(self, x) -> returns logits of shape (batch_size, num_classes)
    
    Implement your architecture below.
    """
    
    def __init__(self, input_dim, num_classes=7):
        super(ForestCoverNet, self).__init__()
        
        # TODO: Implement your model architecture
        #Start
        # Same Hidden layer as in numpy model [128, 64], and same activation function "relu"
        self.hidden_dim = [128, 64, 32, 16]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes) # Output Logits
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # start
        return self.net(x)