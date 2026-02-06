import numpy as np

class ForestCoverNet:
    """
    Neural Network for Forest Cover Type Classification (NumPy implementation)
    
    REQUIRED SIGNATURE:
    - __init__(self, input_dim, num_classes=7)
    - forward(self, x) -> returns logits of shape (batch_size, num_classes)
    """
    
    def __init__(self, input_dim, num_classes=7):
        self.input_dim = input_dim
        self.hidden_dim = [128, 64, 32, 16]
        self.num_classes = num_classes
        self.L = len(self.hidden_dim)
        
        # Initializing weights and biases for hidden layers
        self.W = {}
        self.B = {}

        # Weight initialization
        for l in range(self.L):
            in_dim = self.input_dim if l == 0 else self.hidden_dim[l-1]
            out_dim = self.hidden_dim[l]
            scale = np.sqrt(2/in_dim)
            self.W[l] = np.random.randn(in_dim, out_dim) * scale
            self.B[l] = np.zeros((1, out_dim))

        # Initializing weights and biases for output layer
        scale = np.sqrt(2 / self.hidden_dim[-1])
        self.W[self.L] = np.random.randn(self.hidden_dim[-1], num_classes) * scale
        self.B[self.L] = np.zeros((1, num_classes))

        # Activation and pre activation for later use in backprop
        self.A = {}
        self.Z = {}

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, input_dim)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass using NumPy
        
        # Starts
        self.A[0] = x

        # For hidden layers, loop 1 to L-1
        for l in range(0, self.L):
            self.Z[l] = np.dot(self.A[l], self.W[l]) + self.B[l]
            self.A[l+1] = self.relu(self.Z[l])
        
        # Logit   
        logits = np.dot(self.A[self.L], self.W[self.L]) + self.B[self.L]
        return logits

    def __call__(self, x):
        return self.forward(x)
    
    def load_state_dict(self, state_dict):
        self.W = state_dict["W"]
        self.B = state_dict["B"]
    