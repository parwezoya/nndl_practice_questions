import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class RegularizedFFN:
    def __init__(self, input_size, hidden_size, output_size, lambda_l2=0.01, dropout_rate=0.2):
        # Hyperparameters
        self.lambda_l2 = lambda_l2  # L2 Penalty strength
        self.dropout_rate = dropout_rate
        
        # Weights and Biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X, training=True):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # --- DROPOUT IMPLEMENTATION ---
        if training:
            # Create a mask of 0s and 1s
            self.mask = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            self.a1 *= self.mask
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        # Output error
        error_out = y - output
        d_output = error_out * sigmoid_derivative(output)

        # Hidden error
        error_hidden = d_output.dot(self.W2.T)
        # Apply the same dropout mask to the gradient
        d_hidden = error_hidden * sigmoid_derivative(self.a1) * getattr(self, 'mask', 1)

        # --- L2 REGULARIZATION GRADIENT ---
        # The gradient of (lambda/2 * W^2) is (lambda * W)
        # We subtract this from the weight update to "decay" the weights
        l2_grad_W2 = self.lambda_l2 * self.W2
        l2_grad_W1 = self.lambda_l2 * self.W1

        # Update Weights with L2 Decay
        self.W2 += (self.a1.T.dot(d_output) - l2_grad_W2) * lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
        self.W1 += (X.T.dot(d_hidden) - l2_grad_W1) * lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# --- Test the Program ---
# Dataset: Simple XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize Network with L2=0.01 and Dropout=20%
nn = RegularizedFFN(input_size=2, hidden_size=5, output_size=1, lambda_l2=0.01, dropout_rate=0.2)

print("Training Regularized FFN...")
for epoch in range(10000):
    # Training mode = True (enables Dropout)
    output = nn.forward(X, training=True)
    nn.backward(X, y, output, lr=0.2)
    
    if epoch % 2000 == 0:
        loss = np.mean(np.square(y - output))
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("\nFinal Predictions (Training=False):")
# Testing mode = False (disables Dropout)
final_out = nn.forward(X, training=False)
for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {final_out[i][0]:.4f}")