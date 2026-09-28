import numpy as np

# 1. Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x here is already the sigmoid output: s * (1 - s)
    return x * (1 - x)

# 2. The Neural Network Class
class SigmoidNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        # Initialize biases to zero
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Input -> Hidden Layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Hidden -> Output Layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        # Calculate error at output layer
        error_out = y - output
        d_output = error_out * sigmoid_derivative(output)

        # Calculate error at hidden layer
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)

        # Update weights and biases (Gradient Descent)
        self.W2 += self.a1.T.dot(d_output) * learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(d_hidden) * learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# 3. Training Data (XOR Problem)
# Input: [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 4. Execution
nn = SigmoidNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
epochs = 10000
lr = 0.5

print("Training...")
for i in range(epochs):
    output = nn.forward(X)
    nn.backward(X, y, output, lr)
    if i % 2000 == 0:
        loss = np.mean(np.square(y - output)) # Mean Squared Error
        print(f"Epoch {i} - Loss: {loss:.4f}")

print("\nFinal Results:")
final_out = nn.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} | Target: {y[i]} | Predicted: {final_out[i][0]:.4f}")