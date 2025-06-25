import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def normalize_l1(data):
    norm = np.sum(np.abs(data), axis=0, keepdims=True)
    return data / norm

def normalize_l2(data):
    norm = np.linalg.norm(data, axis=0, keepdims=True)
    return data / norm

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, activation='sigmoid'):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))
        
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.activation(self.Z2)
        return self.A2

    def backward(self, X, y, y_pred, lr):
        dL_dA2 = y_pred - y
        dL_dZ2 = dL_dA2 * self.activation_derivative(self.Z2)
        
        dL_dW2 = np.dot(dL_dZ2, self.A1.T)
        dL_db2 = np.sum(dL_dZ2, axis=1, keepdims=True)
        
        dL_dA1 = np.dot(self.W2.T, dL_dZ2)
        dL_dZ1 = dL_dA1 * self.activation_derivative(self.Z1)
        
        dL_dW1 = np.dot(dL_dZ1, X.T)
        dL_db1 = np.sum(dL_dZ1, axis=1, keepdims=True)
        
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2

    def train(self, X_train, y_train, epochs, lr):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(X_train.shape[1]):
                X_sample = X_train[:, i:i+1]
                y_sample = y_train[:, i:i+1]
                
                y_pred = self.forward(X_sample)
                self.backward(X_sample, y_sample, y_pred, lr)
                
                loss = 0.5 * np.square(y_pred - y_sample)
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / X_train.shape[1])
        return losses

def generate_data(num_samples=1000):
    X = np.random.uniform(-1, 1, (2, num_samples))
    X[X == 0] = 1e-5 
    
    y = (np.sign(X[0, :]) == np.sign(X[1, :])).astype(int)
    return X, y.reshape(1, -1)

if __name__ == '__main__':
    X_train, y_train = generate_data(500)
    X_test, y_test = generate_data(100)

    configs = {
        'Nieznormalizowane': lambda d: d,
        'Normalizacja L1': normalize_l1,
        'Normalizacja L2': normalize_l2
    }
    
    activations = ['sigmoid', 'relu']
    learning_rate = 0.05
    epochs = 200
    
    plt.figure(figsize=(15, 10))
    plot_index = 1

    print("--- Eksperyment 1: Porównanie normalizacji i aktywacji ---")
    for norm_name, norm_func in configs.items():
        for act_name in activations:
            print(f"Testowanie: {norm_name}, Aktywacja: {act_name}")
            
            X_train_processed = norm_func(X_train)
            model = NeuralNetwork(activation=act_name)
            losses = model.train(X_train_processed, y_train, epochs, learning_rate)
            
            plt.subplot(2, 3, plot_index)
            plt.plot(losses)
            plt.title(f'{norm_name} / {act_name.upper()}')
            plt.xlabel('Epoki')
            plt.ylabel('Koszt (MSE)')
            plt.grid(True)
            plot_index += 1

    plt.tight_layout()
    plt.suptitle(f"Przebieg uczenia (lr={learning_rate})", fontsize=16, y=1.02)
    plt.show()

    print("\n--- Eksperyment 2: Wpływ współczynnika uczenia (dla ReLU i L2) ---")
    learning_rates_to_test = [0.001, 0.01, 0.1, 0.5, 0.9]
    plt.figure(figsize=(12, 7))
    X_train_l2 = normalize_l2(X_train)
    
    for lr in learning_rates_to_test:
        print(f"Testowanie współczynnika uczenia: {lr}")
        model = NeuralNetwork(activation='relu')
        losses = model.train(X_train_l2, y_train, epochs=100, lr=lr)
        plt.plot(losses, label=f'lr = {lr}')

    plt.title('Wpływ współczynnika uczenia na konwergencję (ReLU, L2)')
    plt.xlabel('Epoki')
    plt.ylabel('Koszt (MSE)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.3)
    plt.show()