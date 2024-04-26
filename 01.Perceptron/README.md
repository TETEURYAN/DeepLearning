# Perceptron

Perceptron is one of the simplest forms of artificial neural networks, initially proposed by Frank Rosenblatt in 1957. It consists of a single layer of input neurons connected directly to an output neuron, without any hidden layers.

## Key Components:

### 1. Input Layer:
   - Represents the input features of the data.
   - Each input neuron corresponds to a feature.

### 2. Weights and Bias:
   - Each input neuron is associated with a weight, representing the importance of that feature.
   - Bias is an additional input to the neuron, which allows the model to make decisions even when all input features are zero.

### 3. Activation Function:
   - Typically employs a step function, where the output is binary based on a threshold.
   - The output is computed as the weighted sum of inputs plus the bias, passed through the activation function.

### 4. Example using class of Perceptron:

``` python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # Add one for the bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Add bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)  # Update bias

```


## Training:

### 1. Perceptron Learning Rule:
   - Iteratively adjusts the weights and bias based on the error between the predicted output and the true output.
   - If the prediction is correct, no adjustments are made. Otherwise, the weights and bias are updated to reduce the error.

### 2. Convergence:
   - Perceptron learning converges if the data is linearly separable, meaning there exists a hyperplane that can separate the classes.
   - However, it may not converge if the data is not linearly separable.

## Limitations:

### 1. Linear Separability:
   - Perceptrons can only learn linear decision boundaries.
   - Inability to handle non-linearly separable data limits their applicability to certain problems.

### 2. Single Layer:
   - Lack of hidden layers limits the model's ability to learn complex patterns and relationships in the data.

## Conclusion:
Perceptron neural networks represent a fundamental building block in the history of artificial neural networks. While simple and intuitive, perceptrons have limitations in their capability to handle complex tasks. Nevertheless, they laid the foundation for more sophisticated neural network architectures that followed.
