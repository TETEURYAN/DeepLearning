# Activation Functions

Activation functions are crucial components of neural networks as they introduce non-linearities, enabling the network to learn complex patterns and relationships in the data. Each layer of a neural network typically applies an activation function to the weighted sum of its inputs.

## Types of Activation Functions:

### 1. Sigmoid:
   - Maps the input to a range between 0 and 1.
   - Often used in the output layer of binary classification problems.
``` python
   def sigmoid(x):
       return 1 / (1 + np.exp(-x))

   def sigmoid_derivative(x):
       return x * (1 - x)

```

### 2. Tanh (Hyperbolic Tangent):
   - Similar to the sigmoid function but maps the input to a range between -1 and 1.
   - Can be more effective in certain cases due to its symmetric nature.

``` python
   def tanh(x):
       return np.tanh(x)
```

### 3. ReLU (Rectified Linear Unit):
   - Sets negative inputs to zero and leaves positive inputs unchanged.
   - Widely used due to its simplicity and effectiveness in training deep neural networks.

``` python
   def relu(x):
      return np.maximum(0, x)
```

### 4. Leaky ReLU:
   - Similar to ReLU but allows a small, non-zero gradient for negative inputs.
   - Addresses the "dying ReLU" problem where neurons could become inactive during training.
``` python
   def leaky_relu(x, alpha=0.2):
      return np.maximum(alpha * x, x)
```

### 5. ELU (Exponential Linear Unit):
   - Similar to ReLU for positive inputs but allows negative inputs to have a non-zero output.
   - Can alleviate the vanishing gradient problem and produce smoother gradients.

``` python
   def elu(x):
      return np.where(x >= 0, x, np.exp(x) - 1)
```

### 6. Softmax:
   - Maps the inputs to a probability distribution over multiple classes.
   - Commonly used in the output layer of multi-class classification problems.

``` python

   def softmax(x):
      e_x = np.exp(x)
      return e_x / np.sum(e_x, axis=1, keepdims=True)

```

## Benefits of Activation Functions:
- Introduce non-linearities, allowing neural networks to approximate complex functions.
- Enable the network to learn and represent intricate patterns in the data.
- Impact the model's training dynamics, affecting convergence speed and performance.

## Conclusion:
Activation functions play a crucial role in the expressive power and training of neural networks. Choosing the appropriate activation function depends on the nature of the problem, the network architecture, and empirical performance on validation data.
