# Optimizers 

Optimizers are algorithms used to update the parameters (weights and biases) of neural networks during the training process. Their primary goal is to minimize the loss function by adjusting the model's parameters iteratively. Different optimizers employ various techniques to achieve this goal efficiently.

## Types of Optimizers:

### 1. Gradient Descent:
   - The most basic optimization algorithm.
   - Updates the parameters in the direction opposite to the gradient of the loss function with respect to the parameters.
   - Variants include:
     - Stochastic Gradient Descent (SGD)
     - Mini-batch Gradient Descent
     - Batch Gradient Descent

``` python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
´´´

### 2. Momentum:
   - Accelerates gradient descent by adding a fraction of the previous update to the current update.
   - Helps to overcome oscillations and navigate through plateaus more efficiently.

### 3. RMSprop (Root Mean Square Propagation):
   - Adapts the learning rates for each parameter based on the average of the recent gradients for that parameter.
   - Divides the learning rate by the exponentially decaying average of squared gradients.

### 4. Adam (Adaptive Moment Estimation):
   - Combines the ideas of momentum and RMSprop.
   - Maintains both a decaying average of past gradients (like momentum) and past squared gradients (like RMSprop).
   - Adjusts learning rates for each parameter individually.

### 5. AdaGrad (Adaptive Gradient Algorithm):
   - Adjusts the learning rates of each parameter based on the historical gradients.
   - Accumulates the squared gradients and uses the square root of the sum to normalize the learning rate.

## Benefits of Optimizers:
- Speed up convergence by efficiently navigating the parameter space.
- Help prevent getting stuck in local minima by escaping saddle points.
- Allow for adaptive learning rates, leading to better convergence on complex surfaces.

## Conclusion:
Choosing the right optimizer is crucial for training neural networks effectively. Different optimizers have different strengths and weaknesses, and their performance may vary depending on the dataset and the architecture of the neural network.
