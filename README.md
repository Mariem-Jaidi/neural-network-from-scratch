# Neural Network from Scratch — MNIST Digit Recognizer

A fully functional neural network built using **only NumPy and math** :no TensorFlow, no PyTorch, no shortcuts.  
Trained on the MNIST dataset to classify handwritten digits (0–9), achieving **~82% accuracy** on the validation set.



##  What This Is

This project implements a 2-layer neural network completely from scratch to understand what actually happens inside a machine learning model.  
Every component :forward propagation, activation functions, backpropagation, gradient descent is coded manually using NumPy arrays and linear algebra.

> Built as a learning exercise following [Samson Zhang's](https://www.youtube.com/watch?v=w8yWXqWQYmU) from-scratch neural network tutorial on YouTube.



## Architecture

Input Layer     →   784 neurons  (28×28 pixel image, flattened)
Hidden Layer    →   10 neurons   (ReLU activation)
Output Layer    →   10 neurons   (Softmax activation → digit 0–9)




##  How It Works

### Data Preprocessing
- Dataset: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) (MNIST format)
- Pixel values normalized from `[0, 255]` → `[0, 1]`
- Split into **1,000 validation** samples and **41,000 training** samples

### Forward Propagation

Z1 = W1 · X + b1       # Linear transformation
A1 = ReLU(Z1)          # Non-linear activation
Z2 = W2 · A1 + b2      # Second linear layer
A2 = Softmax(Z2)       # Output probabilities


### Backpropagation
Gradients computed manually using the chain rule:

dZ2 = A2 - Y_one_hot
dW2 = (1/m) · dZ2 · A1ᵀ
dZ1 = W2ᵀ · dZ2 * ReLU'(Z1)
dW1 = (1/m) · dZ1 · Xᵀ


### Gradient Descent
Parameters updated iteratively:

W := W - α · dW
b := b - α · db




##  Results


 Training iterations : 500 
 Learning rate (α) : 0.1 
 Validation accuracy : **~82%** 

Loss and accuracy were printed every 10 iterations during training.



## Concepts Covered

- Weight initialization (random, centered around 0)
- ReLU activation function and its derivative
- Softmax for multi-class probability output
- One-hot encoding of labels
- Forward and backward propagation
- Mini-batch gradient descent
- Accuracy evaluation on unseen data
- Visualization of predictions with Matplotlib



##  Tech Stack


Python 3.12 the Core language 
NumPy for all matrix operations & math 
Pandas for Data loading 
Matplotlib for Visualization 



## How to Run

1. Download the dataset from [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
2. Place `train.csv` in the working directory
3. Run the notebook cell by cell

Or open directly on Kaggle and fork the notebook.



## Key Takeaway

Building a neural network from scratch without any ML framework forces you to understand the actual math behind training. After this, concepts like loss functions, backprop, and gradient descent stop being black boxes.


## reference

- Tutorial by [Samson Zhang on YouTube](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- Dataset: [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer)
