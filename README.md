# Deep Neural Network Training Stability Analysis

This project demonstrates how different weight initialization strategies, activation functions, batch normalization, and gradient clipping affect the stability and performance of a deep fully-connected neural network trained on the Fashion-MNIST dataset in CSV format.

The code trains several versions of the same network using different techniques and compares their behavior through:

* Training loss curves
* Gradient norm evolution
* Test accuracy comparison

---

## 1. Project Structure

```
project/
│
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
│
├── main.py                     # Full project code
├── README.md                   # This file
```

---

## 2. Dataset Description

The project uses the Fashion-MNIST dataset in CSV format. Each row of the CSV has the structure:

```
label, pixel1, pixel2, ..., pixel784
```

* **label**: integer from 0 to 9 (class ID)
* **pixels**: 784 grayscale pixel values (0-255), flattened from a 28x28 image

The dataset contains 10 clothing categories:

```
0 — T-shirt/Top
1 — Trouser
2 — Pullover
3 — Dress
4 — Coat
5 — Sandal
6 — Shirt
7 — Sneaker
8 — Bag
9 — Ankle Boot
```

The CSV version is commonly available from Kaggle or public repositories.

---

## 3. How the Code Works

### 3.1 Dataset Loader

The `FashionMNISTDataset` class reads the CSV files and prepares the images and labels by:

* Extracting the label column
* Extracting pixel columns
* Normalizing the pixel values to the 0–1 range
* Converting everything into PyTorch tensors

### 3.2 Deep Neural Network

The model is a fully-connected network with:

* 10 linear hidden layers
* ReLU activation by default
* Optional batch normalization
* Final classification layer outputting 10 logits

### 3.3 Weight Initializations Compared

The project tests:

* Small weight initialization (causes vanishing gradients)
* Large weight initialization (causes exploding gradients)
* Xavier initialization
* He initialization
* Batch normalization
* Gradient clipping

### 3.4 Training Function

The training loop performs:

* Forward pass
* Loss computation
* Backward pass
* Optional gradient clipping
* Accuracy evaluation
* Gradient norm computation for analysis

### 3.5 Graphs

The script produces two graphs:

1. **Loss curve comparison** — shows how different initialization techniques affect learning speed and stability.
2. **Gradient norm comparison** — shows whether gradients explode, vanish, or remain stable across epochs.

---

## 4. Setup Instructions

### 4.1 Install Dependencies

```
pip install torch torchvision pandas matplotlib numpy
```

### 4.2 Place Dataset Files

Ensure the following files are in the project root:

```
fashion-mnist_train.csv
fashion-mnist_test.csv
```

### 4.3 Run the Project

```
python main.py
```

The script automatically:

* Loads the dataset
* Builds all model variants
* Trains each model
* Displays loss and gradient graphs
* Prints accuracy comparison

---

## 5. Output Produced

### 5.1 Console Output

The script prints for each model:

```
Epoch 1/5 | Loss: ... | Test Acc: ...%
...
```

### 5.2 Graphs

Two graphs will appear:

* Loss curves for all methods
* Gradient norm curves for all methods

### 5.3 Accuracy Table

A final accuracy table is printed:

```
Method | Test Accuracy
```

---

## 6. Interpretation Summary

### Small Initialization

Very small weights cause vanishing gradients, slow training, and low accuracy.

### Large Initialization

Very large weights cause exploding gradients, making training unstable.

### Xavier Initialization

Provides balanced gradients and stable learning.

### He Initialization

Best suited for ReLU networks; produces fast and stable learning.

### Gradient Clipping

Useful when initial weights are large; prevents gradient explosion.

### Batch Normalization

Produces the most stable gradients and smooth learning curves, even in deep networks.

---

## 7. Notes

* This project is designed for educational and experimental purposes.
* Any activation function, optimizer, or depth can be modified to explore further stability behaviors.

---

## 8. Author

This README explains the project in clear, human-readable language without emojis.
