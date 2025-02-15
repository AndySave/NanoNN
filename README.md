# NanoNN
**NanoNN** is a minimalist neural network framework designed for building **simple, lightweight models**. Its feature set includes:

- **Layers**  
  - `Dense`  
  - `Dropout`

- **Activation Functions**  
  - `ReLU`  
  - `Sigmoid`  
  - `Softmax`

- **Loss Functions**  
  - `MeanSquaredError`  
  - `BinaryCrossEntropy`  
  - `CategoricalCrossEntropy`

- **Optimizers**  
  - `GradientDescent`  
  - `Adam`

- **Model Persistence**  
  - Easily save and load trained models

## Installation
To install NanoNN, simply run:
```bash
pip install git+https://github.com/AndySave/NanoNN.git
```

Once installed you should be able to import the framework:
```python
import nano_nn
````

## Usage

### 1. Defining the model
Define your model by subclassing nn.Module and adding layers using the add() method as shown below:

```python
import nano_nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.add(nn.Dense(196, 256))  # Dense layer: 196 -> 256
        self.add(nn.ReLU())           # Activation

        self.add(nn.Dropout(0.5))     # Dropout with 50% rate

        self.add(nn.Dense(256, 128))  # Dense layer: 256 -> 128
        self.add(nn.ReLU())           # Activation

        self.add(nn.Dense(128, 64))   # Dense layer: 128 -> 64
        self.add(nn.ReLU())

        self.add(nn.Dropout(0.5))     # Dropout with 50% rate

        self.add(nn.Dense(64, 1))     # Dense layer: 64 -> 1
        self.add(nn.Sigmoid())        # Output activation
```

### 2. Configuring the model
Set the loss function, learning rate, and optimizer as follows:

```python
model = Model()
model.set_loss_fn(nn.MeanSquaredError())
model.set_learning_rate(0.001)
model.set_optimizer(nn.Adam())
```

### 3. Preparing data
- **Input Format:** 
Each training instance should be a vector of length equal to the number of inputs for your first layer.
To train, pass a matrix (2D NumPy array) where each row is one instance, and the number of rows is the number of instances. 
If you only have a single instance you must still pass it as a matrix. You can reshape it to 2D (e.g., `x = x.reshape(1, -1)`)
- **Output Format:**
The target must be a vector of length corresponding to the number of inputs.
If using `CategoricalCrossEntropy` for multi class prediction, be sure to one-hot encode your target labels.

### 4. Training the model
A simple training loop would look like this:

```python
epochs = 100
for i in range(epochs):
    # Training phase: enable dropout, etc.
    model.train()
    output, loss = model.forward(X_train, y_train)
    model.backward()

    # Evaluation phase: disable dropout layers
    model.eval()
    output_test, loss_test = model.forward(X_test, y_test)
    
    print(f'epoch: {i+1}/{epochs}, loss: {loss:.6f}, test loss: {loss_test:.6f}')
```

- ```model.train()```: Makes sure layers like dropout are active during training.
- ```model.eval()```: Disables dropout during evaluation.

### 5. Running Inference
Inference works the same as how we did evaluation above. It's important to ensure that the input is a 2D NumPy array, 
even for a single data point. Example with only 1 instance:

```python
x = np.array([5, 8, 1, 2, 6])  # Some random input vector
x = x.reshape(1, -1)           # Reshape vector into matrix
output, _ = model.forward(x)   # Perform inference
print("Predicted output:", output)
```

### 6. Saving and Loading
Once your model is trained, you can save it to a file and load it later for inference or continued training.

```python
model.save_model(file_path="/path/to/save/model_file")
```

When loading a model, ensure that you have defined the model architecture exactly as it was when saved. For example: 
```python
class Model(nn.Module):  # This definition must match the saved model.
    def __init__(self):
        super().__init__()
        self.add(nn.Dense(10, 32))  
        self.add(nn.ReLU())           
        self.add(nn.Dense(32, 1))     
        self.add(nn.Sigmoid())        

model = nn.load_model(file_path="/path/to/load/model_file")
```
