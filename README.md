# README: How to Run the Deep Learning Model for Predicting Cancer Mortality Rates

## Overview

This project builds a deep neural network (DNN) model to predict cancer mortality rates based on a dataset. The model includes preprocessing steps like data imputation, feature selection, and synthetic data generation. In addition, it compares the performance of the DNN with a traditional linear regression model. 

The code provides functions for training, saving, and loading models, as well as testing the model's performance on a test dataset. This guide explains how to set up the environment, preprocess the data, train the models, and test them.

## Requirements

Ensure the following libraries are installed before running the code:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

### Libraries used:
- `pandas`: For data manipulation and loading the dataset.
- `numpy`: For numerical operations and synthetic data generation.
- `matplotlib`: For plotting training and validation loss curves.
- `scikit-learn`: For data preprocessing, KNN imputation, feature selection, linear regression, and R² score computation.
- `tensorflow`: For building, training, and saving deep learning models.

## Dataset

The dataset used is assumed to be a CSV file named `cancer_reg.csv`, containing various features related to cancer mortality rates. You need to place this file in the appropriate directory (or change the file path in the code accordingly).

### Dataset File Path

You can customize the dataset path in the `preprocess_data` function:
```python
def preprocess_data(data_path='/Users/danielvennemeyer/Workspace/Deep Learning/cancer_reg.csv'):
```

Make sure to replace the path with the correct location of your dataset.

## Project Structure

### Files
- `dnn.py`: The main Python script containing the code for data preprocessing, model training, and model testing.
- `cancer_reg.csv`: The dataset file (not included in this repository but should be placed locally).

### Functions
- **`preprocess_data()`**: Preprocesses the dataset by imputing missing values, performing feature selection, generating synthetic data, and scaling features.
- **`save_model(model, model_name)`**: Saves the trained model to an HDF5 file.
- **`load_trained_model(model_name)`**: Loads a pre-trained model from an HDF5 file.
- **`train_model(layers, lr)`**: Trains a deep neural network model with customizable layers and learning rate.
- **`test_model(X_test_scaled, y_test, model_name)`**: Tests a pre-trained model on the test dataset and evaluates its performance.

## How to Run the Models

### 1. Preprocessing the Data

To preprocess the dataset, the `preprocess_data()` function performs the following steps:

- Handles missing values using KNN imputation.
- Extracts numerical features from categorical string columns.
- Selects the top 13 most important features using mutual information regression.
- Generates synthetic data by adding noise to the training data.
- Standardizes the feature values using `StandardScaler`.

To preprocess the data, call:

```python
X_train_scaled, y_train_synthetic, X_val_scaled, y_val, X_test_scaled, y_test = preprocess_data()
```

### 2. Training the Model

To train the deep neural network model, use the `train_model` function. You can specify the number of layers and the learning rate:

```python
train_model(layers=[30,16,8], lr=0.001)
```

The function will:
- Train the DNN model with the specified architecture.
- Print the R-squared score of the model on the test data.
- Plot the training vs validation loss over epochs.
- Save the trained model as `dnn_model1.h5`.

### 3. Testing the Model

Once the model is trained and saved, you can load and test the model using `test_model`. Specify the test dataset and the model file path:

```python
y_pred_dnn, r2_dnn = test_model(X_test_scaled, y_test, model_name="dnn_model1.h5")
```

This function will load the saved model, make predictions on the test dataset, and print the R-squared score.

### 4. Running the Script

To run the complete pipeline (preprocessing, training, and testing), follow these steps in the script:

```python
if __name__ == "__main__":
    # Train the model
    train_model(layers=[30,16,8], lr=0.001)
    
    # Test the model
    X_train_scaled, y_train_synthetic, X_val_scaled, y_val, X_test_scaled, y_test = preprocess_data()
    y_pred_dnn, r2_dnn = test_model(X_test_scaled, y_test, model_name="dnn_model1.h5")
```

### 5. Plotting Training and Validation Loss

The training process plots the training and validation loss curves automatically after the model is trained. This helps you visualize how well the model is learning over time.

## Model Performance

The models' performance is evaluated using the R-squared metric. You will get the R-squared score for both the linear regression model and the DNN model:

- **Linear Regression**: The traditional model for comparison.
- **Deep Neural Network (DNN)**: The primary model for predicting cancer mortality rates.

### Example Output

```bash
Linear Regression Test R-squared: 0.65
DNN Test R-squared: 0.72
```

This indicates that the DNN model performs better in predicting cancer mortality rates.

## Customizing the Model

- **Change Layers**: Modify the number of layers and neurons by changing the `layers` argument in the `train_model` function.
  
  Example:
  ```python
  train_model(layers=[30, 16, 8], lr=0.001)
  ```

- **Adjust Learning Rate**: Modify the learning rate by setting the `lr` parameter in the `train_model` function.

  Example:
  ```python
  train_model(layers=[30, 16, 8], lr=0.005)
  ```

## Conclusion

This code demonstrates a machine learning pipeline for predicting cancer mortality rates using a deep neural network model. The DNN model is trained, tested, and evaluated based on R-squared scores, and its performance is compared with linear regression.

