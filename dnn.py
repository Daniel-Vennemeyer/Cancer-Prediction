import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For plotting the training vs validation error
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model




# Define the function to save and load the model
def save_model(model, model_name="dnn_model1.h5"):
    model.save(model_name)
    print(f"Model saved as {model_name}")

def load_trained_model(model_name="dnn_model1.h5"):
    model = load_model(model_name)
    print(f"Model {model_name} loaded.")
    return model

# Test model function: Load the trained model and test it on the test dataset
def test_model(X_test_scaled, y_test, model_name="dnn_model1.h5"):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train_synthetic)
    y_pred_lin = lin_reg.predict(X_test_scaled)
    r2_lin = r2_score(y_test, y_pred_lin)
    
    # Load the trained model
    model = load_trained_model(model_name)
    
    # Predict on the test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate R-squared on the test data
    r2 = r2_score(y_test, y_pred)
    print(f"DNN Test R-squared: {r2}")
    print(f"Linear Regression Test R-squared: {r2_lin}")
    
    return y_pred, r2

def preprocess_data(data_path='/Users/danielvennemeyer/Workspace/Deep Learning/cancer_reg.csv'):
    # Load the dataset
    df = pd.read_csv(data_path, encoding='ISO-8859-1')

    # Step 1: Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Encode categorical variables (assuming they have been processed in a similar manner)
    def extract_mean_from_string(interval_str):
        interval = interval_str.strip('[]()').split(', ')
        return (float(interval[0]) + float(interval[1])) / 2

    df['binnedInc'] = df['binnedInc'].apply(extract_mean_from_string)

    # Combine numeric and categorical columns back for imputation
    df_encoded = df.drop(columns=['Geography']).reset_index(drop=True)

    # Step 2: KNN Imputation for missing values
    imputer = KNNImputer(n_neighbors=5)  # You can adjust n_neighbors based on your dataset
    df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

    # Split the dataset into features (X) and label (y)
    X = df_imputed.drop(columns=['TARGET_deathRate'])
    y = df_imputed['TARGET_deathRate']

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Feature Selection using mutual information regression
    mutual_info = mutual_info_regression(X_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    drop_cols = mutual_info.sort_values(ascending=False)[13:].keys().tolist()
    # print(mutual_info.sort_values(ascending=False))

    # Drop less important features based on mutual information
    X_train = X_train.drop(columns=drop_cols).reset_index(drop=True)
    X_val = X_val.drop(columns=drop_cols).reset_index(drop=True)
    X_test = X_test.drop(columns=drop_cols).reset_index(drop=True)

    # Step 3: Synthetic Data Generation (Adding Noise to Create Synthetic Samples)
    def generate_synthetic_data(X_train, y_train, num_samples=500):
        X_synthetic = X_train + np.random.normal(0, 0.01, X_train.shape)
        y_synthetic = y_train + np.random.normal(0, 0.01, y_train.shape)
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_train, X_synthetic])
        y_combined = np.hstack([y_train, y_synthetic])
        
        return X_combined, y_combined

    # Generate synthetic data
    X_train_synthetic, y_train_synthetic = generate_synthetic_data(X_train.to_numpy(), y_train.to_numpy(), num_samples=500)

    # Combine synthetic and original data
    X_train_synthetic = np.vstack([X_train.to_numpy(), X_train_synthetic])
    y_train_synthetic = np.hstack([y_train.to_numpy(), y_train_synthetic])


    # Step 4: Standardize the feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_synthetic)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train_synthetic, X_val_scaled, y_val, X_test_scaled, y_test


def train_model(layers=[30,16,8], lr=.001, data_path='/Users/danielvennemeyer/Workspace/Deep Learning/cancer_reg.csv'):
    X_train_scaled, y_train_synthetic, X_val_scaled, y_val, X_test_scaled, y_test = preprocess_data(data_path=data_path)
    # Linear Regression Model (just for reference)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train_synthetic)
    y_pred_lin = lin_reg.predict(X_test_scaled)
    r2_lin = r2_score(y_test, y_pred_lin)
    print(f"Linear Regression Test R-squared: {r2_lin}")

    # Deep Neural Network (DNN) with Custom Learning Rate
    optimizer = Adam(learning_rate=lr)

    # Define the DNN model with layers 30-16-8
    model_dnn = Sequential()
    model_dnn.add(Dense(layers[0], activation='relu', input_shape=(X_train_scaled.shape[1],)))
    for layer in layers[1:]:
        model_dnn.add(Dense(layer, activation='relu'))
    model_dnn.add(Dense(1))  # Output layer for regression

    # Compile the model with Adam optimizer and MSE loss
    model_dnn.compile(optimizer=optimizer, loss='mse')

    # Train the DNN model
    history_dnn = model_dnn.fit(X_train_scaled, y_train_synthetic, epochs=4000, batch_size=64, validation_data=(X_val_scaled, y_val))

    # Predict on the test data using the trained DNN model
    y_pred_dnn = model_dnn.predict(X_test_scaled)

    # Calculate R-squared on the test data for DNN
    r2_dnn = r2_score(y_test, y_pred_dnn)
    print(f"DNN Test R-squared: {r2_dnn}")
    print(f"Regression Test R-squared: {r2_lin}")
    print(f"Time: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    save_model(model_dnn, "dnn_model3.h5")

    # Plot training vs validation loss over epochs
    plt.plot(history_dnn.history['loss'], label='Training Loss')
    plt.plot(history_dnn.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    


# Example: Integrating with previous model training
if __name__ == "__main__":
    # example code for training the model
    train_model(layers=[30,8], lr=.01, data_path='/Users/danielvennemeyer/Workspace/Deep Learning/cancer_reg.csv')

    # X_train_scaled, y_train_synthetic, X_val_scaled, y_val, X_test_scaled, y_test = preprocess_data(data_path='/Users/danielvennemeyer/Workspace/Deep Learning/cancer_reg.csv')
    # y_pred_dnn, r2_dnn = test_model(X_test_scaled, y_test, model_name="/Users/danielvennemeyer/Workspace/Deep Learning/dnn_model1.h5")
    
