"""
Design machine learning application which follows below steps as

Step 1:
Get Data
Load data from WinePredictor.csv file into python application.

Step 2:
Clean, Prepare and Manipulate data
As we want to use the above data into machine learning application we have prepare that in the format which is accepted by the algorithms.

Step 3:
Step Train Data
Now we want to train our data for that we have to select the Machine learning algorithm.
For that we select K Nearest Neighbour algorithm.
use fit method for training purpose.
For training use 70% dataset and for testing purpose use 30% dataset.

Step 4:
Test Data
After successful training now we can test our trained data by passing some value of
wether and temperature.
As we are using KNN algorithm use value of K as 3. After providing the values check the result and display on screen.
Result may be Yes or No.

Step 5:
Calculate Accuracy
Write one function as CheckAccuracy() which calculate the accuracy of our algorithm. For calculating the accuracy divide the dataset into two equal parts as Training data and Testing data.
Calculate Accuracy by changing value of K.
Before designing the application first consider all features of data set.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Get Data
def get_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Clean, Prepare and Manipulate data
def clean_prepare_data(data):
    X = data.drop('Class', axis=1)  # Features
    y = data['Class']  # Target variable
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Step 3: Train Data
def train_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Initialize KNN classifier with K=3
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    return knn, X_test, y_test

# Step 4: Test Data
def test_knn(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# Step 5: Calculate Accuracy
def check_accuracy(X, y, k_values, test_size=0.5):
    accuracies = []
    for k in k_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((k, accuracy))
    
    return accuracies

# Main function to run the application
if __name__ == "__main__":
    # Step 1: Get Data
    file_path = 'WinePredictor.csv'
    data = get_data(file_path)
    
    # Step 2: Clean, Prepare and Manipulate data
    X, y = clean_prepare_data(data)
    
    # Step 3: Train Data
    knn_model, X_test, y_test = train_knn(X, y)
    
    # Step 4: Test Data
    predictions = test_knn(knn_model, X_test)
    print("Predictions:", predictions)
    
    # Step 5: Calculate Accuracy
    k_values_to_test = [3]
    accuracy_results = check_accuracy(X, y, k_values_to_test)
    
    for k, accuracy in accuracy_results:
        print(f"Accuracy for K={k}: {accuracy*100:.2f}%")
